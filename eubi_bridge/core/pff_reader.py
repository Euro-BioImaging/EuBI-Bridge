import fsspec
import fsspec.core
import fsspec.compression
import fsspec.spec
import numpy as np

from dask import delayed
import dask, zarr
import dask.array as da
from eubi_bridge.ngff.multiscales import Pyramid

from bioio_bioformats import utils

# The block below moved to the 'ebridge.py' module in the 'to_zarr' method.
# import scyjava
# import jpype
# # IMPORTANT: jvm must be started before importing bioio_bioformats readers
# if not scyjava.jvm_started():
#     scyjava.config.endpoints.append("ome:formats-gpl:6.7.0")
#     scyjava.start_jvm()


readable_formats = ('.ome.tiff', '.ome.tif', '.czi', '.lif',
                    '.nd2', '.tif', '.tiff', '.lsm',
                    '.png', '.jpg', '.jpeg')


def read_pff(input_path,
            **kwargs
            ):
    """
    Reads a single image file.

    Parameters
    ----------
    input_path : str
        Path to the image file.
    **kwargs : dict
        Additional keyword arguments, such as `verbose` and `scene`.

    Returns
    -------
    arr : dask.array.Array
        The image array.
    """
    from eubi_bridge.utils.logging_config import get_logger
    logger = get_logger(__name__)
    reader_kwargs = {}
    dimensions = 'TCZYX'

    if input_path.endswith(('ome.tiff', 'ome.tif')):
        from bioio_ome_tiff.reader import Reader as reader  # pip install bioio-ome-tiff --no-deps
    elif input_path.endswith(('.tif', '.tiff')):
        from eubi_bridge.core.tiff_reader import read_tiff_image as reader
        reader_kwargs.update(**kwargs)
        reader_kwargs['aszarr'] = aszarr
    elif input_path.endswith('.czi'):
        from eubi_bridge.core.czi_reader import read_czi as reader
        reader_kwargs = dict(
            as_mosaic=False,
            view_index=0,
            phase_index=0,
            illumination_index=0,
            scene_index=0,
            rotation_index=0,
            mosaic_tile_index=0,
            sample_index=0
        )
        for k, v in kwargs.items():
            if k in reader_kwargs:
                kwargs[k] = kwargs[k]
    elif input_path.endswith('.lif'):
        from bioio_lif.reader import Reader as reader
    elif input_path.endswith('.nd2'):
        from bioio_nd2.reader import Reader as reader
    elif input_path.endswith(('.png', '.jpg', '.jpeg')):
        from bioio_imageio.reader import Reader as reader
    else:
        from bioio_bioformats.reader import Reader as reader
    verbose = kwargs.get('verbose', False)
    if verbose:
        logger.info(f"Reading with {reader.__qualname__}.")
    # print(reader_kwargs)
    # im = reader(input_path, **reader_kwargs)
    img = reader(input_path, **reader_kwargs)

    class MockImg:
        def __init__(self,
                     img,
                     path
                     ):
            self.img = img
            self.path = path
            self.set_scene(0)
            self._set_series_path()
        @property
        def n_scenes(self):
            return len(self.img.scenes)
        def _set_series_path(self):
            # self.series_path = os.path.splitext(self.path)[0] + f'_{self.series}' + os.path.splitext(self.path)[1]
            self.series_path = self.path + f'_{self.series}'
        def get_image_dask_data(self, *args, **kwargs):
            try:
                dimensions_to_read = 'TCZYX'
                return self.img.get_image_dask_data(dimensions_to_read)
            except Exception as e:
                raise RuntimeError(f"Failed to read image data: {str(e)}") from e
        def set_scene(self, scene_index: int):
            self.series = scene_index
            self.img.set_scene(scene_index)
            self._set_series_path()
    mock = MockImg(img, input_path)
    return mock