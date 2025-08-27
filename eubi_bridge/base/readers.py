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


def read_tiff_aszarr(input_path, **kwargs):
    import tifffile, zarr
    tiff_file = tifffile.TiffFile(input_path)
    tiffzarrstore = tiff_file.aszarr()
    array = zarr.open(tiffzarrstore,
                      mode='r',
                      )
    return array

def read_tiff(input_path, **kwargs):
    if kwargs.get('skip_dask', False):
        return read_tiff_aszarr(input_path, **kwargs)
    from bioio_tifffile.reader import Reader as reader  # pip install bioio-tifffile --no-deps
    kwargs['chunk_dims'] = 'YX'
    img = reader(input_path, **kwargs)
    dimensions_present = img.standard_metadata.dimensions_present
    dimensions_to_read = 'TCZYX'
    if 'S' in dimensions_present and 'C' in dimensions_to_read and 'S' not in dimensions_to_read:
        dimensions_to_read = dimensions_to_read.replace('C', 'S')
    im = img.get_image_dask_data(dimensions_to_read)
    return im

@delayed
def read_single_image_asarray(input_path,
                              use_bioformats_readers = False,
                              **kwargs
                              ):
    """
    Reads a single image file with Dask and returns the array.

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

    if input_path.endswith('.zarr'):
        reader = Pyramid().from_ngff
    else:
        if use_bioformats_readers:
            from bioio_bioformats.reader import Reader as reader
        elif input_path.endswith(('ome.tiff', 'ome.tif')):
            from bioio_ome_tiff.reader import Reader as reader # pip install bioio-ome-tiff --no-deps
        elif input_path.endswith(('.tif', '.tiff', '.lsm')):
            reader = read_tiff
            reader_kwargs.update(**kwargs)
        elif input_path.endswith('.czi'):
            from eubi_bridge.base.czi_reader import read_czi as reader
            reader_kwargs = dict(
                as_mosaic = False,
                view_index = 0,
                phase_index = 0,
                illumination_index = 0,
                scene_index = 0,
                rotation_index = 0,
                mosaic_tile_index = 0,
                sample_index = 0
            )
            for k, v in reader_kwargs.items():
                if k in kwargs:
                    reader_kwargs[k] = kwargs[k]
        elif input_path.endswith('.lif'):
            from bioio_lif.reader import Reader as reader
        elif input_path.endswith('.nd2'):
            from bioio_nd2.reader import Reader as reader
        elif input_path.endswith(('.png','.jpg','.jpeg')):
            from bioio_imageio.reader import Reader as reader
        else:
            from bioio_bioformats.reader import Reader as reader
    verbose = kwargs.get('verbose', False)
    if verbose:
        logger.info(f"Reading with {reader.__qualname__}.")
    print(reader_kwargs)
    im = reader(input_path, **reader_kwargs)
    if isinstance(im, da.Array):
        assert im.ndim == 5
        return im
    if isinstance(im, zarr.Array):
        return im
    if hasattr(im, 'set_scene'):
        im.set_scene(kwargs.get('scene_index', 0))
        arr = im.get_image_dask_data('TCZYX')
    elif hasattr(im, 'base_array'):
        arr = im.base_array
    else:
        raise Exception(f"Unknown reader: {reader.__qualname__}")
    if arr.ndim > 5:
        arr = arr[0].transpose((0, 4, 1, 2, 3))
    return arr

def get_metadata_reader_by_path(input_path, **kwargs):
    if input_path.endswith(('ome.tiff', 'ome.tif')):
        from bioio_ome_tiff.reader import Reader as reader # pip install bioio-ome-tiff --no-deps
    elif input_path.endswith(('.tif', '.tiff', '.lsm')):
        from bioio_tifffile.reader import Reader as reader
    elif input_path.endswith('.czi'):
        from bioio_czi.reader import Reader as reader
    elif input_path.endswith('.lif'):
        from bioio_lif.reader import Reader as reader
    elif input_path.endswith('.nd2'):
        from bioio_nd2.reader import Reader as reader
    elif input_path.endswith(('.png','.jpg','.jpeg')):
        from bioio_imageio.reader import Reader as reader
    else:
        from bioio_bioformats.reader import Reader as reader
    return reader

def read_metadata_via_bioio_bioformats(input_path, **kwargs):
    from bioio_bioformats.reader import Reader
    series = kwargs.get('series', None)
    try:
        img = Reader(input_path)
        if series is not None:
            img.set_scene(series)
        omemeta = img.ome_metadata
    except FileNotFoundError as e:
        if ".jgo" in str(e):
            raise RuntimeError("JGO cache may be corrupted. Run `rm -rf ~/.jgo/` and retry.") from e
    return omemeta

def read_metadata_via_bfio(input_path, **kwargs):
    from bfio import BioReader
    try:
        omemeta = BioReader(input_path, backend='bioformats').metadata
    except FileNotFoundError as e:
        if ".jgo" in str(e):
            raise RuntimeError("JGO cache may be corrupted. Run `rm -rf ~/.jgo/` and retry.") from e
    return omemeta

def read_metadata_via_extension(input_path, **kwargs):
    Reader = get_metadata_reader_by_path(input_path)
    series = kwargs.get('series', None)
    try:
        img = Reader(input_path)
        if series is not None:
            img.set_scene(series)
        omemeta = img.ome_metadata
    except FileNotFoundError as e:
        if ".jgo" in str(e):
            raise RuntimeError("JGO cache may be corrupted. Run `rm -rf ~/.jgo/` and retry.") from e
    return omemeta
