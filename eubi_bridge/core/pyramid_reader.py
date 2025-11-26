import fsspec
import fsspec.core
import fsspec.compression
import fsspec.spec
import numpy as np

from dask import delayed
import dask, zarr
import dask.array as da


# The block below moved to the 'ebridge.py' module in the 'to_zarr' method.
# import scyjava
# import jpype
# # IMPORTANT: jvm must be started before importing bioio_bioformats readers
# if not scyjava.jvm_started():
#     scyjava.config.endpoints.append("ome:formats-gpl:6.7.0")
#     scyjava.start_jvm()

def read_pyramid(input_path, aszarr, **kwargs):
    from eubi_bridge.ngff.multiscales import Pyramid
    pyr = Pyramid(input_path)
    class MockImg:
        def __init__(self, pyr, path, aszarr):
            if aszarr:
                self.pyr = pyr
            else:
                self.pyr = pyr.to5D()
            self.path = path
            self.set_scene(0)
            self._set_series_path()
            self.n_scenes = 1
        def _set_series_path(self): # placeholder
            # self.series_path = self.path
            self.series_path = self.path + f'_{self.series}'

        def get_image_dask_data(self, *args, **kwargs): ### This does not have to be dask!!!
            try:
                if aszarr:
                    return self.pyr.layers['0']
                else:
                    return self.pyr.base_array
            except Exception as e:
                raise RuntimeError(f"Failed to read image data: {str(e)}") from e
        def set_scene(self, scene_index: int): ### For the moment, this is a placeholder.
            self.series = 0
            if scene_index != 0:
                print(f"Only scene 0 is supported for now. Falling back to scene 0.")
    mock = MockImg(pyr, input_path, aszarr)
    return mock
