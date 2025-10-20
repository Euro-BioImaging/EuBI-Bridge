import fsspec
import fsspec.core
import fsspec.compression
import fsspec.spec
import numpy as np

from dask import delayed
import dask, zarr
import dask.array as da


def read_h5(input_path, **kwargs):
    path = input_path
    import h5py
    img = h5py.File(path)
    class MockImg:
        def __init__(self,
                     img,
                     path):
            self.img = img
            self.path = path
            self.set_scene(0)
            self._set_series_path()
        @property
        def n_scenes(self):
            return len(list(self.img.keys()))
        def _set_series_path(self): # placeholder
            self.series_path = self.path + f'_{self.series}'
        def set_scene(self, scene_index: int):
            self.series = scene_index
            dset_name = list(self.img.keys())[scene_index]
            ds = self.img[dset_name]
            self._attrs = dict(ds.attrs)
            self._set_series_path()
        def get_image_dask_data(self, *args, **kwargs): ### This does not have to be dask!!!
            try:
                dset_name = list(self.img.keys())[self.series]
                ds = self.img[dset_name]
                array = da.from_array(ds, *args, **kwargs)
                return array
            except Exception as e:
                raise RuntimeError(f"Failed to read image data: {str(e)}") from e
    mock = MockImg(img, input_path)
    return mock


