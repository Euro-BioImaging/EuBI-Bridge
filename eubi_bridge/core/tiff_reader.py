import fsspec
import numpy as np

from dask import delayed
import dask, zarr
from eubi_bridge.ngff.multiscales import Pyramid

from eubi_bridge.utils.logging_config import get_logger

# import logging, warnings

logger = get_logger(__name__)


readable_formats = ('.ome.tiff', '.ome.tif', '.czi', '.lif',
                    '.nd2', '.tif', '.tiff', '.lsm',
                    '.png', '.jpg', '.jpeg')


# TODO: Make MockImg objects separate classes.


# path = f"/home/oezdemir/PycharmProjects/TIM2025/data/example_images1/multichannel_timeseries/Channel1-T0003.tif"
# import tifffile, zarr
#
# img = tifffile.TiffFile(path)
# s = img.series[0]
# z = s.aszarr()


def read_tiff_with_zarr(input_path, **kwargs):
    import tifffile, zarr
    img = tifffile.TiffFile(input_path)
    class MockImg:
        def __init__(self,
                     img,
                     path):
            self.tiff_file = img
            self.path = path
            self.set_scene(0)
            self._set_series_path()
        @property
        def n_scenes(self):
            return len(self.tiff_file.series)
        def _set_series_path(self):
            self.series_path = self.path + f'_{self.series}'
        def get_image_dask_data(self, *args, **kwargs):
            ### This is not actually dask!!!

            self.tiffzarrstore = self.tiff_file_series.aszarr()
            try:
                array = zarr.open(self.tiffzarrstore,
                                  mode='r'
                                  )
                return array
            except Exception as e:
                raise RuntimeError(f"Failed to read image data: {str(e)}") from e
        def set_scene(self, scene_index: int):
            class MockDims: # Transfer this to Pyramid objects, too.
                def __init__(self):
                    self.name = 'MockDims'
            dims = MockDims()
            self.series = scene_index
            self.tiff_file_series = self.tiff_file.series[scene_index]
            self._set_series_path()
            self.img = self.tiff_file
            for char in self.tiff_file_series.axes:
                setattr (dims, char, self.tiff_file_series.shape[self.tiff_file_series.axes.index(char)])
            self.img.dims = dims
    mock = MockImg(img, input_path)
    return mock


# path = f"/home/oezdemir/PycharmProjects/TIM2025/data/example_images1/pff/ftsz3.tif"
# tz = read_tiff_with_zarr(path)


def read_tiff_with_bioio(input_path, **kwargs):
    from bioio_tifffile.reader import Reader as reader  # pip install bioio-tifffile --no-deps
    kwargs['chunk_dims'] = 'YX'
    img = reader(input_path, **kwargs)
    dimensions_present = img.standard_metadata.dimensions_present
    dimensions_to_read = 'TCZYX'
    class MockImg:
        def __init__(self,
                     img,
                     path,
                     ):
            self.img = img
            self.path = path
            self.set_scene(0)
            self._set_series_path()
        @property
        def n_scenes(self):
            return len(self.img.scenes)
        def _set_series_path(self):
            self.series_path = self.path + f'_{self.series}'
        # def get_image_dask_data(self, *args, **kwargs):
        #     try:
        #         dimensions_present = self.img.standard_metadata.dimensions_present
        #         dimensions_to_read = kwargs.get('dimensions_to_read', 'TCZYX')
        #         if 'S' in dimensions_present and 'C' in dimensions_to_read and 'C' not in dimensions_present:
        #             dimensions_to_read = dimensions_to_read.replace('C', 'S')
        #         return self.img.get_image_dask_data(dimensions_to_read)
        #     except Exception as e:
        #         raise RuntimeError(f"Failed to read image data: {str(e)}") from e
        def get_image_dask_data(self, *args, **kwargs):
            # if self.is_ometiff:
            #     return self._get_image_dask_data_from_ometiff(*args, **kwargs)
            try:
                dimensions_to_read = kwargs.get('dimensions_to_read', 'TCZYX')
                return self.img.get_image_dask_data(dimensions_to_read)
            except Exception as e:
                raise RuntimeError(f"Failed to read image data: {str(e)}") from e
        def set_scene(self, scene_index: int):
            self.series = scene_index
            self.img.set_scene(scene_index)
            self._set_series_path()
    mock = MockImg(img, input_path)
    return mock

def read_tiff_image(input_path, aszarr = True, **kwargs):
    if aszarr:
        logger.info(f"Actual reader: tifffile.")
        return read_tiff_with_zarr(input_path, **kwargs)
    else:
        logger.info(f"Actual reader: bioio_tifffile.")
        return read_tiff_with_bioio(input_path, **kwargs)