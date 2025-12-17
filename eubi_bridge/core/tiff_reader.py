"""
Readers for TIFF and OME-TIFF files.

Supports both native tifffile-based reading and bioio-tifffile-based reading
for metadata-rich TIFF files.
"""

import fsspec
import numpy as np

from dask import delayed
import dask, zarr
import dask.array as da
from typing import Any, Optional
from eubi_bridge.ngff.multiscales import Pyramid

from eubi_bridge.utils.logging_config import get_logger
from eubi_bridge.core.reader_interface import ImageReader

logger = get_logger(__name__)


readable_formats = ('.ome.tiff', '.ome.tif', '.czi', '.lif',
                    '.nd2', '.tif', '.tiff', '.lsm',
                    '.png', '.jpg', '.jpeg')


class TIFFZarrReader(ImageReader):
    """
    Reader for TIFF files using tifffile and zarr backing.
    
    Leverages tifffile's zarr interface for efficient reading without
    requiring bioformats.
    """
    
    def __init__(self, path: str, tiff_file: Any, **kwargs):
        """
        Initialize TIFF reader.
        
        Parameters
        ----------
        path : str
            Path to the TIFF file.
        tiff_file : tifffile.TiffFile
            Opened TiffFile object.
        """
        self._path = path
        self.tiff_file = tiff_file
        self.series = 0
        self._set_series_path()
    
    @property
    def path(self) -> str:
        """Path to the TIFF file."""
        return self._path
    
    @property
    def series_path(self) -> str:
        """Current series identifier."""
        return self._series_path
    
    @property
    def n_scenes(self) -> int:
        """Number of series in the TIFF file."""
        return len(self.tiff_file.series)
    
    @property
    def n_tiles(self) -> int:
        """TIFF doesn't support tiles."""
        return 1
    
    def _set_series_path(self) -> None:
        """Update series path."""
        self._series_path = self._path + f'_{self.series}'
    
    def set_scene(self, scene_index: int) -> None:
        """Set the current series."""
        if scene_index < 0 or scene_index >= self.n_scenes:
            raise IndexError(f"Scene index {scene_index} out of range [0, {self.n_scenes})")
        
        class MockDims:
            """Temporary dims container for compatibility."""
            def __init__(self):
                self.name = 'MockDims'
        
        dims = MockDims()
        self.series = scene_index
        self.tiff_file_series = self.tiff_file.series[scene_index]
        self._set_series_path()
        self.img = self.tiff_file
        
        # Populate dims from series axes
        for char in self.tiff_file_series.axes:
            setattr(dims, char, self.tiff_file_series.shape[self.tiff_file_series.axes.index(char)])
        self.img.dims = dims
    
    def set_tile(self, tile_index: int) -> None:
        """No-op for TIFF (no tile support)."""
        if tile_index != 0:
            logger.warning("TIFF does not support tiles. Ignoring set_tile().")
    
    def get_image_dask_data(self, **kwargs) -> da.Array:
        """Get image data as zarr-backed array."""
        try:
            self.tiffzarrstore = self.tiff_file_series.aszarr()
            array = zarr.open(self.tiffzarrstore, mode='r')
            return array
        except Exception as e:
            raise RuntimeError(f"Failed to read TIFF data: {str(e)}") from e


class TIFFBioIOReader(ImageReader):
    """
    Reader for TIFF files using bioio-tifffile.
    
    Provides better metadata extraction via bioio but may be slower than
    native tifffile reading.
    """
    
    def __init__(self, path: str, img: Any, **kwargs):
        """
        Initialize TIFF BioIO reader.
        
        Parameters
        ----------
        path : str
            Path to the TIFF file.
        img : bioio Reader
            Initialized bioio TIFF reader.
        """
        self._path = path
        self.img = img
        self.series = 0
        self._set_series_path()
    
    @property
    def path(self) -> str:
        """Path to the TIFF file."""
        return self._path
    
    @property
    def series_path(self) -> str:
        """Current series identifier."""
        return self._series_path
    
    @property
    def n_scenes(self) -> int:
        """Number of scenes."""
        return len(self.img.scenes)
    
    @property
    def n_tiles(self) -> int:
        """TIFF doesn't support tiles."""
        return 1
    
    def _set_series_path(self) -> None:
        """Update series path."""
        self._series_path = self._path + f'_{self.series}'
    
    def set_scene(self, scene_index: int) -> None:
        """Set the current series."""
        if scene_index < 0 or scene_index >= self.n_scenes:
            raise IndexError(f"Scene index {scene_index} out of range [0, {self.n_scenes})")
        self.series = scene_index
        self.img.set_scene(scene_index)
        self._set_series_path()
    
    def set_tile(self, tile_index: int) -> None:
        """No-op for TIFF (no tile support)."""
        if tile_index != 0:
            logger.warning("TIFF does not support tiles. Ignoring set_tile().")
    
    def get_image_dask_data(self, **kwargs) -> da.Array:
        """Get image data as dask array."""
        try:
            dimensions_to_read = kwargs.get('dimensions_to_read', 'TCZYX')
            return self.img.get_image_dask_data(dimensions_to_read)
        except Exception as e:
            raise RuntimeError(f"Failed to read TIFF data: {str(e)}") from e


def read_tiff_image(input_path: str, aszarr: bool = True, **kwargs) -> ImageReader:
    """
    Read a TIFF or OME-TIFF file.
    
    Parameters
    ----------
    input_path : str
        Path to the TIFF file.
    aszarr : bool, default True
        If True, use native tifffile with zarr backing.
        If False, use bioio-tifffile for better metadata.
    **kwargs
        Additional keyword arguments.
        
    Returns
    -------
    ImageReader
        A reader instance implementing the ImageReader interface.
    """
    if aszarr:
        logger.info(f"Reading TIFF with tifffile: {input_path}")
        import tifffile
        img = tifffile.TiffFile(input_path)
        return TIFFZarrReader(input_path, img, **kwargs)
    else:
        logger.info(f"Reading TIFF with bioio-tifffile: {input_path}")
        from bioio_tifffile.reader import Reader as reader
        kwargs['chunk_dims'] = 'YX'
        img = reader(input_path, **kwargs)
        return TIFFBioIOReader(input_path, img, **kwargs)
        return read_tiff_with_bioio(input_path, **kwargs)