"""
Reader for HDF5 files.
"""

import fsspec
import fsspec.core
import fsspec.compression
import fsspec.spec
import numpy as np
from typing import Any, Optional

from dask import delayed
import dask, zarr
import dask.array as da

from eubi_bridge.utils.logging_config import get_logger
from eubi_bridge.core.reader_interface import ImageReader

logger = get_logger(__name__)


class H5Reader(ImageReader):
    """
    Reader for HDF5 files.
    
    HDF5 files can contain multiple datasets. This reader provides access to
    them via a scene-based interface.
    """
    
    def __init__(self, path: str, h5file: Any, **kwargs):
        """
        Initialize H5 reader.
        
        Parameters
        ----------
        path : str
            Path to the HDF5 file.
        h5file : h5py.File
            Opened HDF5 file handle.
        """
        self._path = path
        self.h5file = h5file
        self.series = 0
        self._set_series_path()
    
    @property
    def path(self) -> str:
        """Path to the HDF5 file."""
        return self._path
    
    @property
    def series_path(self) -> str:
        """Current series identifier."""
        return self._series_path
    
    @property
    def n_scenes(self) -> int:
        """Number of datasets in the file."""
        return len(list(self.h5file.keys()))
    
    @property
    def n_tiles(self) -> int:
        """HDF5 doesn't support tiles."""
        return 1
    
    def _set_series_path(self) -> None:
        """Update series path."""
        self._series_path = self._path + f'_{self.series}'
    
    def set_scene(self, scene_index: int) -> None:
        """Set the current dataset/scene."""
        if scene_index < 0 or scene_index >= self.n_scenes:
            raise IndexError(f"Scene index {scene_index} out of range [0, {self.n_scenes})")
        self.series = scene_index
        dset_name = list(self.h5file.keys())[scene_index]
        ds = self.h5file[dset_name]
        self._attrs = dict(ds.attrs)
        self._set_series_path()
    
    def set_tile(self, tile_index: int) -> None:
        """No-op for HDF5 (no tile support)."""
        if tile_index != 0:
            logger.warning("HDF5 does not support tiles. Ignoring set_tile().")
    
    def get_image_dask_data(self, **kwargs) -> da.Array:
        """Get image data as dask array."""
        try:
            dset_name = list(self.h5file.keys())[self.series]
            ds = self.h5file[dset_name]
            array = da.from_array(ds, **kwargs)
            return array
        except Exception as e:
            raise RuntimeError(f"Failed to read image data from {self._path}: {str(e)}") from e


def read_h5(input_path: str, **kwargs) -> H5Reader:
    """
    Read an HDF5 file.
    
    Parameters
    ----------
    input_path : str
        Path to the HDF5 file.
    **kwargs
        Additional keyword arguments (unused).
        
    Returns
    -------
    H5Reader
        A reader instance implementing the ImageReader interface.
    """
    import h5py
    img = h5py.File(input_path, 'r')
    return H5Reader(input_path, img, **kwargs)


