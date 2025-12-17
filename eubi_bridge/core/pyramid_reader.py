"""
Reader for NGFF/Zarr pyramid datasets.
"""

import fsspec
import numpy as np
from typing import Any, Optional

from dask import delayed
import dask, zarr
import dask.array as da

from eubi_bridge.ngff.multiscales import Pyramid
from eubi_bridge.utils.logging_config import get_logger
from eubi_bridge.core.reader_interface import ImageReader

logger = get_logger(__name__)


class NGFFReader(ImageReader):
    """
    Reader for NGFF (Next-Generation File Format) Zarr datasets.
    
    NGFF is the OME's standard for hierarchical image data in Zarr format,
    including multi-scale (pyramid) representations.
    """
    
    def __init__(self, path: str, pyr: Pyramid, aszarr: bool = False, **kwargs):
        """
        Initialize NGFF reader.
        
        Parameters
        ----------
        path : str
            Path to the Zarr group.
        pyr : Pyramid
            Initialized Pyramid object.
        aszarr : bool, default False
            Whether to return zarr arrays or converted 5D arrays.
        """
        self._path = path
        self.pyr = pyr
        self.aszarr = aszarr
        self.series = 0
        self._set_series_path()
        # NGFF stores pyramids with scales, typically represented as series
        self.n_scenes_val = 1  # NGFF typically has one pyramid per group
    
    @property
    def path(self) -> str:
        """Path to the Zarr group."""
        return self._path
    
    @property
    def series_path(self) -> str:
        """Current series identifier."""
        return self._series_path
    
    @property
    def n_scenes(self) -> int:
        """Number of scenes (typically 1 for NGFF)."""
        return self.n_scenes_val
    
    @property
    def n_tiles(self) -> int:
        """NGFF doesn't support tiles."""
        return 1
    
    def _set_series_path(self) -> None:
        """Update series path."""
        self._series_path = self._path + f'_{self.series}'
    
    def set_scene(self, scene_index: int) -> None:
        """Set the current scene (no-op for NGFF, only one scene)."""
        if scene_index != 0:
            logger.warning("NGFF typically has only one scene. Using scene 0.")
        self.series = 0
        self._set_series_path()
    
    def set_tile(self, tile_index: int) -> None:
        """No-op for NGFF (no tile support)."""
        if tile_index != 0:
            logger.warning("NGFF does not support tiles. Ignoring set_tile().")
    
    def get_image_dask_data(self, **kwargs) -> da.Array:
        """Get base image layer as dask array."""
        try:
            if self.aszarr:
                return self.pyr.layers['0']
            else:
                return self.pyr.base_array
        except Exception as e:
            raise RuntimeError(f"Failed to read pyramid data: {str(e)}") from e


def read_pyramid(input_path: str, aszarr: bool = False, **kwargs) -> NGFFReader:
    """
    Read an NGFF/Zarr pyramid dataset.
    
    Parameters
    ----------
    input_path : str
        Path to the Zarr group.
    aszarr : bool, default False
        Whether to return zarr arrays or converted 5D arrays.
    **kwargs
        Additional keyword arguments (unused).
        
    Returns
    -------
    NGFFReader
        A reader instance implementing the ImageReader interface.
    """
    pyr = Pyramid(input_path)
    return NGFFReader(input_path, pyr, aszarr=aszarr, **kwargs)
