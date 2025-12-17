"""
Reader for Zeiss CZI microscopy files.
"""

import os
from typing import Any, Iterable, Optional, Union

import dask.array as da
import numpy as np

from eubi_bridge.core.reader_interface import ImageReader
from eubi_bridge.utils.logging_config import get_logger

logger = get_logger(__name__)


class CZIReader(ImageReader):
    """
    Reader for Zeiss CZI microscopy files.
    
    Supports both single-image and mosaic (tiled) reading with dimension
    mapping for non-standard dimensions (views, phases, illuminations, etc.).
    """
    
    def __init__(
        self,
        path: str,
        img: Any,
        index_map: dict,
        as_mosaic: bool = False,
        **kwargs
    ):
        """
        Initialize CZI reader.
        
        Parameters
        ----------
        path : str
            Path to the CZI file.
        img : bioio Reader
            Initialized bioio CZI reader.
        index_map : dict
            Mapping of non-standard dimensions to their indices.
        as_mosaic : bool, default False
            Whether to read as mosaic (tiled) image.
        """
        self._path = path
        self.img = img
        self.index_map = index_map
        self.as_mosaic = as_mosaic
        self.series = 0
        self.tile = 0
        self._set_series_path()
    
    @property
    def path(self) -> str:
        """Path to the CZI file."""
        return self._path
    
    @property
    def series_path(self) -> str:
        """Current series identifier."""
        return self._series_path
    
    @property
    def n_scenes(self) -> int:
        """Number of scenes in the file."""
        return len(self.img.scenes)
    
    @property
    def n_tiles(self) -> int:
        """Number of mosaic tiles in the current scene."""
        if hasattr(self.img.dims, 'M'):
            return self.img.dims.M
        elif hasattr(self.img._dims, 'M'):
            return self.img._dims.M
        else:
            return 1
    
    @property
    def scenes(self):
        """Available scenes."""
        return self.img.scenes
    
    def _set_series_path(self, add_tile_index: bool = False) -> None:
        """Update the series path based on current scene/tile."""
        if add_tile_index:
            tile_index = self.index_map.get('M', 0)
            self._series_path = self._path + f'_{self.series}' + f'_tile{self.tile}'
        else:
            self._series_path = self._path + f'_{self.series}'
    
    def set_scene(self, scene_index: int) -> None:
        """Set the current scene/series."""
        if scene_index < 0 or scene_index >= self.n_scenes:
            raise IndexError(f"Scene index {scene_index} out of range [0, {self.n_scenes})")
        self.index_map['S'] = scene_index
        self.series = scene_index
        self.img.set_scene(scene_index)
        self._set_series_path()
    
    def set_tile(self, tile_index: int) -> None:
        """Set the current mosaic tile."""
        if tile_index < 0 or tile_index >= self.n_tiles:
            raise IndexError(f"Tile index {tile_index} out of range [0, {self.n_tiles})")
        self.index_map['M'] = tile_index
        self.tile = tile_index
        self._set_series_path()
    
    def get_image_dask_data(self, **kwargs) -> da.Array:
        """Get image data as dask array with dimension order TCZYX."""
        try:
            return self.img.get_image_dask_data(
                dimension_order_out='TCZYX',
                **self.index_map
            )
        except Exception as e:
            raise RuntimeError(f"Failed to read image data from {self._path}: {str(e)}") from e


def read_czi(
    input_path: str,
    as_mosaic: bool = False,
    view_index: int = 0,
    phase_index: int = 0,
    illumination_index: int = 0,
    scene_index: Union[int, Iterable[int]] = 0,
    rotation_index: int = 0,
    mosaic_tile_index: int = 0,
    sample_index: int = 0,
    **kwargs
) -> CZIReader:
    """
    Read a CZI (Zeiss microscopy) file with specified dimension indices.

    Parameters
    ----------
    input_path : str
        Path to the CZI file.
    as_mosaic : bool, default False
        Whether to read as a mosaic (tiled) image.
    view_index : int, default 0
        Index for the view dimension (v).
    phase_index : int, default 0
        Index for the phase dimension (h).
    illumination_index : int, default 0
        Index for the illumination dimension (i).
    scene_index : int, default 0
        Index for the scene dimension (s).
    rotation_index : int, default 0
        Index for the rotation dimension (r).
    mosaic_tile_index : int, default 0
        Index for the mosaic tile dimension (m).
    sample_index : int, default 0
        Index for the sample dimension (a).
    **kwargs
        Additional keyword arguments (unused).

    Returns
    -------
    CZIReader
        A reader instance implementing the ImageReader interface.

    Raises
    ------
    FileNotFoundError
        If the input file doesn't exist.
    RuntimeError
        If the CZI file cannot be opened.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    # Import the appropriate reader
    if as_mosaic:
        from bioio_czi.pylibczirw_reader.reader import Reader
    else:
        from bioio_czi.aicspylibczi_reader.reader import Reader

    # Initialize reader and get image metadata
    try:
        img = Reader(input_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CZI file: {str(e)}") from e

    # Process non-standard dimensions
    nonstandard_dims = [
        dim.upper() for dim in img.standard_metadata.dimensions_present
        if dim.upper() not in {"X", "Y", "C", "T", "Z"}
    ]

    # Handle mosaic-specific logic
    if as_mosaic:
        if 'M' in nonstandard_dims:
            nonstandard_dims.remove('M')
        else:
            logger.warning(
                f"Mosaic tile dimension not found in {input_path}. "
                "Ignoring 'as_mosaic' parameter."
            )
            as_mosaic = False
        if mosaic_tile_index != 0:
            logger.warning(
                "Mosaic tile index is ignored when reading the entire mosaic. "
                "Set as_mosaic=False to read specific tiles."
            )

    # Map dimension indices
    czi_dim_map = {
        'V': view_index,
        'H': phase_index,
        'I': illumination_index,
        'S': scene_index,
        'R': rotation_index,
        'M': mosaic_tile_index,
        'A': sample_index
    }

    # Create index map for non-standard dimensions
    index_map = {
        dim: czi_dim_map[dim]
        for dim in nonstandard_dims
        if dim in czi_dim_map
    }
    
    return CZIReader(input_path, img, index_map, as_mosaic=as_mosaic, **kwargs)



