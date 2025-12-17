"""
Abstract base classes defining the formal interface for all image readers.

This module establishes clear contracts that all image reader implementations
must follow, enabling consistent behavior across different file formats and
improving type safety and extensibility.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any
import dask.array as da


class ImageReader(ABC):
    """
    Abstract base class for all image format readers.
    
    An ImageReader is responsible for:
    - Opening and reading a specific image file format
    - Providing access to pixel data as dask arrays
    - Managing multi-scene/series datasets
    - Supporting tile-based reading (for mosaic images)
    
    All concrete implementations must provide these methods and properties.
    """
    
    # ========== Properties ==========
    
    @property
    @abstractmethod
    def path(self) -> str:
        """Path to the image file being read."""
        pass
    
    @property
    @abstractmethod
    def series_path(self) -> str:
        """
        Identifier for the current series.
        
        Used to differentiate outputs when reading multi-series files.
        Format: "{file_path}_{series_index}"
        """
        pass
    
    @property
    @abstractmethod
    def n_scenes(self) -> int:
        """Number of scenes/series available in this file."""
        pass
    
    @property
    @abstractmethod
    def n_tiles(self) -> int:
        """Number of tiles in the current scene (for mosaic images)."""
        pass
    
    # ========== Scene Management ==========
    
    @abstractmethod
    def set_scene(self, scene_index: int) -> None:
        """
        Set the current scene/series to read from.
        
        Parameters
        ----------
        scene_index : int
            Index of the scene to select (0-based).
            
        Raises
        ------
        IndexError
            If scene_index is out of range.
        """
        pass
    
    @abstractmethod
    def set_tile(self, tile_index: int) -> None:
        """
        Set the current tile to read from (for mosaic images).
        
        Parameters
        ----------
        tile_index : int
            Index of the tile to select (0-based).
            If the image doesn't support tiles, this should be a no-op.
            
        Raises
        ------
        IndexError
            If tile_index is out of range.
        """
        pass
    
    # ========== Data Access ==========
    
    @abstractmethod
    def get_image_dask_data(self, **kwargs) -> da.Array:
        """
        Get the current image as a dask array.
        
        Returns
        -------
        dask.array.Array
            The image data. Dimension order should be normalized to TCZYX
            where applicable (or subset thereof, e.g., CYX for 2D RGB).
            
        Notes
        -----
        - Data is lazy-loaded via dask, not immediately read
        - All readers should support lazy reading for memory efficiency
        - Dimension order normalization happens here
        
        Keyword Arguments
        ------------------
        dimensions_to_read : str, optional
            Which dimensions to return. Default is 'TCZYX' or supported subset.
        **kwargs : Any
            Format-specific options passed to underlying reader.
            
        Raises
        ------
        RuntimeError
            If image data cannot be accessed.
        """
        pass
