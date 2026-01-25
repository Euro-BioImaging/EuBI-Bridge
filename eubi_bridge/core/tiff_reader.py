"""
Readers for TIFF and OME-TIFF files.

Supports both native tifffile-based reading and bioio-tifffile-based reading
for metadata-rich TIFF files.
"""

from typing import Any, Optional

import dask
import dask.array as da
import fsspec
import numpy as np
import zarr
from dask import delayed

from eubi_bridge.core.reader_interface import ImageReader
from eubi_bridge.ngff.multiscales import Pyramid
from eubi_bridge.utils.logging_config import get_logger
from eubi_bridge.external.dyna_zarr.dynamic_array import DynamicArray

logger = get_logger(__name__)


readable_formats = ('.ome.tiff', '.ome.tif', '.czi', '.lif',
                    '.nd2', '.tif', '.tiff', '.lsm',
                    '.png', '.jpg', '.jpeg')


class TIFFDynaZarrReader(ImageReader):
    """
    High-performance TIFF reader using dyna_zarr library.
    
    Wraps tifffile's zarr interface in a DynamicArray with automatic
    dimension normalization to TCZYX format.
    
    Key features:
    - Returns DynamicArray via get_image_dask_data() (name kept for compatibility)
    - Automatically expands and transposes dimensions to TCZYX
    - Lazy operations - dimension transforms applied without loading data
    - High-performance parallel I/O when writing
    """
    
    def __init__(self, path: str, tiff_file: Any, **kwargs):
        """
        Initialize TIFF reader with dyna_zarr backend.
        
        Parameters
        ----------
        path : str
            Path to the TIFF file.
        tiff_file : tifffile.TiffFile
            Opened TiffFile object.
        **kwargs
            Additional keyword arguments.
        """
        self._path = path
        self.tiff_file = tiff_file
        self.series = 0
        self._set_series_path()
        
        # Set initial scene to populate data
        if self.n_scenes > 0:
            self.set_scene(0)
    
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
        """
        Set the current series and prepare normalized TCZYX array.
        
        This method:
        1. Reads the TIFF series as zarr array
        2. Determines dimension order from TIFF metadata
        3. Constructs DynamicArray wrapper
        4. Applies lazy expand_dims and transpose to normalize to TCZYX
        
        Parameters
        ----------
        scene_index : int
            Index of the scene to select (0-based).
            
        Raises
        ------
        IndexError
            If scene_index is out of range.
        """
        if scene_index < 0 or scene_index >= self.n_scenes:
            raise IndexError(f"Scene index {scene_index} out of range [0, {self.n_scenes})")
        
        # Import dyna_zarr operations
        from eubi_bridge.external.dyna_zarr.dynamic_array import DynamicArray
        from eubi_bridge.external.dyna_zarr import operations
        
        self.series = scene_index
        self.tiff_file_series = self.tiff_file.series[scene_index]
        self._set_series_path()
        
        # Get zarr array via tifffile's aszarr (existing functionality)
        tiffzarrstore = self.tiff_file_series.aszarr()
        zarr_array = zarr.open(tiffzarrstore, mode='r')
        
        # Wrap in DynamicArray
        dyna_array = DynamicArray(zarr_array)
        
        # Get dimension information from TIFF series
        # tiff_file_series.axes contains dimension order (e.g., 'YX', 'ZYX', 'TZYX', etc.)
        tiff_axes = self.tiff_file_series.axes.upper()
        logger.debug(f"TIFF axes for series {scene_index}: {tiff_axes}")        
        #print(f"[set_scene] Initial TIFF axes: {tiff_axes}, shape: {dyna_array.shape}")
        
        # Preprocess axes to handle 'S' dimension (sample → channel)
        processed_axes, processed_shape = self._preprocess_tiff_axes(tiff_axes, dyna_array.shape)
        #print(f"[set_scene] After preprocess: axes={processed_axes}, shape={processed_shape}")
        
        # If shape changed (S dimension removed), we need to reshape the array
        if processed_shape != dyna_array.shape:
            # Remove the spurious 'S' dimension by slicing
            s_index = tiff_axes.index('S')
            slices = tuple(0 if i == s_index else slice(None) for i in range(len(tiff_axes)))
            dyna_array = dyna_array[slices]
            logger.debug(f"Removed spurious 'S' dimension, new shape: {dyna_array.shape}")
            #print(f"[set_scene] After slicing: shape={dyna_array.shape}")
        
        # Normalize to TCZYX using lazy operations
        #print(f"[set_scene] Calling _normalize_to_tczyx with axes={processed_axes}, array shape={dyna_array.shape}")
        dyna_array = self._normalize_to_tczyx(dyna_array, processed_axes)
        #print(f"[set_scene] After normalize_to_tczyx: shape={dyna_array.shape}")
        
        # Store the normalized array
        self._dyna_array = dyna_array
        
        # Create mock dims object for compatibility (if needed by downstream code)
        class MockDims:
            """Temporary dims container for compatibility."""
            def __init__(self, shape_dict):
                self.name = 'MockDims'
                for axis, size in shape_dict.items():
                    setattr(self, axis, size)
        
        # Build shape dict from normalized TCZYX
        shape_dict = {
            'T': self._dyna_array.shape[0],
            'C': self._dyna_array.shape[1],
            'Z': self._dyna_array.shape[2],
            'Y': self._dyna_array.shape[3],
            'X': self._dyna_array.shape[4],
        }
        
        self.img = type('MockImg', (), {'dims': MockDims(shape_dict)})()
    
    def _preprocess_tiff_axes(self, tiff_axes: str, array_shape: tuple) -> tuple:
        """
        Preprocess TIFF axes to handle tifffile's 'S' (sample) dimension.
        
        tifffile sometimes labels the channel dimension as 'S' instead of 'C'.
        This method normalizes the axes string:
        - If both 'C' and 'S' exist: Remove 'S' dimension (it's spurious)
        - If only 'S' exists (no 'C'): Replace 'S' with 'C' (it's the channel dimension)
        
        Parameters
        ----------
        tiff_axes : str
            Original axis string from TIFF metadata (e.g., 'SCYX', 'SYX', 'CZYX')
        array_shape : tuple
            Shape of the array (used to remove spurious dimensions)
            
        Returns
        -------
        tuple
            (processed_axes, processed_shape) with 'S' handled appropriately
            
        Examples
        --------
        'SCYX' with both C and S → 'CZYX' (remove S)
        'SYX' with only S → 'CYX' (replace S with C)
        'CZYX' with no S → 'CZYX' (unchanged)
        """
        has_c = 'C' in tiff_axes
        has_s = 'S' in tiff_axes
        
        if has_s and has_c:
            # Both exist: Remove 'S' dimension (it's spurious)
            logger.debug(f"Found both 'C' and 'S' in axes '{tiff_axes}', removing 'S'")
            s_index = tiff_axes.index('S')
            
            # Remove 'S' from axes
            processed_axes = tiff_axes.replace('S', '')
            
            # Remove corresponding dimension from shape
            processed_shape = tuple(s for i, s in enumerate(array_shape) if i != s_index)
            
            logger.debug(f"Preprocessed axes: '{tiff_axes}' → '{processed_axes}'")
            logger.debug(f"Preprocessed shape: {array_shape} → {processed_shape}")
            
            return processed_axes, processed_shape
            
        elif has_s and not has_c:
            # Only 'S' exists: Replace with 'C' (it's actually the channel dimension)
            logger.debug(f"Found 'S' but no 'C' in axes '{tiff_axes}', treating 'S' as 'C'")
            processed_axes = tiff_axes.replace('S', 'C')
            
            logger.debug(f"Preprocessed axes: '{tiff_axes}' → '{processed_axes}'")
            
            return processed_axes, array_shape
            
        else:
            # No 'S' dimension: Return unchanged
            return tiff_axes, array_shape
    
    def _normalize_to_tczyx(self, dyna_array: 'DynamicArray', tiff_axes: str) -> 'DynamicArray':
        """
        Normalize array dimensions to TCZYX format using lazy operations.
        
        This method uses dyna_zarr's operations.expand_dims() and operations.transpose()
        to transform arrays from their native TIFF dimension order to the standard
        TCZYX format expected by the rest of the pipeline.
        
        All operations are lazy - no data is loaded during this process.
        
        Parameters
        ----------
        dyna_array : DynamicArray
            Input array with native TIFF dimensions
        tiff_axes : str
            Dimension order string from TIFF metadata (e.g., 'YX', 'CZYX', 'TZCYX')
            
        Returns
        -------
        DynamicArray
            Array with dimensions normalized to TCZYX order
            
        Examples
        --------
        Input: 'YX' (2D) → expand to 'TCZYX' (add T, C, Z as singleton dimensions)
        Input: 'ZYX' (3D) → expand to 'TCZYX' (add T, C as singleton dimensions)
        Input: 'CZYX' → expand to 'TCZYX' (add T as singleton dimension)
        Input: 'TZYX' → expand to 'TCZYX' (add C as singleton dimension)
        Input: 'TZCYX' → transpose to 'TCZYX'
        """
        from eubi_bridge.external.dyna_zarr import operations
        
        current_axes = tiff_axes
        target_axes = 'TCZYX'
        
        #print(f"[_normalize_to_tczyx] Starting with axes={current_axes}, shape={dyna_array.shape}")
        
        # Step 1: Add missing dimensions using expand_dims
        # Build list of dimensions to add with their target positions
        dims_to_add = []
        for target_dim in target_axes:
            if target_dim not in current_axes:
                target_pos = target_axes.index(target_dim)
                dims_to_add.append((target_dim, target_pos))
        
        #print(f"[_normalize_to_tczyx] Dimensions to add: {dims_to_add}")
        
        # Sort by position (reverse order) to maintain correct indices during insertion
        dims_to_add.sort(key=lambda x: x[1], reverse=True)
        
        # Expand dimensions one at a time, updating axes string
        for target_dim, target_pos in dims_to_add:
            # Calculate where to insert in current_axes based on target position
            # We need to find the correct insertion point in the current order
            insert_pos = 0
            for i, ax in enumerate(target_axes[:target_pos]):
                if ax in current_axes:
                    insert_pos = current_axes.index(ax) + 1
            
            #print(f"[_normalize_to_tczyx] Expanding {target_dim} at position {insert_pos}")
            # Expand at the calculated position
            dyna_array = operations.expand_dims(dyna_array, axis=insert_pos)
            #print(f"[_normalize_to_tczyx] After expanding {target_dim}: shape={dyna_array.shape}")
            
            # Update current_axes to reflect the new dimension
            current_axes = current_axes[:insert_pos] + target_dim + current_axes[insert_pos:]
            
            logger.debug(f"Expanded dimension {target_dim} at position {insert_pos}, new axes: {current_axes}")
        
        # Step 2: Transpose to TCZYX if needed
        if current_axes != target_axes:
            # Build permutation map
            perm = tuple(current_axes.index(dim) for dim in target_axes)
            #print(f"[_normalize_to_tczyx] Transposing from {current_axes} to {target_axes}, perm={perm}")
            dyna_array = operations.transpose(dyna_array, perm)
            #print(f"[_normalize_to_tczyx] After transpose: shape={dyna_array.shape}")
            
            logger.debug(f"Transposed from {current_axes} to {target_axes} using permutation {perm}")
            current_axes = target_axes
        
        #print(f"[_normalize_to_tczyx] Final shape: {dyna_array.shape}, axes: {current_axes}")
        
        # Verify final shape has 5 dimensions
        if len(dyna_array.shape) != 5:
            raise RuntimeError(
                f"Dimension normalization failed: expected 5 dimensions (TCZYX), "
                f"got {len(dyna_array.shape)} with axes {current_axes}"
            )
        
        logger.debug(f"Final normalized shape (TCZYX): {dyna_array.shape}")
        return dyna_array
    
    def set_tile(self, tile_index: int) -> None:
        """No-op for TIFF (no tile support)."""
        if tile_index != 0:
            logger.warning("TIFF does not support tiles. Ignoring set_tile().")
    
    def get_image_dask_data(self, **kwargs) -> 'DynamicArray':
        """
        Get image data as DynamicArray (NOT dask array, despite the name).
        
        This method name is kept for compatibility with the ImageReader interface.
        Downstream code expects this method, so we preserve the name even though
        it returns a DynamicArray instead of a dask array.
        
        The DynamicArray provides:
        - Lazy operations (no data loaded until write)
        - Array-like interface (shape, dtype, slicing)
        - High-performance parallel I/O when passed to dyna_zarr writer
        - Dimension normalization to TCZYX already applied
        
        Returns
        -------
        DynamicArray
            Array with TCZYX dimension order, ready for writing.
            
        Raises
        ------
        RuntimeError
            If series has not been set (call set_scene first).
        """
        if not hasattr(self, '_dyna_array'):
            raise RuntimeError(
                "Array not initialized. Call set_scene() before get_image_dask_data()"
            )
        
        return self._dyna_array


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
    
    def _get_dimension_order(self) -> str:
        """
        Determine the correct dimension order for bioio reader.
        
        Handles the case where tifffile uses 'S' (sample) instead of 'C' (channel).
        
        Logic:
        - If both 'S' and 'C' exist → use 'TCZYX' (prefer 'C', ignore spurious 'S')
        - If only 'S' exists → use 'TSZYX' (treat 'S' as channel)
        - If neither exists → use 'TCZYX' (standard order)
        
        Returns
        -------
        str
            Dimension order string to use with bioio's get_image_dask_data()
        """
        dims = self.img.dims.order
        has_s = 'S' in dims
        has_c = 'C' in dims
        
        if has_s and has_c:
            # Both exist: prefer C, ignore S
            logger.debug(f"TIFF has both 'S' and 'C' dimensions. Using 'C' (ignoring 'S').")
            return 'TCZYX'
        elif has_s and not has_c:
            # Only S exists: treat as channel
            logger.debug(f"TIFF has 'S' dimension but no 'C'. Treating 'S' as channel dimension.")
            return 'TSZYX'
        else:
            # Standard case (no S, or has C)
            return 'TCZYX'
    
    def get_image_dask_data(self, **kwargs) -> da.Array:
        """
        Get image data as dask array.
        
        Returns actual dask.array (unlike TIFFDynaZarrReader which returns DynamicArray).
        Automatically handles 'S' (sample) vs 'C' (channel) dimension confusion.
        """
        try:
            dimensions_to_read = kwargs.get('dimensions_to_read', None)
            if dimensions_to_read is None:
                # Auto-detect correct dimension order
                dimensions_to_read = self._get_dimension_order()
            
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
        If True, use dyna_zarr reader for high-performance I/O (returns DynamicArray).
        If False, use bioio-tifffile for better metadata extraction (returns dask array).
    **kwargs
        Additional keyword arguments.
        
    Returns
    -------
    ImageReader
        A reader instance implementing the ImageReader interface.
        - If aszarr=True: TIFFDynaZarrReader (high-performance, returns DynamicArray)
        - If aszarr=False: TIFFBioIOReader (metadata-rich, returns dask array)
    """
    if aszarr:
        logger.info(f"Reading TIFF with dyna_zarr (high-performance): {input_path}")
        import tifffile
        img = tifffile.TiffFile(input_path)
        return TIFFDynaZarrReader(input_path, img, **kwargs)
    else:
        logger.info(f"Reading TIFF with bioio-tifffile: {input_path}")
        from bioio_tifffile.reader import Reader as reader
        kwargs['chunk_dims'] = 'YX'
        img = reader(input_path, **kwargs)
        return TIFFBioIOReader(input_path, img, **kwargs)