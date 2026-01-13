"""
TIFF File Reader with Concurrent Access Support

Provides lazy, thread-safe reading of TIFF files using tifffile's zarr bridge.
This enables efficient parallel I/O for multi-dimensional scientific TIFF files.
"""

import tifffile
import tensorstore as ts
import numpy as np
from pathlib import Path
import json
import zarr


class TiffZarrReader:
    """
    Lazy reader for TIFF files with concurrent access support.
    
    Uses tifffile's aszarr() method to expose TIFF as a zarr array,
    enabling thread-safe parallel reads. The interface mimics TensorStore
    for compatibility with existing code.
    
    Performance characteristics:
    - Opening: Very fast (~0.02s) - only reads metadata
    - Slicing: Lazy - stores slice info, defers data loading until .result() or .read().result()
    - Concurrent reads: Thread-safe via zarr's locking mechanism
    - Limitation: TIFF single-file format doesn't parallelize as well as
      multi-file Zarr format (use large regions, moderate thread counts)
    """
    def __init__(self, tiff_path, zarr_array=None, slice_key=None, parent_shape=None):
        if zarr_array is None:
            # Initial construction from path
            self.tiff_path = str(Path(tiff_path).resolve())
            self._tif = tifffile.TiffFile(self.tiff_path)
            raw_store = self._tif.aszarr()
            
            # Open with zarr - keeps it lazy
            self._zarr_array = zarr.open(raw_store, mode='r')
            self._slice_key = None  # No slicing yet
            self._parent_shape = None
            
            self.shape = tuple(self._zarr_array.shape)
            self.dtype = self._zarr_array.dtype
            self.chunks = self._zarr_array.chunks if hasattr(self._zarr_array, 'chunks') else None
        else:
            # Construction from sliced zarr array - store slice for lazy evaluation
            self.tiff_path = tiff_path
            self._tif = None
            self._zarr_array = zarr_array
            self._slice_key = slice_key  # Store the slice, don't apply it yet!
            self._parent_shape = parent_shape  # Shape before this slice was applied
            
            # Compute shape from the slice WITHOUT loading data
            if slice_key is not None:
                # Use parent_shape (previous slice's result shape) if available
                base_shape = parent_shape if parent_shape is not None else zarr_array.shape
                self.shape = self._compute_sliced_shape(base_shape, slice_key)
            else:
                self.shape = tuple(parent_shape if parent_shape is not None else zarr_array.shape)
            
            self.dtype = zarr_array.dtype
            self.chunks = None
    
    def _compute_sliced_shape(self, original_shape, key):
        """Compute the shape that would result from slicing, without actually slicing."""
        # Normalize the key to a tuple
        if not isinstance(key, tuple):
            key = (key,)
        
        # Handle chained slices - if key is (first_key, second_key), we need to think differently
        if len(key) == 2 and not isinstance(key[0], (int, slice)):
            # This is a chained slice from _combine_slices - DON'T use it for shape computation
            # Instead, just use the last slice in the chain
            # Actually, this shouldn't happen because parent_shape should already account for earlier slices
            pass
        
        # Pad with full slices if needed
        key = key + (slice(None),) * (len(original_shape) - len(key))
        
        new_shape = []
        for dim_size, idx in zip(original_shape, key):
            if isinstance(idx, slice):
                start, stop, step = idx.indices(dim_size)
                length = len(range(start, stop, step))
                new_shape.append(length)
            elif isinstance(idx, int):
                # Integer indexing removes dimension
                continue
            else:
                # For other types (arrays, etc), just use original size
                new_shape.append(dim_size)
        
        return tuple(new_shape)
    
    def __getitem__(self, key):
        """
        Lazy slicing - stores slice info without loading data.
        
        IMPORTANT: This does NOT load data! It returns a new TiffZarrReader
        with the slice stored for later evaluation when .result() is called.
        """
        # For shape computation, we only need the NEW key applied to current shape
        # For materialization, we need ALL keys applied sequentially
        # So store both: combined_key for materialization, key for shape computation
        
        if self._slice_key is not None:
            # Chain slices together for materialization
            combined_key = self._combine_slices(self._slice_key, key)
        else:
            combined_key = key
        
        # Compute shape using ONLY the new key on current shape
        # (parent_shape is passed as self.shape, which already includes previous slices)
        new_shape = self._compute_sliced_shape(self.shape, key)
        
        # Create new reader
        new_reader = TiffZarrReader(
            self.tiff_path, 
            zarr_array=self._zarr_array, 
            slice_key=combined_key,
            parent_shape=new_shape  # Pass computed shape as parent for next slice
        )
        # Override the shape with our computed one
        new_reader.shape = new_shape
        
        return new_reader
    
    def _combine_slices(self, first_key, second_key):
        """Combine two slice operations into one."""
        # For simplicity, we can just store them as a tuple
        # and apply them sequentially when needed
        # A more sophisticated implementation could optimize this
        return (first_key, second_key)
    
    def read(self):
        """
        Async-style read that returns a Future-like object.
        
        For compatibility with TensorStore-style code that uses .read().result()
        This is when data is ACTUALLY loaded from disk.
        """
        class FutureResult:
            def __init__(self, reader):
                self._reader = reader
            def result(self):
                return self._reader._materialize()
        
        return FutureResult(self)
    
    def result(self):
        """Direct result() call - THIS is when data is actually loaded from disk."""
        return self._materialize()
    
    def _materialize(self):
        """
        Actually load the data from disk by applying stored slices.
        
        IMPORTANT: This is where zarr array slicing happens, which materializes data!
        Optimized to apply slices sequentially without loading full array.
        """
        if self._slice_key is None:
            # No slicing, return full array
            return np.asarray(self._zarr_array[:])
        else:
            # Apply the stored slice(s)
            # Check if this is a chained slice (stored as tuple by _combine_slices)
            if self._is_chained_slice(self._slice_key):
                # Chained slices - apply sequentially on zarr array, then on numpy arrays
                slices_to_apply = self._flatten_chained_slices(self._slice_key)
                
                # Apply first slice on zarr array (this loads data from disk)
                result = self._zarr_array[slices_to_apply[0]]
                if not isinstance(result, np.ndarray):
                    result = np.asarray(result)
                
                # Apply remaining slices on numpy array (in-memory)
                for s in slices_to_apply[1:]:
                    result = result[s]
                
                return result
            else:
                # Single slice
                result = self._zarr_array[self._slice_key]
                if isinstance(result, np.ndarray):
                    return result
                else:
                    # Scalar or other type
                    return np.asarray(result)
    
    def _is_chained_slice(self, key):
        """Check if a key represents chained slices from _combine_slices."""
        # A chained slice is a 2-tuple where first element is not a slice/int
        if isinstance(key, tuple) and len(key) == 2:
            first = key[0]
            # If first element is itself a tuple or a chained slice, it's chained
            if isinstance(first, tuple):
                return True
        return False
    
    def _flatten_chained_slices(self, key):
        """Flatten nested chained slices into a list of slice tuples."""
        slices = []
        if self._is_chained_slice(key):
            first_key, second_key = key
            # Recursively flatten first_key if it's also chained
            if self._is_chained_slice(first_key):
                slices.extend(self._flatten_chained_slices(first_key))
            else:
                slices.append(first_key)
            # Add second_key
            slices.append(second_key)
        else:
            slices.append(key)
        return slices
    
    def __array__(self, dtype=None):
        """
        Support numpy array protocol for np.asarray() conversion.
        
        CRITICAL: This MUST call _materialize() to respect stored slices!
        Otherwise np.asarray() would load the entire array, ignoring slices.
        """
        arr = self._materialize()
        if dtype is not None:
            return arr.astype(dtype)
        return arr
    
    @property
    def spec(self):
        """Provide a spec property for compatibility with TensorStore-style code."""
        return {
            'driver': 'tiff+zarr',
            'path': self.tiff_path,
            'shape': self.shape,
            'dtype': str(self.dtype),
            'chunks': self.chunks
        }


class ArrayResult:
    """Wrapper for already-materialized numpy arrays."""
    def __init__(self, array):
        self._array = array
        self.shape = array.shape
        self.dtype = array.dtype
    
    def read(self):
        class FutureResult:
            def __init__(self, data):
                self._data = data
            def result(self):
                return self._data
        return FutureResult(self._array)
    
    def result(self):
        """Direct result() call."""
        return self._array
    
    def __array__(self, dtype=None):
        """Support numpy array protocol."""
        if dtype is not None:
            return self._array.astype(dtype)
        return self._array
    
    def __getitem__(self, key):
        return ArrayResult(self._array[key])


class ScalarResult:
    """Wrapper for scalar values."""
    def __init__(self, value):
        self._value = value
        self.shape = ()
        self.dtype = np.array(value).dtype
    
    def read(self):
        class FutureResult:
            def __init__(self, data):
                self._data = data
            def result(self):
                return self._data
        return FutureResult(self._value)
    
    def result(self):
        """Direct result() call."""
        return self._value
    
    def __array__(self, dtype=None):
        """Support numpy array protocol."""
        arr = np.array(self._value)
        if dtype is not None:
            return arr.astype(dtype)
        return arr


def read_tiff_lazy(tiff_path):
    """
    Open a TIFF file for lazy, thread-safe reading.
    
    Fast opening (~0.02s) - only reads metadata, not pixel data.
    Returns a reader that provides:
    - Lazy slicing: data only loaded on access
    - Thread-safe: multiple threads can read concurrently
    - TensorStore-compatible API: supports .read().result() pattern
    - Efficient chunked access via tifffile's zarr bridge
    
    Args:
        tiff_path: Path to the TIFF file
        
    Returns:
        TiffZarrReader with lazy, concurrent read support
    """
    return TiffZarrReader(tiff_path)
