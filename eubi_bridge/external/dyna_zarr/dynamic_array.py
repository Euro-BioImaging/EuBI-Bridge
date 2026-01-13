"""
dynamic_array: A lightweight library for lazy operations on Zarr arrays
without the overhead of task graphs.
"""

import zarr
import numpy as np
from typing import Tuple, Union, List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
from pathlib import Path
import gc
import time
import threading
import json
from queue import Queue, Empty

# Import Transform base class from operations module
from .operations import Transform
from .codecs import Codecs

        


def _maybe_collect(state, interval=5):
    """Run a full GC at most once every `interval` seconds to avoid frequent small collections."""
    last = state.get('_last_gc_time', 0)
    now = time.time()
    if now - last >= interval:
        gc.collect()
        state['_last_gc_time'] = now


class DynamicArray:
    """
    Wrapper around Zarr arrays or TensorStore arrays that enables lazy operations.
    
    Supports multiple input types:
    - zarr.Array: Wraps for lazy operations
    - TensorStore arrays: Wraps for lazy operations
    - DynamicArray: Copy constructor
    
    To read from files, use operations.read(path) instead.
    """

    def __init__(self, source: Union[zarr.Array, 'DynamicArray']):
        if isinstance(source, DynamicArray):
            # Copy constructor
            self._source = source._source
            self._zarr_array = source._zarr_array
            self._ts_array = source._ts_array
            self._is_tensorstore = source._is_tensorstore
            self._shape = source._shape
            self._chunks = source._chunks
            self._dtype = source._dtype
            self._transform = source._transform
            self._zarr_format = source._zarr_format
            self._compressor = source._compressor
            self._compressors = source._compressors
            self._shards = source._shards
            self._codecs = source._codecs
        else:
            # Wrap a real Zarr array or TensorStore array
            # Check if it's tensorstore
            if hasattr(source, 'read') and hasattr(source, 'spec'):
                # It's a tensorstore array
                self._is_tensorstore = True
                self._ts_array = source
                self._zarr_array = None
            else:
                # It's a zarr array
                self._is_tensorstore = False
                self._ts_array = None
                self._zarr_array = source
                self._extract_zarr_metadata(source)
            
            self._source = source
            self._shape = tuple(source.shape)
            self._chunks = getattr(source, 'chunks', None)
            self._dtype = source.dtype
            self._transform = None
            if not self._is_tensorstore:
                self._extract_zarr_metadata(source)
            else:
                self._zarr_format = None
                self._compressor = None
                self._compressors = None
                self._shards = None
                self._codecs = None

    def _extract_zarr_metadata(self, zarr_array):
        """Extract metadata from Zarr array, handling both v2 and v3."""
        # Detect Zarr format version
        if hasattr(zarr_array, '_version'):
            self._zarr_format = zarr_array._version
        elif hasattr(zarr_array, 'store'):
            # Try to detect from store structure
            store = zarr_array.store
            if hasattr(store, 'path'):
                store_path = Path(store.path) if isinstance(store.path, str) else store.path
                if (store_path / 'zarr.json').exists():
                    self._zarr_format = 3
                elif (store_path / '.zarray').exists():
                    self._zarr_format = 2
                else:
                    self._zarr_format = 2  # Default to v2
            else:
                self._zarr_format = 2  # Default to v2
        else:
            self._zarr_format = 2  # Default to v2

        # Extract compressor/compressors based on format
        if self._zarr_format == 3:
            # Zarr v3
            self._compressor = None
            if hasattr(zarr_array, 'metadata'):
                metadata = zarr_array.metadata
                if 'codecs' in metadata:
                    self._compressors = metadata['codecs']
                else:
                    self._compressors = None
            elif hasattr(zarr_array, 'compressors'):
                self._compressors = zarr_array.compressors
            else:
                self._compressors = None

            # Extract shards
            if hasattr(zarr_array, 'metadata') and 'shards' in zarr_array.metadata:
                self._shards = zarr_array.metadata['shards']
            elif hasattr(zarr_array, 'shards'):
                self._shards = zarr_array.shards
            else:
                self._shards = None
            
            # Convert v3 codecs to Codecs instance
            self._codecs = self._extract_codecs_from_v3(self._compressors)
        else:
            # Zarr v2
            try:
                self._compressor = zarr_array.compressor if hasattr(zarr_array, 'compressor') else None
            except (TypeError, AttributeError):
                # Handle Zarr v3 arrays that error on compressor access
                self._compressor = None
            self._compressors = None
            self._shards = None
            
            # Convert v2 compressor to Codecs instance
            if self._compressor is not None:
                try:
                    self._codecs = Codecs.from_numcodecs(self._compressor)
                except Exception:
                    # If conversion fails, use default
                    self._codecs = None
            else:
                self._codecs = None

    def _extract_codecs_from_v3(self, codecs_list):
        """Extract Codecs instance from Zarr v3 codec pipeline."""
        if codecs_list is None:
            return None
        
        # Look for compression codec in the pipeline
        for codec in codecs_list:
            codec_name = codec.get('name', '')
            config = codec.get('configuration', {})
            
            if codec_name == 'blosc':
                return Codecs(
                    compressor='blosc',
                    clevel=config.get('clevel', 5),
                    cname=config.get('cname', 'lz4'),
                    shuffle=config.get('shuffle', 1)
                )
            elif codec_name == 'zstd':
                return Codecs(
                    compressor='zstd',
                    clevel=config.get('level', 5)
                )
            elif codec_name == 'gzip':
                return Codecs(
                    compressor='gzip',
                    clevel=config.get('level', 5)
                )
            elif codec_name == 'lz4':
                return Codecs(compressor='lz4')
            elif codec_name == 'bz2':
                return Codecs(
                    compressor='bz2',
                    clevel=config.get('level', 5)
                )
        
        # No compression codec found
        return None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def chunks(self) -> Tuple[int, ...]:
        """Get the chunk shape of the underlying array."""
        if self._chunks is not None:
            return self._chunks
        # Fallback: try to get chunks from underlying zarr array
        if self._zarr_array is not None and hasattr(self._zarr_array, 'chunks'):
            return self._zarr_array.chunks
        # If still None, return None (TensorStore or unknown)
        return None

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def zarr_format(self) -> int:
        return self._zarr_format

    @property
    def compressor(self):
        """Compressor for Zarr v2."""
        return self._compressor

    @property
    def compressors(self):
        """Compressors for Zarr v3."""
        return self._compressors

    @property
    def shards(self):
        """Shards for Zarr v3."""
        return self._shards

    @property
    def codecs(self):
        """Unified compression configuration (Codecs instance)."""
        return self._codecs

    @property
    def is_tensorstore(self) -> bool:
        """True if backed by TensorStore, False if backed by zarr."""
        return self._is_tensorstore

    @property
    def array(self):
        """Get the underlying array (zarr.Array or tensorstore.TensorStore)."""
        if self._is_tensorstore:
            return self._ts_array
        else:
            return self._zarr_array

    def __getitem__(self, key):
        """
        Create a lazy slice of the array.
        For immediate execution, use .compute() method.
        """
        # Create a lazy slice transform
        transform = SliceTransform(self, key)
        return self._with_transform(transform)
    
    def compute(self):
        """
        Execute all lazy transforms and return the result as a numpy array.
        """
        if self._transform is None:
            # No transform - read entire array
            if self._is_tensorstore:
                return self._ts_array[:].read().result()
            else:
                return self._zarr_array[:]
        else:
            # Apply transformation to read all data
            full_slice = tuple(slice(None) for _ in range(len(self.shape)))
            return self._transform.read(full_slice)
    
    def _read_direct(self, key):
        """
        Internal method to read data directly without creating transforms.
        Used by Transform.read() methods to actually fetch data.
        """
        if self._transform is None:
            # Direct read from source
            if self._is_tensorstore:
                # TensorStore: call .read().result() for async operations
                return self._ts_array[key].read().result()
            else:
                # Zarr: direct indexing
                return self._zarr_array[key]
        else:
            # Apply transformation chain
            return self._transform.read(key)

    def _with_transform(self, transform):
        """
        Create a new DynamicArray with a transformation applied.
        """
        result = DynamicArray(self)
        result._transform = transform
        result._shape = transform.shape
        result._chunks = transform.chunks
        result._dtype = transform.dtype
        # Keep zarr metadata from original
        return result


# Import Transform subclasses
from .operations import SliceTransform


def slice_array(array: DynamicArray, key) -> DynamicArray:
    """
    Create a lazy slice of an array.
    """
    transform = SliceTransform(array, key)
    return array._with_transform(transform)





# Global memory pool for region buffers
class RegionBufferPool:
    """
    Thread-safe memory pool for reusing region buffers in TensorStore I/O.
    
    Benefits:
    - Reduces allocation overhead (malloc/free are expensive)
    - Improves memory locality and cache performance
    - Reduces GC pressure from frequent large allocations
    - Particularly valuable for TensorStore's C++ backend
    """
    
    def __init__(self, max_size: int = 64):
        self.pool = {}  # Key: (shape, dtype) -> List[buffers]
        self.lock = threading.Lock()
        self.max_per_key = 8  # Max buffers per shape/dtype combo
        self.hits = 0
        self.misses = 0
        self.max_size = max_size
        self.total_buffers = 0
    
    def get_buffer(self, shape: Tuple[int, ...], dtype) -> Optional[np.ndarray]:
        """Get a buffer from pool if available, otherwise return None."""
        key = (shape, str(dtype))
        
        with self.lock:
            if key in self.pool and self.pool[key]:
                buf = self.pool[key].pop()
                self.hits += 1
                self.total_buffers -= 1
                return buf
        
        self.misses += 1
        return None
    
    def return_buffer(self, buf: np.ndarray):
        """Return buffer to pool for reuse."""
        if buf is None:
            return
            
        key = (buf.shape, str(buf.dtype))
        
        with self.lock:
            # Don't exceed max buffers per key
            if key not in self.pool:
                self.pool[key] = []
            
            if len(self.pool[key]) < self.max_per_key and self.total_buffers < self.max_size:
                # Zero out buffer for safety (optional, can be removed for speed)
                # buf.fill(0)  # Comment out for max performance
                self.pool[key].append(buf)
                self.total_buffers += 1
    
    def clear(self):
        """Clear the entire pool and reset stats."""
        with self.lock:
            self.pool.clear()
            self.hits = 0
            self.misses = 0
            self.total_buffers = 0
    
    def get_stats(self):
        """Get pool statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'total_buffers': self.total_buffers,
                'unique_shapes': len(self.pool)
            }


# Global buffer pool instance
_REGION_BUFFER_POOL = RegionBufferPool(max_size=64)



