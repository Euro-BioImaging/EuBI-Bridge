"""
Advanced memory management for high-performance data processing.
Handles memory mapping, allocation optimization, and garbage collection.
"""

import os
import mmap
import psutil
import gc
import weakref
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from pathlib import Path


class MemoryManager:
    """Advanced memory manager with mmap support and optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory_limit = config.get("max_memory_mb", None)
        self.chunk_memory_mb = config.get("chunk_memory_mb", None)
        self.use_mmap = config.get("use_mmap", True)
        
        # Memory tracking
        self.allocated_memory = 0
        self.mmap_objects = weakref.WeakValueDictionary()
        self.memory_pools = {}
        
        # Auto-detect optimal settings if not specified
        if self.memory_limit is None:
            self.memory_limit = self._detect_available_memory()
        
        if self.chunk_memory_mb is None:
            self.chunk_memory_mb = self._calculate_optimal_chunk_size()
    
    def _detect_available_memory(self) -> int:
        """Detect available system memory in MB."""
        memory = psutil.virtual_memory()
        # Use 80% of available memory, leaving room for system
        available_mb = int((memory.available * 0.8) / (1024 * 1024))
        return available_mb
    
    def _calculate_optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size based on system characteristics."""
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        
        # Adaptive chunk sizing based on total memory
        if total_gb >= 64:  # High-memory system
            chunk_mb = 512
        elif total_gb >= 32:  # Medium-memory system
            chunk_mb = 256
        elif total_gb >= 16:  # Standard system
            chunk_mb = 128
        else:  # Low-memory system
            chunk_mb = 64
        
        # Consider CPU cache size if available
        try:
            # L3 cache size heuristic
            cpu_count = psutil.cpu_count()
            cache_mb = cpu_count * 8  # Estimate 8MB L3 per core
            chunk_mb = min(chunk_mb, cache_mb * 4)  # 4x cache size
        except:
            pass
        
        return max(64, chunk_mb)  # Minimum 64MB chunks
    
    async def calculate_optimal_chunks(self,
                                       shape: Tuple[int, ...],
                                       dtype: np.dtype,
                                       axes: str) -> Tuple[int, ...]:
        """Calculate optimal chunk sizes for given array shape and data type."""
        # Target chunk size in bytes
        target_chunk_bytes = self.chunk_memory_mb * 1024 * 1024
        dtype_size = np.dtype(dtype).itemsize

        print(f"Target chunk size: {target_chunk_bytes} bytes")
        print(f"Data type size: {dtype_size} bytes")
        # Start with a base chunk that fits in target memory
        total_elements = target_chunk_bytes // dtype_size
        
        # Distribute elements across dimensions optimally
        chunk_shape = list(shape)
        
        # Prioritize spatial dimensions (x, y, z) for better cache locality
        spatial_axes = {'x': axes.find('x'), 'y': axes.find('y'), 'z': axes.find('z')}
        spatial_axes = {k: v for k, v in spatial_axes.items() if v >= 0}
        
        # Calculate chunk sizes
        if spatial_axes:
            # For spatial data, optimize for spatial locality
            chunk_shape = self._optimize_spatial_chunks(
                shape, spatial_axes, total_elements, axes
            )
        else:
            # For non-spatial data, use uniform chunking
            chunk_shape = self._calculate_uniform_chunks(shape, total_elements)
        
        # Ensure minimum chunk sizes
        min_chunk = 64  # Minimum 64 elements per dimension
        chunk_shape = [max(min_chunk, c) for c in chunk_shape]
        
        # Ensure chunks don't exceed array dimensions
        chunk_shape = [min(c, s) for c, s in zip(chunk_shape, shape)]
        
        return tuple(chunk_shape)
    
    def _optimize_spatial_chunks(self, shape: Tuple[int, ...], 
                               spatial_axes: Dict[str, int], 
                               total_elements: int, axes: str) -> List[int]:
        """Optimize chunk sizes for spatial data."""
        chunk_shape = list(shape)
        
        # Get spatial dimensions
        spatial_dims = list(spatial_axes.values())
        spatial_dims.sort()  # Process in order: z, y, x typically
        
        if len(spatial_dims) >= 2:
            # For 2D/3D spatial data, prioritize x-y plane locality
            x_idx = spatial_axes.get('x', -1)
            y_idx = spatial_axes.get('y', -1)
            z_idx = spatial_axes.get('z', -1)
            
            # Calculate target spatial chunk size
            spatial_elements = int(total_elements ** (1.0 / len(spatial_dims)))
            
            if x_idx >= 0:
                chunk_shape[x_idx] = min(shape[x_idx], spatial_elements)
            if y_idx >= 0:
                chunk_shape[y_idx] = min(shape[y_idx], spatial_elements)
            if z_idx >= 0:
                # Z dimension can be smaller for better memory access
                chunk_shape[z_idx] = min(shape[z_idx], max(1, spatial_elements // 4))
        
        # For non-spatial dimensions, use smaller chunks
        for i, ax in enumerate(axes):
            if ax in ['t', 'c'] and i < len(chunk_shape):
                # Time and channel dimensions: smaller chunks
                chunk_shape[i] = min(shape[i], max(1, int(shape[i] ** 0.5)))
        
        return chunk_shape
    
    def _calculate_uniform_chunks(self, shape: Tuple[int, ...], 
                                total_elements: int) -> List[int]:
        """Calculate uniform chunk sizes across all dimensions."""
        ndim = len(shape)
        chunk_per_dim = int(total_elements ** (1.0 / ndim))
        
        chunk_shape = []
        for dim_size in shape:
            chunk_size = min(dim_size, max(1, chunk_per_dim))
            chunk_shape.append(chunk_size)
        
        return chunk_shape
    
    def create_memory_mapped_array(self, file_path: str, shape: Tuple[int, ...], 
                                 dtype: np.dtype, mode: str = 'r') -> np.ndarray:
        """Create memory-mapped array for efficient large file access."""
        if not self.use_mmap:
            # Fallback to regular file reading
            return np.fromfile(file_path, dtype=dtype).reshape(shape)
        
        try:
            # Create memory-mapped array
            mmap_array = np.memmap(file_path, dtype=dtype, mode=mode, shape=shape)
            
            # Track the mmap object
            self.mmap_objects[id(mmap_array)] = mmap_array
            
            return mmap_array
            
        except Exception as e:
            # Fallback to regular array
            print(f"Memory mapping failed for {file_path}: {e}")
            return np.fromfile(file_path, dtype=dtype).reshape(shape)
    
    def create_temp_array(self, shape: Tuple[int, ...], dtype: np.dtype, 
                         use_mmap: bool = None) -> np.ndarray:
        """Create temporary array with optimal memory strategy."""
        if use_mmap is None:
            use_mmap = self.use_mmap
        
        array_size_mb = (np.prod(shape) * dtype.itemsize) / (1024 * 1024)
        
        # Use memory mapping for large arrays
        if use_mmap and array_size_mb > self.chunk_memory_mb:
            # Create temporary file for memory mapping
            temp_file = self._create_temp_file(shape, dtype)
            return self.create_memory_mapped_array(temp_file, shape, dtype, mode='w+')
        else:
            # Regular numpy array
            return np.empty(shape, dtype=dtype)
    
    def _create_temp_file(self, shape: Tuple[int, ...], dtype: np.dtype) -> str:
        """Create temporary file for memory mapping."""
        import tempfile
        
        # Calculate file size
        file_size = np.prod(shape) * dtype.itemsize
        
        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.dat', prefix='bigtiff_conv_')
        
        try:
            # Pre-allocate file
            os.ftruncate(temp_fd, file_size)
            os.close(temp_fd)
            return temp_path
        except Exception:
            os.close(temp_fd)
            os.unlink(temp_path)
            raise
    
    def optimize_memory_layout(self, array: np.ndarray) -> np.ndarray:
        """Optimize array memory layout for better cache performance."""
        # Ensure C-contiguous layout for better cache locality
        if not array.flags.c_contiguous:
            return np.ascontiguousarray(array)
        return array
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            "process_memory_mb": memory_info.rss / (1024 * 1024),
            "process_memory_percent": process.memory_percent(),
            "system_memory_total_gb": system_memory.total / (1024**3),
            "system_memory_available_gb": system_memory.available / (1024**3),
            "system_memory_percent": system_memory.percent,
            "mapped_objects_count": len(self.mmap_objects),
            "allocated_memory_mb": self.allocated_memory / (1024 * 1024)
        }
    
    def should_use_memory_mapping(self, file_size_mb: float) -> bool:
        """Determine if memory mapping should be used for a file."""
        if not self.use_mmap:
            return False
        
        # Use memory mapping for files larger than chunk size
        return file_size_mb > self.chunk_memory_mb
    
    def cleanup_memory(self):
        """Force cleanup of unused memory."""
        # Clear weak references
        self.mmap_objects.clear()
        
        # Force garbage collection
        gc.collect()
        
        # Reset allocation tracking
        self.allocated_memory = 0
    
    async def monitor_memory_pressure(self) -> bool:
        """Monitor system memory pressure and return True if under pressure."""
        memory = psutil.virtual_memory()
        
        # Consider system under memory pressure if < 20% available
        if memory.percent > 80:
            return True
        
        # Check if we're approaching our configured limit
        if self.memory_limit:
            current_mb = psutil.Process().memory_info().rss / (1024 * 1024)
            if current_mb > self.memory_limit * 0.9:
                return True
        
        return False
    
    async def handle_memory_pressure(self):
        """Handle memory pressure by cleaning up and optimizing."""
        # Force garbage collection
        self.cleanup_memory()
        
        # Clear any cached data
        self.memory_pools.clear()
        
        # Reduce chunk size temporarily
        self.chunk_memory_mb = max(32, self.chunk_memory_mb // 2)


class MemoryPool:
    """Memory pool for efficient array allocation and reuse."""
    
    def __init__(self, pool_size_mb: int = 1024):
        self.pool_size_mb = pool_size_mb
        self.available_buffers = {}  # size -> list of buffers
        self.allocated_buffers = weakref.WeakSet()
    
    def get_buffer(self, size_bytes: int, dtype: np.dtype) -> np.ndarray:
        """Get buffer from pool or allocate new one."""
        # Round up to nearest power of 2 for better reuse
        rounded_size = 1 << (size_bytes - 1).bit_length()
        
        key = (rounded_size, dtype)
        
        if key in self.available_buffers and self.available_buffers[key]:
            buffer = self.available_buffers[key].pop()
            self.allocated_buffers.add(buffer)
            return buffer
        
        # Allocate new buffer
        num_elements = rounded_size // dtype.itemsize
        buffer = np.empty(num_elements, dtype=dtype)
        self.allocated_buffers.add(buffer)
        return buffer
    
    def return_buffer(self, buffer: np.ndarray):
        """Return buffer to pool for reuse."""
        key = (buffer.nbytes, buffer.dtype)
        
        if key not in self.available_buffers:
            self.available_buffers[key] = []
        
        # Only keep limited number of buffers per size
        if len(self.available_buffers[key]) < 10:
            self.available_buffers[key].append(buffer)
    
    def cleanup(self):
        """Cleanup memory pool."""
        self.available_buffers.clear()
        self.allocated_buffers.clear()
