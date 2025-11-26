"""
Asynchronous I/O manager for high-performance file operations.
Implements non-blocking I/O with buffering and prefetching.
"""

import asyncio
import aiofiles
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, BinaryIO, AsyncGenerator
import numpy as np
import mmap
import time
from concurrent.futures import ThreadPoolExecutor


class AsyncIOManager:
    """High-performance asynchronous I/O manager."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.buffer_size = config.get("io_buffer_size_mb", 64) * 1024 * 1024
        self.max_concurrent_reads = config.get("max_concurrent_reads", 8)
        self.max_concurrent_writes = config.get("max_concurrent_writes", 4)
        self.use_direct_io = config.get("use_direct_io", False)
        
        # I/O pools and queues
        self.read_semaphore = asyncio.Semaphore(self.max_concurrent_reads)
        self.write_semaphore = asyncio.Semaphore(self.max_concurrent_writes)
        self.prefetch_queue = asyncio.Queue(maxsize=100)
        self.write_queue = asyncio.Queue(maxsize=50)
        
        # Performance tracking
        self.io_stats = {
            "bytes_read": 0,
            "bytes_written": 0,
            "read_operations": 0,
            "write_operations": 0,
            "read_time": 0.0,
            "write_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Buffer cache
        self.read_cache = {}
        self.max_cache_size = config.get("read_cache_mb", 512) * 1024 * 1024
        self.current_cache_size = 0
        
        # Background tasks
        self.background_tasks = set()
    
    async def read_chunk_async(self, file_path: str, offset: int, 
                             size: int, use_cache: bool = True) -> bytes:
        """Asynchronously read data chunk from file."""
        cache_key = (file_path, offset, size) if use_cache else None
        
        # Check cache first
        if cache_key and cache_key in self.read_cache:
            self.io_stats["cache_hits"] += 1
            return self.read_cache[cache_key]
        
        async with self.read_semaphore:
            start_time = time.time()
            
            try:
                # Use aiofiles for async I/O
                async with aiofiles.open(file_path, 'rb') as f:
                    await f.seek(offset)
                    data = await f.read(size)
                
                # Update statistics
                self.io_stats["bytes_read"] += len(data)
                self.io_stats["read_operations"] += 1
                self.io_stats["read_time"] += time.time() - start_time
                self.io_stats["cache_misses"] += 1
                
                # Cache the data if requested and space available
                if use_cache and len(data) <= self.buffer_size:
                    await self._cache_data(cache_key, data)
                
                return data
                
            except Exception as e:
                raise IOError(f"Failed to read from {file_path} at offset {offset}: {e}")
    
    async def write_chunk_async(self, file_path: str, offset: int, 
                              data: bytes, sync: bool = False) -> None:
        """Asynchronously write data chunk to file."""
        async with self.write_semaphore:
            start_time = time.time()
            
            try:
                # Ensure directory exists
                Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                
                # Use aiofiles for async I/O
                mode = 'rb+' if Path(file_path).exists() else 'wb'
                async with aiofiles.open(file_path, mode) as f:
                    await f.seek(offset)
                    await f.write(data)
                    
                    if sync:
                        await f.fsync()
                
                # Update statistics
                self.io_stats["bytes_written"] += len(data)
                self.io_stats["write_operations"] += 1
                self.io_stats["write_time"] += time.time() - start_time
                
            except Exception as e:
                raise IOError(f"Failed to write to {file_path} at offset {offset}: {e}")
    
    async def read_array_async(self, file_path: str, shape: tuple, 
                             dtype: np.dtype, offset: int = 0) -> np.ndarray:
        """Asynchronously read numpy array from file."""
        total_bytes = np.prod(shape) * dtype.itemsize
        
        # For large arrays, use memory mapping
        if total_bytes > self.buffer_size * 2:
            return await self._read_large_array_mmap(file_path, shape, dtype, offset)
        
        # For smaller arrays, read directly
        data_bytes = await self.read_chunk_async(file_path, offset, total_bytes)
        return np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
    
    async def write_array_async(self, file_path: str, array: np.ndarray, 
                              offset: int = 0, sync: bool = False) -> None:
        """Asynchronously write numpy array to file."""
        # Ensure array is contiguous
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        
        data_bytes = array.tobytes()
        await self.write_chunk_async(file_path, offset, data_bytes, sync)
    
    async def _read_large_array_mmap(self, file_path: str, shape: tuple, 
                                   dtype: np.dtype, offset: int) -> np.ndarray:
        """Read large array using memory mapping in thread pool."""
        def _mmap_read():
            try:
                return np.memmap(
                    file_path, 
                    dtype=dtype, 
                    mode='r', 
                    offset=offset, 
                    shape=shape
                )
            except Exception as e:
                # Fallback to regular file reading
                with open(file_path, 'rb') as f:
                    f.seek(offset)
                    data = f.read(np.prod(shape) * dtype.itemsize)
                    return np.frombuffer(data, dtype=dtype).reshape(shape)
        
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, _mmap_read)
    
    async def prefetch_chunks(self, file_path: str, 
                            chunks: List[Dict[str, Any]]) -> None:
        """Prefetch multiple chunks asynchronously."""
        async def prefetch_single_chunk(chunk_info):
            try:
                offset = chunk_info["offset"]
                size = chunk_info["size"]
                await self.read_chunk_async(file_path, offset, size, use_cache=True)
            except Exception:
                pass  # Ignore prefetch errors
        
        # Limit concurrent prefetch operations
        semaphore = asyncio.Semaphore(4)
        
        async def bounded_prefetch(chunk_info):
            async with semaphore:
                await prefetch_single_chunk(chunk_info)
        
        # Start prefetch tasks
        prefetch_tasks = [bounded_prefetch(chunk) for chunk in chunks]
        
        # Don't wait for completion, just start the tasks
        for task in prefetch_tasks:
            task_obj = asyncio.create_task(task)
            self.background_tasks.add(task_obj)
            task_obj.add_done_callback(self.background_tasks.discard)
    
    async def batch_read_chunks(self, file_path: str, 
                              chunk_specs: List[Dict[str, Any]]) -> List[bytes]:
        """Read multiple chunks in parallel with optimal batching."""
        # Group chunks by proximity for better I/O efficiency
        sorted_chunks = sorted(chunk_specs, key=lambda x: x.get("offset", 0))
        
        # Create read tasks
        read_tasks = []
        for chunk_spec in sorted_chunks:
            task = self.read_chunk_async(
                file_path, 
                chunk_spec["offset"], 
                chunk_spec["size"],
                chunk_spec.get("use_cache", True)
            )
            read_tasks.append(task)
        
        # Execute reads with controlled concurrency
        results = await asyncio.gather(*read_tasks, return_exceptions=True)
        
        # Handle any exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                raise IOError(f"Failed to read chunk {i}: {result}")
        
        return results
    
    async def batch_write_chunks(self, write_specs: List[Dict[str, Any]]) -> None:
        """Write multiple chunks in parallel with optimal batching."""
        # Create write tasks
        write_tasks = []
        for spec in write_specs:
            task = self.write_chunk_async(
                spec["file_path"],
                spec["offset"],
                spec["data"],
                spec.get("sync", False)
            )
            write_tasks.append(task)
        
        # Execute writes with controlled concurrency
        await asyncio.gather(*write_tasks)
    
    async def _cache_data(self, cache_key: tuple, data: bytes) -> None:
        """Cache data with LRU eviction."""
        data_size = len(data)
        
        # Check if we need to evict data
        while self.current_cache_size + data_size > self.max_cache_size and self.read_cache:
            # Simple LRU: remove oldest entry
            oldest_key = next(iter(self.read_cache))
            oldest_data = self.read_cache.pop(oldest_key)
            self.current_cache_size -= len(oldest_data)
        
        # Add new data to cache
        if data_size <= self.max_cache_size:
            self.read_cache[cache_key] = data
            self.current_cache_size += data_size
    
    def clear_cache(self):
        """Clear the read cache."""
        self.read_cache.clear()
        self.current_cache_size = 0
    
    async def flush_write_queue(self):
        """Flush any pending write operations."""
        # Wait for all background write tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
            self.background_tasks.clear()
    
    def get_io_stats(self) -> Dict[str, Any]:
        """Get I/O performance statistics."""
        stats = self.io_stats.copy()
        
        # Calculate derived metrics
        if stats["read_time"] > 0:
            stats["read_throughput_mb_s"] = (
                (stats["bytes_read"] / (1024 * 1024)) / stats["read_time"]
            )
        else:
            stats["read_throughput_mb_s"] = 0.0
        
        if stats["write_time"] > 0:
            stats["write_throughput_mb_s"] = (
                (stats["bytes_written"] / (1024 * 1024)) / stats["write_time"]
            )
        else:
            stats["write_throughput_mb_s"] = 0.0
        
        if stats["cache_hits"] + stats["cache_misses"] > 0:
            stats["cache_hit_ratio"] = (
                stats["cache_hits"] / (stats["cache_hits"] + stats["cache_misses"])
            )
        else:
            stats["cache_hit_ratio"] = 0.0
        
        stats["cache_size_mb"] = self.current_cache_size / (1024 * 1024)
        stats["active_background_tasks"] = len(self.background_tasks)
        
        return stats
    
    async def optimize_for_sequential_access(self, file_path: str, 
                                           expected_read_size: int):
        """Optimize I/O settings for sequential access patterns."""
        # Increase buffer size for large sequential reads
        if expected_read_size > self.buffer_size:
            self.buffer_size = min(expected_read_size, 256 * 1024 * 1024)  # Max 256MB
        
        # Increase prefetch for sequential patterns
        self.max_concurrent_reads = min(16, self.max_concurrent_reads * 2)
    
    async def optimize_for_random_access(self, file_path: str):
        """Optimize I/O settings for random access patterns."""
        # Smaller buffer size for random access
        self.buffer_size = max(self.buffer_size // 2, 4 * 1024 * 1024)  # Min 4MB
        
        # Increase cache size for random access
        self.max_cache_size = min(self.max_cache_size * 2, 2048 * 1024 * 1024)  # Max 2GB
    
    async def cleanup(self):
        """Cleanup I/O manager resources."""
        # Cancel all background tasks
        for task in self.background_tasks:
            task.cancel()
        
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Clear cache
        self.clear_cache()
        
        # Clear queues
        while not self.prefetch_queue.empty():
            try:
                self.prefetch_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        while not self.write_queue.empty():
            try:
                self.write_queue.get_nowait()
            except asyncio.QueueEmpty:
                break


class StreamingArrayReader:
    """Streaming reader for large arrays with async I/O."""
    
    def __init__(self, file_path: str, shape: tuple, dtype: np.dtype, 
                 io_manager: AsyncIOManager, chunk_size: tuple = None):
        self.file_path = file_path
        self.shape = shape
        self.dtype = dtype
        self.io_manager = io_manager
        self.element_size = dtype.itemsize
        
        # Calculate optimal chunk size if not provided
        if chunk_size is None:
            self.chunk_size = self._calculate_optimal_chunk_size()
        else:
            self.chunk_size = chunk_size
    
    def _calculate_optimal_chunk_size(self) -> tuple:
        """Calculate optimal chunk size for streaming."""
        target_chunk_bytes = self.io_manager.buffer_size
        total_elements = target_chunk_bytes // self.element_size
        
        # Distribute elements across dimensions
        ndim = len(self.shape)
        elements_per_dim = int(total_elements ** (1.0 / ndim))
        
        chunk_size = tuple(
            min(dim_size, max(1, elements_per_dim)) 
            for dim_size in self.shape
        )
        
        return chunk_size
    
    async def read_chunks(self) -> AsyncGenerator[tuple, None]:
        """Asynchronously yield chunks of the array."""
        import itertools
        
        # Calculate chunk coordinates
        chunks_per_dim = [
            (dim_size + chunk_size - 1) // chunk_size 
            for dim_size, chunk_size in zip(self.shape, self.chunk_size)
        ]
        
        for chunk_coords in itertools.product(*[range(n) for n in chunks_per_dim]):
            # Calculate slice for this chunk
            chunk_slice = tuple(
                slice(
                    coord * self.chunk_size[i], 
                    min((coord + 1) * self.chunk_size[i], self.shape[i])
                )
                for i, coord in enumerate(chunk_coords)
            )
            
            # Calculate byte offset and size
            offset = self._calculate_byte_offset(chunk_slice)
            chunk_shape = tuple(s.stop - s.start for s in chunk_slice)
            chunk_bytes = np.prod(chunk_shape) * self.element_size
            
            # Read chunk data
            data_bytes = await self.io_manager.read_chunk_async(
                self.file_path, offset, chunk_bytes
            )
            
            # Convert to numpy array
            chunk_data = np.frombuffer(data_bytes, dtype=self.dtype).reshape(chunk_shape)
            
            yield chunk_slice, chunk_data
    
    def _calculate_byte_offset(self, chunk_slice: tuple) -> int:
        """Calculate byte offset for chunk slice."""
        # Simple row-major offset calculation
        offset = 0
        multiplier = 1
        
        for i in reversed(range(len(chunk_slice))):
            offset += chunk_slice[i].start * multiplier
            multiplier *= self.shape[i]
        
        return offset * self.element_size
