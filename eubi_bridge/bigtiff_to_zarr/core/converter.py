"""
High-Performance BigTIFF to OME-NGFF Converter
Core conversion engine with parallel processing and memory optimization.
"""

import asyncio
import os
import time
import math
import mmap
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import tifffile
import zarr
import psutil
from numcodecs import Blosc

from .memory_manager import MemoryManager
from .compression_engine import CompressionEngine
from .async_io import AsyncIOManager
from .system_optimizer import SystemOptimizer
from .progress_monitor import ProgressMonitor
from ..utils.ngff_metadata import NGFFMetadataBuilder

from eubi_bridge.base.data_manager import ArrayManager


class HighPerformanceConverter:
    """High-performance converter with parallel processing and memory optimization."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory_manager = MemoryManager(config)
        self.compression_engine = CompressionEngine(config)
        self.async_io = AsyncIOManager(config)
        self.system_optimizer = SystemOptimizer(config)
        self.metadata_builder = NGFFMetadataBuilder()

        # Performance tracking
        self.start_time = None
        self.end_time = None
        self.bytes_processed = 0
        self.compression_ratio = 1.0
        self.is_cancelled = False

        # Worker pools
        self.process_pool = None
        self.thread_pool = None

    async def convert(self, input_path: str, output_path: str,
                     progress_monitor: Optional[ProgressMonitor] = None,
                     resume: bool = False) -> bool:
        """
        Convert BigTIFF to OME-NGFF with high-performance optimizations.

        Args:
            input_path: Path to input BigTIFF file
            output_path: Path to output OME-NGFF directory
            progress_monitor: Optional progress monitoring
            resume: Whether to resume interrupted conversion

        Returns:
            True if conversion successful, False otherwise
        """
        self.start_time = time.time()

        try:
            # Initialize system optimizations
            await self.system_optimizer.optimize_system()

            # Setup worker pools
            await self._setup_worker_pools()

            # Open input file with memory mapping
            tiff_file, tiff_store = await self._open_input_file(input_path)

            # Analyze input structure
            analysis = await self._analyze_input(tiff_file, tiff_store)

            # Create output store
            output_store = await self._create_output_store(output_path, analysis, resume)

            # Initialize progress monitoring
            if progress_monitor:
                await progress_monitor.initialize_conversion(analysis)

            # Convert pyramid levels in parallel
            success = await self._convert_pyramid_levels(
                tiff_file, tiff_store, output_store, analysis, progress_monitor
            )

            if success:
                # Generate NGFF metadata
                await self._generate_metadata(output_store, analysis)

                # Verify conversion integrity
                if self.config.get("verify_integrity", True):
                    await self._verify_conversion(output_store, analysis)

            self.end_time = time.time()
            return success

        except Exception as e:
            if progress_monitor:
                await progress_monitor.report_error(str(e))
            raise
        finally:
            await self._cleanup()

    async def _setup_worker_pools(self):
        """Setup optimized worker pools for parallel processing."""
        num_workers = self.config.get("workers", psutil.cpu_count())
        threads_per_worker = self.config.get("threads_per_worker", 2)

        # Resolve "auto" workers and ensure integers
        if num_workers == "auto" or num_workers is None:
            num_workers = psutil.cpu_count()
        elif isinstance(num_workers, str):
            num_workers = int(num_workers)

        if isinstance(threads_per_worker, str):
            threads_per_worker = int(threads_per_worker)

        # Process pool for CPU-intensive tasks
        self.process_pool = ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=None  # Use default context
        )

        # Thread pool for I/O operations
        self.thread_pool = ThreadPoolExecutor(
            max_workers=num_workers * threads_per_worker
        )

        # Setup NUMA affinity if enabled
        if self.config.get("use_numa", False):
            await self.system_optimizer.setup_numa_affinity(num_workers)

    async def _open_input_file(self, input_path: str) -> Tuple[tifffile.TiffFile, Any]:
        """Open input TIFF file with memory mapping."""
        try:
            # Open with memory mapping for large files
            tiff_file = tifffile.TiffFile(input_path)

            # Create zarr store from TIFF
            zarr_store = tiff_file.aszarr()

            # Open the zarr store as a zarr array for proper data access
            import zarr
            tiff_array = zarr.open(zarr_store, mode='r')

            return tiff_file, tiff_array

        except Exception as e:
            raise RuntimeError(f"Failed to open input file {input_path}: {e}")

    async def _analyze_input(self, tiff_file: tifffile.TiffFile, tiff_store: Any) -> Dict[str, Any]:
        """Analyze input file structure and determine optimal processing strategy."""
        # Get basic info
        shape = tiff_file.series[0].shape
        axes = tiff_file.series[0].axes.lower()
        dtype = tiff_file.series[0].dtype

        # Convert 's' (sample) to 'c' (channel) for consistency
        if 's' in axes:
            axes = axes.replace('s', 'c')

        # Extract pixel size information from TIFF metadata
        pixel_sizes = self._extract_pixel_sizes(tiff_file)

        # Calculate file size and optimal chunk sizes
        file_size = Path(tiff_file.filehandle.path).stat().st_size
        self.bytes_processed = file_size

        # Determine optimal chunking strategy
        optimal_chunks = await self.memory_manager.calculate_optimal_chunks(
            shape, dtype, axes
        )

        # Override with user-specified chunk sizes if provided
        user_chunks = self.config.get("chunk_sizes", {})
        if user_chunks:
            optimal_chunks = self._apply_user_chunk_sizes(optimal_chunks, axes, user_chunks)

        # Calculate pyramid levels
        pyramid_levels = self._calculate_pyramid_levels(shape, axes)

        analysis = {
            "shape": shape,
            "axes": axes,
            "dtype": dtype,
            "file_size": file_size,
            "optimal_chunks": optimal_chunks,
            "pyramid_levels": pyramid_levels,
            "pixel_sizes": pixel_sizes,
            "estimated_output_size": self._estimate_output_size(shape, dtype, pyramid_levels)
        }

        return analysis

    def _calculate_pyramid_levels(self, shape: Tuple[int, ...], axes: str) -> List[Dict[str, Any]]:
        """Calculate pyramid levels with user-specified downsampling parameters."""
        levels = []
        current_shape = list(shape)
        level = 0

        # Get all axes indices
        axis_indices = {}
        for i, axis in enumerate(axes):
            if axis == 't':
                axis_indices['time'] = i
            elif axis == 'c':
                axis_indices['channel'] = i
            elif axis == 'z':
                axis_indices['z'] = i
            elif axis == 'y':
                axis_indices['y'] = i
            elif axis == 'x':
                axis_indices['x'] = i

        # Get user-specified scale factors from pyramid_scales config
        pyramid_scales = self.config.get("pyramid_scales", {})
        scale_factors = {
            'time': pyramid_scales.get('time', 1),      # Default: no time downsampling
            'channel': pyramid_scales.get('channel', 1), # Default: no channel downsampling
            'z': pyramid_scales.get('z', 2),            # Default: 2x z downsampling
            'y': pyramid_scales.get('y', 2),            # Default: 2x y downsampling
            'x': pyramid_scales.get('x', 2)             # Default: 2x x downsampling
        }

        min_size = self.config.get("min_dimension_size", 64)
        max_levels = self.config.get("pyramid_levels", None)

        # Handle None pyramid_levels by calculating reasonable default
        if max_levels is None:
            # Calculate max levels based on smallest spatial dimension with custom scaling
            spatial_dims = []
            if 'y' in axis_indices:
                spatial_dims.append(shape[axis_indices['y']])
            if 'x' in axis_indices:
                spatial_dims.append(shape[axis_indices['x']])
            if 'z' in axis_indices:
                spatial_dims.append(shape[axis_indices['z']])

            if spatial_dims:
                min_spatial = min(spatial_dims)
                # Use the largest scale factor for estimation
                max_scale = max(scale_factors.get('y', 2), scale_factors.get('x', 2), scale_factors.get('z', 2))
                max_levels = max(1, int(np.log2(min_spatial / min_size) / np.log2(max_scale)) + 1)
            else:
                max_levels = 1

        print(f"Pyramid generation: {max_levels} levels, scale factors: {scale_factors}")

        while level < max_levels:
            # Calculate downsample factors for this level
            level_factors = {}
            for axis_name in scale_factors:
                if scale_factors[axis_name] > 1:
                    level_factors[axis_name] = scale_factors[axis_name] ** level
                else:
                    level_factors[axis_name] = 1

            levels.append({
                "level": level,
                "shape": tuple(current_shape),
                "downsample_factors": level_factors
            })

            # Check if we should continue (based on spatial dimensions)
            spatial_sizes = []
            if 'y' in axis_indices:
                spatial_sizes.append(current_shape[axis_indices['y']])
            if 'x' in axis_indices:
                spatial_sizes.append(current_shape[axis_indices['x']])
            if 'z' in axis_indices:
                spatial_sizes.append(current_shape[axis_indices['z']])

            if spatial_sizes and min(spatial_sizes) < min_size:
                break

            # Calculate next level shape using custom scale factors
            for axis_name, axis_idx in axis_indices.items():
                if axis_name in scale_factors and scale_factors[axis_name] > 1:
                    current_shape[axis_idx] = max(1, current_shape[axis_idx] // scale_factors[axis_name])

            level += 1

        print(f"Generated {len(levels)} pyramid levels")
        for i, level_info in enumerate(levels):
            print(f"  Level {i}: shape={level_info['shape']}, factors={level_info['downsample_factors']}")

        return levels

    def _extract_pixel_sizes(self, tiff_file_obj: tifffile.TiffFile) -> Dict[str, float]:
        """Extract pixel sizes from TIFF metadata."""

        pixel_sizes = {}
        path = tiff_file_obj.filehandle.path
        axes = tiff_file_obj.series[0].axes
        # path = f"/home/oezdemir/PycharmProjects/eubizarr1/test_tiff1/Channel/channel0_image.tif"
        manager = ArrayManager(path)
        for ax in axes:
            if ax.lower() == 's':
                pixel_sizes['c'] = manager.scaledict['c']
            elif ax.lower() in manager.scaledict:
                pixel_sizes[ax.lower()] = manager.scaledict[ax.lower()]

        print(f"Extracted pixel sizes: {pixel_sizes}")
        return pixel_sizes

    def _apply_user_chunk_sizes(self, optimal_chunks: Tuple[int, ...], axes: str, user_chunks: Dict[str, int]) -> Tuple[int, ...]:
        """Apply user-specified chunk sizes to override optimal chunks."""
        chunk_list = list(optimal_chunks)

        # Map axis names to indices
        axis_mapping = {'t': 'time', 'c': 'channel', 'z': 'z', 'y': 'y', 'x': 'x'}

        for i, axis in enumerate(axes):
            axis_name = axis_mapping.get(axis, axis)
            if axis_name in user_chunks and user_chunks[axis_name] > 0:
                if i < len(chunk_list):
                    chunk_list[i] = user_chunks[axis_name]
                    print(f"Applied user chunk size for {axis} ({axis_name}): {user_chunks[axis_name]}")

        return tuple(chunk_list)

    def _estimate_output_size(self, shape: Tuple[int, ...], dtype: np.dtype,
                            pyramid_levels: List[Dict[str, Any]]) -> int:
        """Estimate total output size for all pyramid levels."""
        base_size = np.prod(shape) * dtype.itemsize

        # Estimate compression ratio
        compression_ratio = self.compression_engine.get_estimated_compression_ratio()

        # Sum all pyramid levels (geometric series approximation)
        total_size = 0
        for level_info in pyramid_levels:
            level_size = np.prod(level_info["shape"]) * dtype.itemsize
            total_size += level_size

        return int(total_size / compression_ratio)

    async def _create_output_store(self, output_path: str, analysis: Dict[str, Any],
                                 resume: bool) -> zarr.Group:
        """Create output Zarr store with optimal configuration."""
        output_dir = Path(output_path)

        if output_dir.exists() and not resume:
            import shutil
            shutil.rmtree(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Create Zarr store - use simple path for compatibility with Zarr 3.x
        print(f"Creating output store with config: {self.config}")
        zarr_format = self.config.get("zarr_format", 2)

        # Use path-based store creation for Zarr 3.x compatibility
        root_group = zarr.group(store=str(output_dir), overwrite=True, zarr_format=zarr_format)
        return root_group

    async def _convert_pyramid_levels(self, tiff_file: tifffile.TiffFile, tiff_store: Any,
                                    output_store: zarr.Group, analysis: Dict[str, Any],
                                    progress_monitor: Optional[ProgressMonitor]) -> bool:
        """Convert all pyramid levels using parallel processing."""
        pyramid_levels = analysis["pyramid_levels"]

        # Process levels in parallel (level 0 first, then others)
        tasks = []

        for level_info in pyramid_levels:
            if self.is_cancelled:
                return False

            task = self._process_pyramid_level(
                tiff_file, tiff_store, output_store, analysis, level_info, progress_monitor
            )
            tasks.append(task)

        # Execute tasks with concurrency control
        max_concurrent = min(len(tasks), self.config.get("max_concurrent_levels", 3))

        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_task(task):
            async with semaphore:
                return await task

        results = await asyncio.gather(*[bounded_task(task) for task in tasks], return_exceptions=True)

        # Check for failures
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                if progress_monitor:
                    await progress_monitor.report_error(f"Level {i} failed: {result}")
                return False

        return True

    async def _process_pyramid_level(self, tiff_file: tifffile.TiffFile, tiff_store: Any,
                                   output_store: zarr.Group, analysis: Dict[str, Any],
                                   level_info: Dict[str, Any],
                                   progress_monitor: Optional[ProgressMonitor]) -> bool:
        """Process a single pyramid level with race condition protection."""
        level = level_info["level"]
        level_shape = level_info["shape"]

        # Create level array in output store (Zarr 3.x compatible)
        level_array = output_store.create_dataset(
            str(level),
            shape=level_shape,
            chunks=analysis["optimal_chunks"],
            dtype=analysis["dtype"],
            overwrite=True,
            chunk_key_encoding={"name": "v2", "separator": "/"}
        )

        # Calculate chunk processing plan
        chunks_plan = self._calculate_chunk_plan(level_shape, analysis["optimal_chunks"])

        # Create write coordination system to prevent race conditions
        # Each output chunk gets a unique lock to ensure sequential writes to overlapping regions
        output_chunk_locks = {}
        output_chunk_size = analysis["optimal_chunks"]

        # Calculate output chunk coordinates for each processing chunk
        for chunk_info in chunks_plan:
            chunk_slice = chunk_info["slice"]
            overlapping_output_chunks = self._get_overlapping_output_chunks(
                chunk_slice, output_chunk_size, level_shape
            )
            chunk_info["output_chunks"] = overlapping_output_chunks

            # Create locks for overlapping output chunks
            for output_coord in overlapping_output_chunks:
                if output_coord not in output_chunk_locks:
                    output_chunk_locks[output_coord] = asyncio.Lock()

        # Process chunks with write coordination
        chunk_tasks = []
        for chunk_info in chunks_plan:
            if self.is_cancelled:
                return False

            task = self._process_chunk_with_coordination(
                tiff_store, level_array, chunk_info, level_info, analysis, output_chunk_locks
            )
            chunk_tasks.append(task)

        # Execute chunk tasks with progress tracking
        completed_chunks = 0
        total_chunks = len(chunk_tasks)

        for coro in asyncio.as_completed(chunk_tasks):
            try:
                await coro
                completed_chunks += 1

                if progress_monitor:
                    await progress_monitor.update_progress(
                        level=level,
                        chunks_completed=completed_chunks,
                        total_chunks=total_chunks
                    )

            except Exception as e:
                if progress_monitor:
                    await progress_monitor.report_error(f"Chunk processing failed: {e}")
                return False

        return True

    def _calculate_chunk_plan(self, shape: Tuple[int, ...],
                            chunk_size: Tuple[int, ...]) -> List[Dict[str, Any]]:
        """Calculate optimal chunk processing plan."""
        chunks = []

        # Calculate number of chunks per dimension
        chunks_per_dim = [math.ceil(s / c) for s, c in zip(shape, chunk_size)]

        # Generate chunk coordinates
        import itertools
        for chunk_coords in itertools.product(*[range(n) for n in chunks_per_dim]):
            # Calculate slice for this chunk
            chunk_slice = tuple(
                slice(coord * chunk_size[i], min((coord + 1) * chunk_size[i], shape[i]))
                for i, coord in enumerate(chunk_coords)
            )

            chunks.append({
                "coords": chunk_coords,
                "slice": chunk_slice,
                "shape": tuple(s.stop - s.start for s in chunk_slice)
            })

        return chunks

    def _get_overlapping_output_chunks(self, input_slice: Tuple[slice, ...],
                                     output_chunk_size: Tuple[int, ...],
                                     level_shape: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Calculate which output chunks overlap with an input slice."""
        overlapping_chunks = []

        # Calculate the range of output chunks that this input slice touches
        for dim_ranges in range(len(input_slice)):
            start_idx = input_slice[dim_ranges].start // output_chunk_size[dim_ranges]
            stop_idx = (input_slice[dim_ranges].stop - 1) // output_chunk_size[dim_ranges]

        # Generate all combinations of overlapping chunk coordinates
        import itertools
        chunk_ranges = []
        for i in range(len(input_slice)):
            start_chunk = input_slice[i].start // output_chunk_size[i]
            end_chunk = (input_slice[i].stop - 1) // output_chunk_size[i]
            chunk_ranges.append(range(start_chunk, end_chunk + 1))

        for chunk_coord in itertools.product(*chunk_ranges):
            overlapping_chunks.append(chunk_coord)

        return overlapping_chunks

    async def _process_chunk_with_coordination(self, tiff_store: Any, level_array: zarr.Array,
                                             chunk_info: Dict[str, Any], level_info: Dict[str, Any],
                                             analysis: Dict[str, Any], output_locks: Dict[Tuple[int, ...], asyncio.Lock]):
        """Process chunk with write coordination to prevent race conditions."""
        chunk_slice = chunk_info["slice"]
        overlapping_output_chunks = chunk_info["output_chunks"]
        level = level_info["level"]

        # Acquire all necessary locks in a consistent order to prevent deadlocks
        sorted_chunks = sorted(overlapping_output_chunks)
        locks_to_acquire = [output_locks[coord] for coord in sorted_chunks]

        try:
            # Acquire all locks for overlapping output chunks
            for lock in locks_to_acquire:
                await lock.acquire()

            # Now safely process the chunk
            await self._process_chunk(tiff_store, level_array, chunk_info, level_info, analysis)

        finally:
            # Release all locks in reverse order
            for lock in reversed(locks_to_acquire):
                lock.release()

    async def _process_chunk(self, tiff_store: Any, level_array: zarr.Array,
                           chunk_info: Dict[str, Any], level_info: Dict[str, Any],
                           analysis: Dict[str, Any]):
        """Process a single chunk with downsampling and optimization."""
        chunk_slice = chunk_info["slice"]
        level = level_info["level"]

        try:
            # Read source data
            if level == 0:
                # Level 0: direct copy - tiff_store is now a proper zarr array
                source_data = np.array(tiff_store[chunk_slice])
            else:
                # Higher levels: downsample from level 0
                downsample_factors = level_info["downsample_factors"]
                source_data = await self._downsample_chunk(
                    tiff_store, chunk_slice, downsample_factors, analysis["axes"]
                )
            
            # Apply any data transformations
            if analysis["dtype"] != source_data.dtype:
                source_data = source_data.astype(analysis["dtype"])
            
            # Write to output array
            level_array[chunk_slice] = source_data
            
        except Exception as e:
            raise RuntimeError(f"Chunk processing failed: {e}")
    
    async def _downsample_chunk(self, tiff_store: Any, chunk_slice: Tuple[slice, ...],
                              downsample_factors: Dict[str, int], axes: str) -> np.ndarray:
        """Downsample chunk data using optimized algorithms."""
        # Calculate source slice for downsampling
        source_slice = []
        for i, s in enumerate(chunk_slice):
            ax = axes[i] if i < len(axes) else 'unknown'
            factor = downsample_factors.get(ax, 1)
            
            if factor > 1:
                # Expand slice to include source pixels
                start = s.start * factor
                stop = s.stop * factor
                source_slice.append(slice(start, stop))
            else:
                source_slice.append(s)
        
        source_slice = tuple(source_slice)
        
        # Read source data
        source_data = tiff_store[source_slice]
        
        # Apply downsampling
        if any(downsample_factors.get(axes[i], 1) > 1 for i in range(len(axes))):
            # Use optimized downsampling (e.g., block reduction)
            downsampled = await self._block_reduce(source_data, downsample_factors, axes)
            return downsampled
        else:
            return source_data
    
    async def _block_reduce(self, data: np.ndarray, downsample_factors: Dict[str, int], 
                          axes: str) -> np.ndarray:
        """Optimized block reduction for downsampling."""
        # Simple implementation using strided indexing for now
        # Could be enhanced with proper block averaging
        
        slices = []
        for i, ax in enumerate(axes):
            factor = downsample_factors.get(ax, 1)
            if factor > 1:
                slices.append(slice(0, None, factor))
            else:
                slices.append(slice(None))
        
        return data[tuple(slices)]
    
    async def _generate_metadata(self, output_store: zarr.Group, analysis: Dict[str, Any]):
        """Generate OME-NGFF metadata."""
        metadata = self.metadata_builder.build_multiscales_metadata(
            name="Converted BigTIFF",
            axes=analysis["axes"],
            pyramid_levels=analysis["pyramid_levels"],
            dtype=analysis["dtype"],
            pixel_sizes=analysis.get("pixel_sizes", {})
        )
        
        # Write metadata to zarr attributes
        output_store.attrs.update(metadata)
    
    async def _verify_conversion(self, output_store: zarr.Group, analysis: Dict[str, Any]):
        """Verify conversion integrity."""
        # Basic verification: check that all levels exist and have correct shapes
        for level_info in analysis["pyramid_levels"]:
            level = level_info["level"]
            expected_shape = level_info["shape"]
            
            if str(level) not in output_store:
                raise RuntimeError(f"Missing pyramid level {level}")
            
            level_array = output_store[str(level)]
            if level_array.shape != expected_shape:
                raise RuntimeError(f"Level {level} shape mismatch: {level_array.shape} != {expected_shape}")
    
    async def cancel(self):
        """Cancel the conversion process."""
        self.is_cancelled = True
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        duration = (self.end_time or time.time()) - (self.start_time or time.time())
        throughput_mb_s = (self.bytes_processed / (1024 * 1024)) / max(duration, 0.001)
        
        return {
            "duration": duration,
            "bytes_processed": self.bytes_processed,
            "throughput_mb_s": throughput_mb_s,
            "compression_ratio": self.compression_ratio
        }
    
    async def _cleanup(self):
        """Cleanup resources."""
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        await self.system_optimizer.cleanup()
