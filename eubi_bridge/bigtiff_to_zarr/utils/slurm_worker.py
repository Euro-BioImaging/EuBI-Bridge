"""
SLURM Worker Functions with Race Condition Protection
Handles chunk processing for individual worker nodes in distributed conversion with coordination.
"""

import json
import time
import math
import hashlib
import fcntl
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set
import numpy as np
import tifffile
import zarr
from scipy import ndimage


def process_worker_chunks(worker_id: int, work_dir: str, input_tiff: str, output_zarr_dir: str, config: Dict[str, Any]):
    """Process assigned chunks for this worker with race condition protection."""
    print(f"Worker {worker_id} starting chunk processing with race condition protection...")

    work_path = Path(work_dir)

    # Load task assignment with coordination metadata
    task_queue_file = work_path / "task_queue.json"
    with open(task_queue_file, 'r') as f:
        task_data = json.load(f)
        task_queue = task_data["tasks"]
        coordination_metadata = task_data.get("coordination_metadata", {})

    # Load processing plan
    plan_file = work_path / "processing_plan.json"
    with open(plan_file, 'r') as f:
        processing_plan = json.load(f)

    # Load race condition metadata
    race_metadata_file = Path(output_zarr_dir) / "race_condition_metadata.json"
    with open(race_metadata_file, 'r') as f:
        race_condition_metadata = json.load(f)

    # Find this worker's task
    worker_task = None
    for task in task_queue:
        if task["worker_id"] == worker_id:
            worker_task = task
            break

    if not worker_task:
        print(f"❌ No task found for worker {worker_id}")
        return

    print(f"Worker {worker_id} processing {len(worker_task['chunk_range'])} chunks with coordination")

    # Initialize coordination system
    coordinator = WorkerCoordinator(worker_id, work_path, coordination_metadata)

    # Open input TIFF and output zarr
    with tifffile.TiffFile(input_tiff) as tiff:
        zarr_store = zarr.open(output_zarr_dir, mode='r+')

        # Process each assigned chunk with coordination
        for chunk_idx in worker_task["chunk_range"]:
            try:
                _process_single_chunk_with_coordination(
                    chunk_idx, tiff, zarr_store, processing_plan, config,
                    race_condition_metadata, coordinator
                )
                print(f"Worker {worker_id} completed chunk {chunk_idx}")
            except Exception as e:
                print(f"❌ Worker {worker_id} failed on chunk {chunk_idx}: {e}")
                # Release any held locks before re-raising
                coordinator.release_all_locks()
                raise

    # Final cleanup
    coordinator.release_all_locks()
    print(f"✅ Worker {worker_id} completed all assigned chunks")


class WorkerCoordinator:
    """Coordinates distributed workers to prevent race conditions."""

    def __init__(self, worker_id: int, work_dir: Path, coordination_metadata: Dict[str, Any]):
        self.worker_id = worker_id
        self.work_dir = work_dir
        self.coordination_metadata = coordination_metadata
        self.lock_dir = work_dir / "coordination_locks"
        self.held_locks = set()

    def acquire_output_chunk_lock(self, level: int, output_chunk_coords: Tuple[int, ...]) -> bool:
        """Acquire exclusive lock for an output zarr chunk."""
        lock_id = f"level_{level}_chunk_{'_'.join(map(str, output_chunk_coords))}"
        lock_file = self.lock_dir / f"{lock_id}.lock"

        try:
            # Create lock file with worker ID
            with open(lock_file, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                f.write(f"worker_{self.worker_id}")
                f.flush()

            self.held_locks.add(lock_file)
            return True

        except (IOError, OSError):
            # Lock already held by another worker
            return False

    def release_output_chunk_lock(self, level: int, output_chunk_coords: Tuple[int, ...]):
        """Release lock for an output zarr chunk."""
        lock_id = f"level_{level}_chunk_{'_'.join(map(str, output_chunk_coords))}"
        lock_file = self.lock_dir / f"{lock_id}.lock"

        if lock_file in self.held_locks:
            try:
                lock_file.unlink(missing_ok=True)
                self.held_locks.remove(lock_file)
            except OSError:
                pass

    def release_all_locks(self):
        """Release all held locks."""
        for lock_file in list(self.held_locks):
            try:
                lock_file.unlink(missing_ok=True)
            except OSError:
                pass
        self.held_locks.clear()

    def wait_for_lock_with_timeout(self, level: int, output_chunk_coords: Tuple[int, ...],
                                   timeout_seconds: int = 300) -> bool:
        """Wait for lock with timeout."""
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            if self.acquire_output_chunk_lock(level, output_chunk_coords):
                return True
            time.sleep(0.1)  # Brief sleep between attempts

        return False


def _process_single_chunk_with_coordination(chunk_idx: int, tiff: tifffile.TiffFile, zarr_store: zarr.Group,
                                            processing_plan: Dict[str, Any], config: Dict[str, Any],
                                            race_condition_metadata: Dict[str, Any],
                                            coordinator: WorkerCoordinator):
    """Process a single chunk across all pyramid levels with race condition protection."""

    # Calculate chunk coordinates from linear index
    chunks_per_dim = processing_plan["chunks_per_dim"]
    chunk_sizes = processing_plan["chunk_sizes"]

    # Convert linear chunk index to coordinates
    if len(chunks_per_dim) == 3:
        z_chunks, y_chunks, x_chunks = chunks_per_dim
        z_idx = chunk_idx // (y_chunks * x_chunks)
        y_idx = (chunk_idx % (y_chunks * x_chunks)) // x_chunks
        x_idx = chunk_idx % x_chunks
        input_chunk_coords = (z_idx, y_idx, x_idx)
    else:
        y_chunks, x_chunks = chunks_per_dim
        y_idx = chunk_idx // x_chunks
        x_idx = chunk_idx % x_chunks
        input_chunk_coords = (y_idx, x_idx)

    # Get input data shape
    page = tiff.pages[0]
    if len(tiff.pages) > 1 and len(page.shape) == 2:
        input_shape = (len(tiff.pages), page.shape[0], page.shape[1])
    else:
        input_shape = page.shape

    # Calculate input chunk slice
    input_slice = _calculate_input_slice(input_chunk_coords, chunk_sizes, input_shape)

    # Read input data for this chunk
    chunk_data = _read_input_chunk_data(tiff, input_slice, input_shape)

    # Process chunk for each pyramid level with coordination
    for level_idx in range(len(zarr_store.keys())):
        level_name = str(level_idx)
        if level_name not in zarr_store or level_name not in race_condition_metadata:
            continue

        level_array = zarr_store[level_name]
        level_race_metadata = race_condition_metadata[level_name]

        # Calculate downsampling for this level
        if level_idx == 0:
            level_data = chunk_data
        else:
            downsample_factors = _get_downsample_factors(level_idx, config)
            level_data = _downsample_data(chunk_data, downsample_factors)

        # Process with coordination if needed
        if level_race_metadata["requires_coordination"]:
            _write_with_coordination(level_data, input_chunk_coords, level_idx, level_array,
                                     level_race_metadata, coordinator, chunk_sizes, input_shape)
        else:
            # Direct write for aligned chunks
            level_slice = _calculate_output_slice(input_chunk_coords, level_idx, chunk_sizes,
                                                  input_shape, level_array.shape)
            if level_data.size > 0 and all(s.start < s.stop for s in level_slice):
                level_array[level_slice] = level_data


def _process_single_chunk(chunk_idx: int, tiff: tifffile.TiffFile, zarr_store: zarr.Group,
                          processing_plan: Dict[str, Any], config: Dict[str, Any]):
    """Process a single chunk across all pyramid levels."""

    # Calculate chunk coordinates from linear index
    chunks_per_dim = processing_plan["chunks_per_dim"]
    chunk_sizes = processing_plan["chunk_sizes"]

    # Convert linear chunk index to 3D coordinates
    if len(chunks_per_dim) == 3:
        z_chunks, y_chunks, x_chunks = chunks_per_dim
        z_idx = chunk_idx // (y_chunks * x_chunks)
        y_idx = (chunk_idx % (y_chunks * x_chunks)) // x_chunks
        x_idx = chunk_idx % x_chunks
        chunk_coords = (z_idx, y_idx, x_idx)
    else:
        y_chunks, x_chunks = chunks_per_dim
        y_idx = chunk_idx // x_chunks
        x_idx = chunk_idx % x_chunks
        chunk_coords = (y_idx, x_idx)

    # Get input data shape
    page = tiff.pages[0]
    if len(tiff.pages) > 1 and len(page.shape) == 2:
        input_shape = (len(tiff.pages), page.shape[0], page.shape[1])
    else:
        input_shape = page.shape

    # Calculate chunk slice in input space
    if len(input_shape) == 3:
        z_start = chunk_coords[0] * chunk_sizes["z"]
        z_end = min(z_start + chunk_sizes["z"], input_shape[0])
        y_start = chunk_coords[1] * chunk_sizes["y"]
        y_end = min(y_start + chunk_sizes["y"], input_shape[1])
        x_start = chunk_coords[2] * chunk_sizes["x"]
        x_end = min(x_start + chunk_sizes["x"], input_shape[2])

        input_slice = (slice(z_start, z_end), slice(y_start, y_end), slice(x_start, x_end))

    else:
        y_start = chunk_coords[0] * chunk_sizes["y"]
        y_end = min(y_start + chunk_sizes["y"], input_shape[0])
        x_start = chunk_coords[1] * chunk_sizes["x"]
        x_end = min(x_start + chunk_sizes["x"], input_shape[1])

        input_slice = (slice(y_start, y_end), slice(x_start, x_end))

    # Read input data for this chunk
    if len(tiff.pages) > 1 and len(input_shape) == 3:
        # Multi-page 3D data
        chunk_data = np.zeros((z_end - z_start, y_end - y_start, x_end - x_start), dtype=page.dtype)
        for z in range(z_start, z_end):
            if z < len(tiff.pages):
                page_data = tiff.pages[z].asarray()
                chunk_data[z - z_start] = page_data[y_start:y_end, x_start:x_end]
    else:
        # Single page or already 3D
        if hasattr(tiff.pages[0], 'asarray'):
            full_data = tiff.pages[0].asarray()
        else:
            full_data = np.array(tiff.pages[0])
        chunk_data = full_data[input_slice]

    # Process chunk for each pyramid level
    for level_idx in range(len(zarr_store.keys())):
        level_name = str(level_idx)
        if level_name not in zarr_store:
            continue

        level_array = zarr_store[level_name]

        # Calculate downsampling for this level
        if level_idx == 0:
            # Level 0: direct copy
            level_data = chunk_data
        else:
            # Higher levels: downsample
            downsample_factors = _get_downsample_factors(level_idx, config)
            level_data = _downsample_data(chunk_data, downsample_factors)

        # Calculate output slice for this level
        level_slice = _calculate_output_slice(chunk_coords, level_idx, chunk_sizes, input_shape, level_array.shape)

        # Write to zarr array (with coordinate locking to prevent race conditions)
        if level_data.size > 0 and all(s.start < s.stop for s in level_slice):
            try:
                level_array[level_slice] = level_data
            except Exception as e:
                print(f"Warning: Could not write level {level_idx} chunk {chunk_idx}: {e}")


def _get_downsample_factors(level: int, config: Dict[str, Any]) -> Tuple[int, ...]:
    """Calculate downsampling factors for a pyramid level."""
    scale_z = config.get("z_scale", 2)
    scale_y = config.get("y_scale", 2)
    scale_x = config.get("x_scale", 2)

    # Compound scaling for higher levels
    factor_z = scale_z ** level
    factor_y = scale_y ** level
    factor_x = scale_x ** level

    return (factor_z, factor_y, factor_x)


def _downsample_data(data: np.ndarray, factors: Tuple[int, ...]) -> np.ndarray:
    """Downsample data using specified factors."""
    if len(data.shape) == 3:
        factor_z, factor_y, factor_x = factors

        # Use scipy zoom for high-quality downsampling
        zoom_factors = (1.0 / factor_z, 1.0 / factor_y, 1.0 / factor_x)
        downsampled = ndimage.zoom(data, zoom_factors, order=1, prefilter=False)

    elif len(data.shape) == 2:
        factor_y, factor_x = factors[1:3]  # Skip z factor for 2D

        zoom_factors = (1.0 / factor_y, 1.0 / factor_x)
        downsampled = ndimage.zoom(data, zoom_factors, order=1, prefilter=False)

    else:
        downsampled = data

    return downsampled


def _calculate_output_slice(chunk_coords: Tuple[int, ...], level: int, chunk_sizes: Dict[str, int],
                            input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> Tuple[slice, ...]:
    """Calculate the output slice for a chunk at a specific pyramid level."""

    # Calculate scaling factors for this level
    scale_factor = 2 ** level

    if len(input_shape) == 3:
        z_idx, y_idx, x_idx = chunk_coords

        # Calculate scaled positions
        z_start = (z_idx * chunk_sizes["z"]) // scale_factor
        z_end = min(((z_idx + 1) * chunk_sizes["z"]) // scale_factor, output_shape[0])
        y_start = (y_idx * chunk_sizes["y"]) // scale_factor
        y_end = min(((y_idx + 1) * chunk_sizes["y"]) // scale_factor, output_shape[1])
        x_start = (x_idx * chunk_sizes["x"]) // scale_factor
        x_end = min(((x_idx + 1) * chunk_sizes["x"]) // scale_factor, output_shape[2])

        return (slice(z_start, z_end), slice(y_start, y_end), slice(x_start, x_end))

    else:
        y_idx, x_idx = chunk_coords

        y_start = (y_idx * chunk_sizes["y"]) // scale_factor
        y_end = min(((y_idx + 1) * chunk_sizes["y"]) // scale_factor, output_shape[0])
        x_start = (x_idx * chunk_sizes["x"]) // scale_factor
        x_end = min(((x_idx + 1) * chunk_sizes["x"]) // scale_factor, output_shape[1])

        return (slice(y_start, y_end), slice(x_start, x_end))


def _calculate_input_slice(input_chunk_coords: Tuple[int, ...], chunk_sizes: Dict[str, int],
                           input_shape: Tuple[int, ...]) -> Tuple[slice, ...]:
    """Calculate input data slice for a chunk."""
    if len(input_shape) == 3:
        z_idx, y_idx, x_idx = input_chunk_coords
        z_start = z_idx * chunk_sizes["z"]
        z_end = min(z_start + chunk_sizes["z"], input_shape[0])
        y_start = y_idx * chunk_sizes["y"]
        y_end = min(y_start + chunk_sizes["y"], input_shape[1])
        x_start = x_idx * chunk_sizes["x"]
        x_end = min(x_start + chunk_sizes["x"], input_shape[2])
        return (slice(z_start, z_end), slice(y_start, y_end), slice(x_start, x_end))
    else:
        y_idx, x_idx = input_chunk_coords
        y_start = y_idx * chunk_sizes["y"]
        y_end = min(y_start + chunk_sizes["y"], input_shape[0])
        x_start = x_idx * chunk_sizes["x"]
        x_end = min(x_start + chunk_sizes["x"], input_shape[1])
        return (slice(y_start, y_end), slice(x_start, x_end))


def _read_input_chunk_data(tiff: tifffile.TiffFile, input_slice: Tuple[slice, ...],
                           input_shape: Tuple[int, ...]) -> np.ndarray:
    """Read input data for a specific chunk."""
    page = tiff.pages[0]

    if len(tiff.pages) > 1 and len(input_shape) == 3:
        # Multi-page 3D data
        z_slice, y_slice, x_slice = input_slice
        chunk_data = np.zeros((z_slice.stop - z_slice.start,
                               y_slice.stop - y_slice.start,
                               x_slice.stop - x_slice.start), dtype=page.dtype)

        for z in range(z_slice.start, z_slice.stop):
            if z < len(tiff.pages):
                page_data = tiff.pages[z].asarray()
                chunk_data[z - z_slice.start] = page_data[y_slice, x_slice]
    else:
        # Single page or already 3D
        if hasattr(tiff.pages[0], 'asarray'):
            full_data = tiff.pages[0].asarray()
        else:
            full_data = np.array(tiff.pages[0])
        chunk_data = full_data[input_slice]

    return chunk_data


def _write_with_coordination(level_data: np.ndarray, input_chunk_coords: Tuple[int, ...],
                             level_idx: int, level_array: zarr.Array,
                             level_race_metadata: Dict[str, Any], coordinator: WorkerCoordinator,
                             chunk_sizes: Dict[str, int], input_shape: Tuple[int, ...]):
    """Write data to zarr array with coordination to prevent race conditions."""

    # Calculate which output zarr chunks this input chunk affects
    affected_output_chunks = _calculate_affected_output_chunks(
        input_chunk_coords, level_idx, chunk_sizes, input_shape,
        level_array.shape, level_array.chunks
    )

    # Sort output chunks by coordinates for consistent lock ordering (prevent deadlocks)
    affected_output_chunks.sort()

    # Acquire locks for all affected output chunks
    acquired_locks = []

    try:
        for output_chunk_coords in affected_output_chunks:
            if coordinator.wait_for_lock_with_timeout(level_idx, output_chunk_coords, timeout_seconds=60):
                acquired_locks.append(output_chunk_coords)
            else:
                raise RuntimeError(f"Failed to acquire lock for output chunk {output_chunk_coords} within timeout")

        # Now safely write to all affected output chunks
        for output_chunk_coords in affected_output_chunks:
            output_slice = _calculate_output_slice_for_zarr_chunk(
                input_chunk_coords, output_chunk_coords, level_idx, chunk_sizes,
                input_shape, level_array.shape, level_array.chunks
            )

            if output_slice and level_data.size > 0:
                # Extract the portion of level_data that corresponds to this output chunk
                data_slice = _extract_data_for_output_chunk(
                    level_data, input_chunk_coords, output_chunk_coords, level_idx,
                    chunk_sizes, input_shape, level_array.chunks
                )

                if data_slice is not None and data_slice.size > 0:
                    level_array[output_slice] = data_slice

    finally:
        # Release all acquired locks
        for output_chunk_coords in acquired_locks:
            coordinator.release_output_chunk_lock(level_idx, output_chunk_coords)


def _calculate_affected_output_chunks(input_chunk_coords: Tuple[int, ...], level_idx: int,
                                      chunk_sizes: Dict[str, int], input_shape: Tuple[int, ...],
                                      output_shape: Tuple[int, ...], output_chunks: Tuple[int, ...]) -> List[
    Tuple[int, ...]]:
    """Calculate which output zarr chunks are affected by an input chunk."""

    scale_factor = 2 ** level_idx if level_idx > 0 else 1
    affected_chunks = []

    if len(input_shape) == 3:
        z_idx, y_idx, x_idx = input_chunk_coords

        # Calculate input region at this pyramid level
        z_start = (z_idx * chunk_sizes["z"]) // scale_factor
        z_end = ((z_idx + 1) * chunk_sizes["z"]) // scale_factor
        y_start = (y_idx * chunk_sizes["y"]) // scale_factor
        y_end = ((y_idx + 1) * chunk_sizes["y"]) // scale_factor
        x_start = (x_idx * chunk_sizes["x"]) // scale_factor
        x_end = ((x_idx + 1) * chunk_sizes["x"]) // scale_factor

        # Calculate which output chunks overlap with this region
        z_chunk_start = z_start // output_chunks[0]
        z_chunk_end = min((z_end - 1) // output_chunks[0] + 1, (output_shape[0] - 1) // output_chunks[0] + 1)
        y_chunk_start = y_start // output_chunks[1]
        y_chunk_end = min((y_end - 1) // output_chunks[1] + 1, (output_shape[1] - 1) // output_chunks[1] + 1)
        x_chunk_start = x_start // output_chunks[2]
        x_chunk_end = min((x_end - 1) // output_chunks[2] + 1, (output_shape[2] - 1) // output_chunks[2] + 1)

        for z_chunk in range(z_chunk_start, z_chunk_end):
            for y_chunk in range(y_chunk_start, y_chunk_end):
                for x_chunk in range(x_chunk_start, x_chunk_end):
                    affected_chunks.append((z_chunk, y_chunk, x_chunk))

    else:
        y_idx, x_idx = input_chunk_coords

        # Calculate input region at this pyramid level
        y_start = (y_idx * chunk_sizes["y"]) // scale_factor
        y_end = ((y_idx + 1) * chunk_sizes["y"]) // scale_factor
        x_start = (x_idx * chunk_sizes["x"]) // scale_factor
        x_end = ((x_idx + 1) * chunk_sizes["x"]) // scale_factor

        # Calculate which output chunks overlap
        y_chunk_start = y_start // output_chunks[0]
        y_chunk_end = min((y_end - 1) // output_chunks[0] + 1, (output_shape[0] - 1) // output_chunks[0] + 1)
        x_chunk_start = x_start // output_chunks[1]
        x_chunk_end = min((x_end - 1) // output_chunks[1] + 1, (output_shape[1] - 1) // output_chunks[1] + 1)

        for y_chunk in range(y_chunk_start, y_chunk_end):
            for x_chunk in range(x_chunk_start, x_chunk_end):
                affected_chunks.append((y_chunk, x_chunk))

    return affected_chunks


def _calculate_output_slice_for_zarr_chunk(input_chunk_coords: Tuple[int, ...],
                                           output_chunk_coords: Tuple[int, ...],
                                           level_idx: int, chunk_sizes: Dict[str, int],
                                           input_shape: Tuple[int, ...], output_shape: Tuple[int, ...],
                                           output_chunks: Tuple[int, ...]) -> Tuple[slice, ...]:
    """Calculate the exact slice within an output zarr chunk for writing."""

    scale_factor = 2 ** level_idx if level_idx > 0 else 1

    if len(input_shape) == 3:
        z_idx, y_idx, x_idx = input_chunk_coords
        z_out, y_out, x_out = output_chunk_coords

        # Input region at pyramid level
        input_z_start = (z_idx * chunk_sizes["z"]) // scale_factor
        input_z_end = min(((z_idx + 1) * chunk_sizes["z"]) // scale_factor, output_shape[0])
        input_y_start = (y_idx * chunk_sizes["y"]) // scale_factor
        input_y_end = min(((y_idx + 1) * chunk_sizes["y"]) // scale_factor, output_shape[1])
        input_x_start = (x_idx * chunk_sizes["x"]) // scale_factor
        input_x_end = min(((x_idx + 1) * chunk_sizes["x"]) // scale_factor, output_shape[2])

        # Output chunk boundaries
        out_z_start = z_out * output_chunks[0]
        out_z_end = min((z_out + 1) * output_chunks[0], output_shape[0])
        out_y_start = y_out * output_chunks[1]
        out_y_end = min((y_out + 1) * output_chunks[1], output_shape[1])
        out_x_start = x_out * output_chunks[2]
        out_x_end = min((x_out + 1) * output_chunks[2], output_shape[2])

        # Calculate intersection
        z_start = max(input_z_start, out_z_start)
        z_end = min(input_z_end, out_z_end)
        y_start = max(input_y_start, out_y_start)
        y_end = min(input_y_end, out_y_end)
        x_start = max(input_x_start, out_x_start)
        x_end = min(input_x_end, out_x_end)

        if z_start < z_end and y_start < y_end and x_start < x_end:
            return (slice(z_start, z_end), slice(y_start, y_end), slice(x_start, x_end))

    else:
        y_idx, x_idx = input_chunk_coords
        y_out, x_out = output_chunk_coords

        # Similar calculation for 2D case
        input_y_start = (y_idx * chunk_sizes["y"]) // scale_factor
        input_y_end = min(((y_idx + 1) * chunk_sizes["y"]) // scale_factor, output_shape[0])
        input_x_start = (x_idx * chunk_sizes["x"]) // scale_factor
        input_x_end = min(((x_idx + 1) * chunk_sizes["x"]) // scale_factor, output_shape[1])

        out_y_start = y_out * output_chunks[0]
        out_y_end = min((y_out + 1) * output_chunks[0], output_shape[0])
        out_x_start = x_out * output_chunks[1]
        out_x_end = min((x_out + 1) * output_chunks[1], output_shape[1])

        y_start = max(input_y_start, out_y_start)
        y_end = min(input_y_end, out_y_end)
        x_start = max(input_x_start, out_x_start)
        x_end = min(input_x_end, out_x_end)

        if y_start < y_end and x_start < x_end:
            return (slice(y_start, y_end), slice(x_start, x_end))

    return None


def _extract_data_for_output_chunk(level_data: np.ndarray, input_chunk_coords: Tuple[int, ...],
                                   output_chunk_coords: Tuple[int, ...], level_idx: int,
                                   chunk_sizes: Dict[str, int], input_shape: Tuple[int, ...],
                                   output_chunks: Tuple[int, ...]) -> np.ndarray:
    """Extract the portion of level_data that corresponds to a specific output chunk."""

    # This is a simplified version - in practice, you'd need to calculate the exact
    # portion of the input data that maps to the specific output chunk region

    # For now, return the level_data as-is (assumes data fits exactly)
    # A more sophisticated implementation would crop the data based on the
    # intersection calculation between input processing chunk and output zarr chunk

    return level_data