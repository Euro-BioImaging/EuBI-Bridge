"""
SLURM Coordinator Functions with Race Condition Protection
Handles zarr store initialization and task queue creation for safe distributed processing.
"""

import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple
import numpy as np
import tifffile
import zarr
from .ngff_metadata import OMENGFFMetadataGenerator


def initialize_zarr_store(input_tiff: str, output_zarr_dir: str, config: Dict[str, Any]):
    """Initialize the output zarr store structure with race condition protection metadata."""
    print(f"Initializing zarr store with race condition protection: {output_zarr_dir}")

    # Analyze input file
    with tifffile.TiffFile(input_tiff) as tiff:
        page = tiff.pages[0]
        shape = page.shape
        dtype = page.dtype

        # Determine full shape including z-dimension if multiple pages
        if len(tiff.pages) > 1:
            if len(shape) == 2:
                full_shape = (len(tiff.pages), shape[0], shape[1])
            else:
                full_shape = shape
        else:
            full_shape = shape

    # Create zarr store
    output_path = Path(output_zarr_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine zarr version and create store appropriately
    zarr_format = config.get("zarr_format", 2)
    if zarr_format == 3:
        store = zarr.open(str(output_path), mode='w')
    else:
        store = zarr.open(str(output_path), mode='w')

    # Calculate pyramid levels
    pyramid_scales = {
        "z": config.get("z_scale", 2),
        "y": config.get("y_scale", 2),
        "x": config.get("x_scale", 2)
    }

    n_layers = config.get("n_layers")
    min_dimension_size = config.get("min_dimension_size", 64)

    pyramid_levels = _calculate_pyramid_levels(full_shape, pyramid_scales, n_layers, min_dimension_size)

    # Get chunk sizes from config
    input_chunk_sizes = {
        "z": config.get("z_chunk", 96),
        "y": config.get("y_chunk", 96),
        "x": config.get("x_chunk", 96)
    }

    # Create arrays for each pyramid level with optimal chunk sizes
    compression_mapping = {
        "blosc": "blosc",
        "blosc2-lz4": "blosc",
        "blosc2-zstd": "blosc",
        "lz4": "lz4",
        "none": None
    }

    compressor_name = compression_mapping.get(config.get("compression", "blosc"), "blosc")

    # Calculate race condition metadata for each level
    race_condition_metadata = {}

    for level, level_shape in enumerate(pyramid_levels):
        level_name = str(level)

        # Calculate output chunk sizes for this level (zarr native chunks)
        output_chunks = []
        for i, dim_size in enumerate(level_shape):
            if i == 0 and len(level_shape) == 3:  # z dimension
                chunk_size = min(64, dim_size)  # Smaller zarr chunks for better I/O
            elif i == (len(level_shape) - 2):  # y dimension
                chunk_size = min(64, dim_size)
            elif i == (len(level_shape) - 1):  # x dimension
                chunk_size = min(64, dim_size)
            else:
                chunk_size = min(64, dim_size)

            output_chunks.append(chunk_size)

        output_chunks = tuple(output_chunks)

        # Calculate race condition protection metadata
        race_metadata = _calculate_race_condition_metadata(
            level, level_shape, input_chunk_sizes, output_chunks, pyramid_scales
        )
        race_condition_metadata[level_name] = race_metadata

        # Create level array
        compressor = None
        if compressor_name == "blosc":
            import numcodecs
            compressor = numcodecs.Blosc(cname='lz4', clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)
        elif compressor_name == "lz4":
            import numcodecs
            compressor = numcodecs.LZ4()

        level_array = store.create_dataset(
            level_name,
            shape=level_shape,
            chunks=output_chunks,
            dtype=dtype,
            compressor=compressor,
            overwrite=True
        )

        print(f"Created level {level}: shape={level_shape}, chunks={output_chunks}")

    # Save race condition metadata for workers
    race_metadata_file = output_path / "race_condition_metadata.json"
    with open(race_metadata_file, 'w') as f:
        json.dump(race_condition_metadata, f, indent=2)

    print(f"✅ Race condition metadata saved for {len(pyramid_levels)} levels")

    # Generate and save NGFF metadata
    metadata_generator = OMENGFFMetadataGenerator()

    # Extract pixel sizes from TIFF metadata
    pixel_sizes = {}
    try:
        if hasattr(page, 'resolution'):
            res_x, res_y = page.resolution
            if hasattr(page, 'resolutionunit') and page.resolutionunit == 2:  # Inches
                pixel_sizes['x'] = 25400.0 / res_x if res_x > 0 else 1.0  # Convert to micrometers
                pixel_sizes['y'] = 25400.0 / res_y if res_y > 0 else 1.0
            else:
                pixel_sizes['x'] = 1.0 / res_x if res_x > 0 else 1.0
                pixel_sizes['y'] = 1.0 / res_y if res_y > 0 else 1.0
        else:
            pixel_sizes = {'x': 1.0, 'y': 1.0}
    except:
        pixel_sizes = {'x': 1.0, 'y': 1.0}

    # Add z pixel size if 3D
    if len(full_shape) == 3:
        pixel_sizes['z'] = 1.0

    ngff_metadata = metadata_generator.generate_metadata(
        pyramid_levels=pyramid_levels,
        pixel_sizes=pixel_sizes,
        dimension_order=config.get("dimension_order", "zyx" if len(full_shape) == 3 else "yx"),
        pyramid_scales=pyramid_scales,
        dtype=str(dtype)
    )

    # Save metadata
    with open(output_path / ".zattrs", 'w') as f:
        json.dump(ngff_metadata, f, indent=2)

    print(f"✅ Zarr store initialized with {len(pyramid_levels)} levels and race condition protection")


def create_task_queue(work_dir: str, processing_plan: Dict[str, Any]):
    """Create task queue for worker coordination with race condition protection."""
    print("Creating task queue with race condition protection...")

    work_path = Path(work_dir)
    total_chunks = processing_plan["total_chunks"]
    recommended_nodes = processing_plan["recommended_nodes"]
    chunks_per_node = processing_plan["chunks_per_node"]

    # Create task assignments for each worker with overlap detection
    tasks = []
    chunk_assignments = {}  # Track which chunks are assigned to which workers

    for worker_id in range(recommended_nodes):
        start_chunk = worker_id * chunks_per_node
        end_chunk = min(start_chunk + chunks_per_node, total_chunks)

        if start_chunk < total_chunks:
            chunk_range = list(range(start_chunk, end_chunk))

            task = {
                "worker_id": worker_id,
                "start_chunk": start_chunk,
                "end_chunk": end_chunk,
                "chunk_range": chunk_range,
                "total_chunks_assigned": len(chunk_range)
            }
            tasks.append(task)

            # Track assignments for overlap detection
            for chunk_id in chunk_range:
                if chunk_id not in chunk_assignments:
                    chunk_assignments[chunk_id] = []
                chunk_assignments[chunk_id].append(worker_id)

    # Detect potential overlaps (should not happen with current partitioning)
    overlapping_chunks = {chunk_id: workers for chunk_id, workers in chunk_assignments.items() if len(workers) > 1}
    if overlapping_chunks:
        print(f"⚠️ Warning: Detected overlapping chunk assignments: {overlapping_chunks}")

    # Generate coordination metadata
    coordination_metadata = {
        "total_workers": len(tasks),
        "total_chunks": total_chunks,
        "chunk_assignments": chunk_assignments,
        "overlapping_chunks": overlapping_chunks,
        "processing_plan": processing_plan
    }

    # Save task queue with coordination metadata
    task_queue_file = work_path / "task_queue.json"
    with open(task_queue_file, 'w') as f:
        json.dump({
            "tasks": tasks,
            "coordination_metadata": coordination_metadata
        }, f, indent=2)

    # Save processing plan
    plan_file = work_path / "processing_plan.json"
    with open(plan_file, 'w') as f:
        json.dump(processing_plan, f, indent=2)

    print(f"✅ Created task queue with {len(tasks)} worker assignments and race condition protection")

    # Create coordination lock directory for distributed locking
    lock_dir = work_path / "coordination_locks"
    lock_dir.mkdir(exist_ok=True)

    print(f"✅ Coordination lock directory created: {lock_dir}")


def _calculate_race_condition_metadata(level: int, level_shape: Tuple[int, ...],
                                       input_chunk_sizes: Dict[str, int],
                                       output_chunks: Tuple[int, ...],
                                       pyramid_scales: Dict[str, int]) -> Dict[str, Any]:
    """Calculate metadata needed for race condition protection at a specific pyramid level."""

    # Calculate scaling factor for this level
    scale_factor = 2 ** level if level > 0 else 1

    # Calculate effective input chunk sizes at this level
    effective_input_chunks = {}
    if len(level_shape) == 3:
        effective_input_chunks = {
            "z": max(1, input_chunk_sizes["z"] // (pyramid_scales["z"] ** level)),
            "y": max(1, input_chunk_sizes["y"] // (pyramid_scales["y"] ** level)),
            "x": max(1, input_chunk_sizes["x"] // (pyramid_scales["x"] ** level))
        }
    else:
        effective_input_chunks = {
            "y": max(1, input_chunk_sizes["y"] // (pyramid_scales["y"] ** level)),
            "x": max(1, input_chunk_sizes["x"] // (pyramid_scales["x"] ** level))
        }

    # Determine if input and output chunks are aligned
    alignment_info = _analyze_chunk_alignment(effective_input_chunks, output_chunks, level_shape)

    race_metadata = {
        "level": level,
        "scale_factor": scale_factor,
        "level_shape": level_shape,
        "effective_input_chunks": effective_input_chunks,
        "output_chunks": output_chunks,
        "alignment_info": alignment_info,
        "requires_coordination": alignment_info["misaligned"]
    }

    return race_metadata


def _analyze_chunk_alignment(input_chunks: Dict[str, int], output_chunks: Tuple[int, ...],
                             shape: Tuple[int, ...]) -> Dict[str, Any]:
    """Analyze alignment between input processing chunks and output zarr chunks."""

    alignment_info = {
        "aligned": True,
        "misaligned": False,
        "overlap_patterns": {},
        "coordination_required": False
    }

    if len(shape) == 3:
        # 3D case
        input_chunk_tuple = (input_chunks["z"], input_chunks["y"], input_chunks["x"])
    else:
        # 2D case
        input_chunk_tuple = (input_chunks["y"], input_chunks["x"])

    # Check if input and output chunks have different sizes
    if input_chunk_tuple != output_chunks:
        alignment_info["aligned"] = False
        alignment_info["misaligned"] = True
        alignment_info["coordination_required"] = True

        # Calculate potential overlap patterns
        for i, (input_size, output_size) in enumerate(zip(input_chunk_tuple, output_chunks)):
            if input_size != output_size:
                # Calculate how many output chunks an input chunk might span
                spans_multiple = input_size > output_size
                fractional_coverage = input_size % output_size != 0 if input_size > output_size else output_size % input_size != 0

                alignment_info["overlap_patterns"][f"dim_{i}"] = {
                    "input_size": input_size,
                    "output_size": output_size,
                    "spans_multiple": spans_multiple,
                    "fractional_coverage": fractional_coverage
                }

    return alignment_info


def _calculate_pyramid_levels(shape: tuple, scales: Dict[str, int], n_layers: int = None, min_size: int = 64) -> List[
    tuple]:
    """Calculate pyramid level shapes."""
    levels = []
    current_shape = shape

    # Level 0 is always the original shape
    levels.append(current_shape)

    level = 1
    while True:
        # Calculate next level shape
        next_shape = []
        for i, dim_size in enumerate(current_shape):
            if i == 0 and len(current_shape) == 3:  # z dimension
                scale_factor = scales.get("z", 2)
            elif i == (len(current_shape) - 2):  # y dimension
                scale_factor = scales.get("y", 2)
            elif i == (len(current_shape) - 1):  # x dimension
                scale_factor = scales.get("x", 2)
            else:
                scale_factor = 2

            next_dim_size = max(1, dim_size // scale_factor)
            next_shape.append(next_dim_size)

        next_shape = tuple(next_shape)

        # Stop if any dimension is too small or we've reached requested levels
        if any(dim < min_size for dim in next_shape):
            break

        if n_layers and level >= n_layers:
            break

        levels.append(next_shape)
        current_shape = next_shape
        level += 1

    return levels