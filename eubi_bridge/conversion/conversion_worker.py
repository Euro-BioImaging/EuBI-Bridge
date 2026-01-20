"""
Conversion workers for image data processing with zarr storage.

This module provides async workers for converting image data to zarr format,
supporting both unary (single-file) and aggregative (multi-file) conversion modes.
"""

import asyncio
import os
import sys
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

# Add these imports at the top of conversion_worker.py
from eubi_bridge.conversion.worker_init import safe_worker_wrapper
from eubi_bridge.core.data_manager import ArrayManager
from eubi_bridge.core.writers import store_multiscale_async
from eubi_bridge.utils.array_utils import autocompute_chunk_shape
from eubi_bridge.utils.jvm_manager import soft_start_jvm
from eubi_bridge.utils.logging_config import get_logger
from eubi_bridge.utils.metadata_utils import parse_channels
from eubi_bridge.utils.path_utils import (is_zarr_array, is_zarr_group,
                                          sensitive_glob, take_filepaths)

# soft_start_jvm()




logger = get_logger(__name__)

# Constants
DEFAULT_CHUNKS = {
    'time_chunk': 1,
    'channel_chunk': 1,
    'z_chunk': 96,
    'y_chunk': 96,
    'x_chunk': 96,
}

DEFAULT_SHARD_COEFS = {
    'time_shard_coef': 1,
    'channel_shard_coef': 1,
    'z_shard_coef': 5,
    'y_shard_coef': 5,
    'x_shard_coef': 5,
}

DEFAULT_SCALE_FACTORS = {
    'time_scale_factor': 1,
    'channel_scale_factor': 1,
    'z_scale_factor': 2,
    'y_scale_factor': 2,
    'x_scale_factor': 2,
}

AXIS_PARAM_MAP = {
    't': ('time_chunk', 'time_shard_coef', 'time_scale', 'time_scale_factor', 'time_unit'),
    'c': ('channel_chunk', 'channel_shard_coef', 'channel_scale', 'channel_scale_factor', None),
    'z': ('z_chunk', 'z_shard_coef', 'z_scale', 'z_scale_factor', 'z_unit'),
    'y': ('y_chunk', 'y_shard_coef', 'y_scale', 'y_scale_factor', 'y_unit'),
    'x': ('x_chunk', 'x_shard_coef', 'x_scale', 'x_scale_factor', 'x_unit'),
}

CROPPING_PARAMS = {'time_range', 'channel_range', 'z_range', 'y_range', 'x_range'}


def _get_param_value(kwargs: Dict, param_name: Optional[str],
                     axis: str, default_dict: Dict) -> Any:
    """
    Extract parameter value from kwargs with fallback to defaults.

    Args:
        kwargs: Keyword arguments dictionary
        param_name: Parameter name to look up
        axis: Axis identifier for fallback
        default_dict: Default values dictionary

    Returns:
        Parameter value or default
    """
    if param_name is None:
        return None

    value = kwargs.get(param_name)
    if pd.isna(value):
        # For scale/unit params, use axis as key; for others use param_name
        default_key = axis if param_name.endswith(('_scale', '_unit')) else param_name
        return default_dict.get(default_key)
    return value


def _parse_axis_params(manager: ArrayManager, kwargs: Dict,
                       param_idx: int, default_dict: Dict) -> Tuple:
    """
    Generic parser for axis-based parameters.

    Args:
        manager: ArrayManager instance
        kwargs: Keyword arguments
        param_idx: Index in AXIS_PARAM_MAP tuple (0=chunk, 1=shard, etc.)
        default_dict: Default values dictionary

    Returns:
        Tuple of parsed values for each axis
    """
    axes = manager.axes #if param_idx <= 1 else manager.caxes
    output = {}

    for axis in axes:
        if axis not in AXIS_PARAM_MAP:
            continue

        param_name = AXIS_PARAM_MAP[axis][param_idx]
        if param_name is None:  # Skip channel unit
            continue

        output[axis] = _get_param_value(kwargs, param_name, axis, default_dict)

    return tuple(output[ax] for ax in axes if ax in output)


def parse_chunks(manager: ArrayManager, **kwargs) -> Tuple:
    """Parse chunk sizes for each axis.
    
    Extracts chunk size parameters from kwargs for each dimension in the array.
    Uses default chunk sizes if not specified.
    
    Parameters
    ----------
    manager : ArrayManager
        Array manager containing dimension information.
    **kwargs
        Keyword arguments containing axis-specific chunk sizes.
        
    Returns
    -------
    Tuple
        Chunk sizes for each axis in the array.
    """
    return _parse_axis_params(manager, kwargs, 0, DEFAULT_CHUNKS)


def parse_shard_coefs(manager: ArrayManager, **kwargs) -> Tuple:
    """Parse shard coefficients for each axis.
    
    Shard coefficients control how chunks are organized into shards
    for optimized storage and access patterns.
    
    Parameters
    ----------
    manager : ArrayManager
        Array manager containing dimension information.
    **kwargs
        Keyword arguments containing shard coefficient values.
        
    Returns
    -------
    Tuple
        Shard coefficients for each axis.
    """
    return _parse_axis_params(manager, kwargs, 1, DEFAULT_SHARD_COEFS)


def parse_scales(manager: ArrayManager, **kwargs) -> Tuple:
    """Parse scale values for each axis.
    
    Scale values define the physical size of pixels/voxels along each dimension,
    essential for NGFF metadata and proper image interpretation.
    
    Parameters
    ----------
    manager : ArrayManager
        Array manager containing dimension and scale information.
    **kwargs
        Keyword arguments containing axis-specific scale values.
        
    Returns
    -------
    Tuple
        Scale values for each axis.
    """
    return _parse_axis_params(manager, kwargs, 2, manager.scaledict)


def parse_scale_factors(manager: ArrayManager, **kwargs) -> Tuple:
    """Parse scale factors for each axis.
    
    Scale factors determine the downsampling ratio for each pyramid level.
    Defines how dimensions shrink as we move down the resolution hierarchy.
    
    Parameters
    ----------
    manager : ArrayManager
        Array manager containing dimension information.
    **kwargs
        Keyword arguments containing axis-specific scale factor values.
        
    Returns
    -------
    Tuple
        Scale factors for each axis in the pyramid.
    """
    return _parse_axis_params(manager, kwargs, 3, DEFAULT_SCALE_FACTORS)


def parse_units(manager: ArrayManager, **kwargs) -> Tuple:
    """Parse unit values for each axis (excluding channel).
    
    Units specify the physical measurement unit for each dimension
    (e.g., 'micrometer' for spatial, 'second' for temporal).
    Channel axis always has no physical unit.
    
    Parameters
    ----------
    manager : ArrayManager
        Array manager containing unit information.
    **kwargs
        Keyword arguments containing axis-specific unit values.
        
    Returns
    -------
    Tuple
        Unit strings for each non-channel axis.
    """
    return _parse_axis_params(manager, kwargs, 4, manager.unitdict)


def _extract_cropping_slices(kwargs: Dict) -> list:
    """Extract cropping range parameters from kwargs.
    
    Collects all cropping-related keyword arguments that define
    rectangular regions of interest in the dataset.
    
    Parameters
    ----------
    kwargs : Dict
        Keyword arguments containing potential cropping parameters.
        
    Returns
    -------
    list
        List of cropping parameter values.
    """
    return [kwargs.get(key) for key in kwargs if key in CROPPING_PARAMS]


async def _prepare_manager(manager: ArrayManager, kwargs: Dict) -> None:
    """
    Apply preprocessing steps to manager.

    Args:
        manager: ArrayManager to prepare
        kwargs: Configuration parameters
    """
    manager.fill_default_meta()
    await manager.sync_pyramid(save_changes=False)

    # Parse and fix channel metadata
    manager._channels = parse_channels(
        manager,
        channel_intensity_limits='from_dtype'
    )
    
    manager.fix_bad_channels()
    if kwargs.get('verbose', False): # TODO: Verify the verbosity check.
        logger.info(f"The manager array shape before squeezing: {manager.array.shape}")
    # Apply optional transformations
    if kwargs.get('squeeze'):
        manager.squeeze()

    cropping_slices = _extract_cropping_slices(kwargs)
    if any(cropping_slices):
        manager.crop(*cropping_slices)


async def _update_channel_metadata(output_path: str, kwargs: Dict) -> None:
    """
    Update channel metadata from array data.

    Args:
        output_path: Path to zarr output
        kwargs: Configuration parameters
    """
    if kwargs.get('channel_intensity_limits', 'from_array') != 'from_array':
        return

    skip_dask = kwargs.get('skip_dask', True)
    chman = ArrayManager(output_path, skip_dask=skip_dask)
    await chman.init()
    chman.fill_default_meta()
    await chman.sync_pyramid(save_changes=False)

    if kwargs.get('squeeze'):
        chman.squeeze()

    channels = parse_channels(chman, **kwargs)
    meta = chman.pyr.meta
    meta.metadata['omero']['channels'] = channels

    # Prevent serialization errors
    if meta.zarr_group is not None:
        if 'ome' not in meta.zarr_group.attrs:
            meta.zarr_group.attrs.update({'omero': []})

    meta._pending_changes = True
    meta.save_changes()


async def _process_single_scene(manager: ArrayManager, output_path: str,
                                kwargs: Dict, sem: asyncio.Semaphore) -> None:
    """
    Process a single scene/tile.

    Args:
        manager: ArrayManager for this scene
        output_path: Output zarr path
        kwargs: Configuration parameters
        sem: Semaphore for concurrency control
    """
    async with sem:
        if kwargs.get('verbose', False):
            logger.info(f"The manager array shape before preparation: {manager.array.shape}")
        await _prepare_manager(manager, kwargs)

        scale_factors = parse_scale_factors(manager, **kwargs)

        # Store multiscale data
        channel_meta = parse_channels(
            manager,
            **dict(kwargs, channel_intensity_limits='from_dtype')
        )
        if kwargs.get('verbose', False): # TODO: Verify the verbosity here.
            logger.info(f"The manager array shape before storing: {manager.array.shape}")
        #import pprint
        #pprint.pprint(f"Channel metadata before storing: {channel_meta}")
        await store_multiscale_async(
            arr=manager.array,
            dtype=kwargs.get('dtype'),
            output_path=output_path,
            zarr_format=kwargs.get('zarr_format', 2),
            axes=manager.axes,
            scales=parse_scales(manager, **kwargs),
            units=parse_units(manager, **kwargs),
            channel_meta=channel_meta,
            auto_chunk=kwargs.get('auto_chunk', True),
            output_chunks=parse_chunks(manager, **kwargs),
            output_shard_coefficients=parse_shard_coefs(manager, **kwargs),
            overwrite=kwargs.get('overwrite', True),
            n_layers=kwargs.get('n_layers', 3),
            min_dimension_size=kwargs.get('min_dimension_size', 64),
            scale_factors=parse_scale_factors(manager, **kwargs),
            max_concurrency=kwargs.get('max_concurrency', 4),
            region_size_mb=kwargs.get('region_size_mb', 128),
            compute_batch_size=kwargs.get('compute_batch_size', 4),
            queue_size=kwargs.get('queue_size', 8),
            memory_limit_per_batch=kwargs.get('memory_limit_per_batch', 1024),
            verbose=kwargs.get('verbose', False),
            compressor=kwargs.get('compressor', 'blosc'),
            compressor_params=kwargs.get('compressor_params', {})
        )

        # Update channel metadata if needed
        await _update_channel_metadata(output_path, kwargs)

        save_omexml = kwargs.get('save_omexml', True)
        # Save OME-XML metadata if needed
        if save_omexml:
            await manager.save_omexml(output_path, overwrite=True)


def _generate_output_path(base_path: str, series_path: str,
                          series_idx: Optional[int] = None,
                          tile_idx: Optional[int] = None) -> str:
    """
    Generate output path with optional scene/tile suffixes.

    Args:
        base_path: Base output directory
        series_path: Source series path
        series_idx: Optional scene index
        tile_idx: Optional tile index

    Returns:
        Complete output path
    """
    basename = os.path.basename(series_path).split('.')[0]
    suffix = ""

    if series_idx is not None:
        suffix += f"_{series_idx}"
    if tile_idx is not None:
        suffix += f"_tile{tile_idx}"

    return f"{base_path}/{basename}{suffix}.zarr"


async def _load_input_manager(input_path: Union[str, ArrayManager],
                              kwargs: Dict) -> ArrayManager:
    """
    Load or validate input ArrayManager.

    Args:
        input_path: File path or existing ArrayManager
        kwargs: Configuration parameters

    Returns:
        Initialized ArrayManager
    """
    if isinstance(input_path, ArrayManager):
        return input_path

    manager = ArrayManager(
        input_path,
        series=0,
        metadata_reader=kwargs.get('metadata_reader', 'bfio'),
        skip_dask=kwargs.get('skip_dask', True),
    )
    await manager.init()
    
    manager.fill_default_meta()
    if kwargs.get('verbose', False):
        logger.info(f"The manager array is of type: {type(manager.array)}")   

    # Load scenes
    series = kwargs.get('scene_index', 'all')
    await manager.load_scenes(scene_indices=series)

    # Load tiles if specified
    mosaic_tile_index = kwargs.get('mosaic_tile_index')
    if mosaic_tile_index is not None and len(manager.loaded_scenes) > 1:
        logger.warning(f"Currently cannot load multiple scenes and multiple tiles at the same time.\n"
                    f"Will load only the first tile.")
    elif mosaic_tile_index is not None and len(manager.loaded_scenes) <= 1:
        await manager.load_tiles(tile_indices=mosaic_tile_index)

    return manager


async def unary_worker(input_path: Union[str, ArrayManager],
                       output_path: str,
                       **kwargs) -> None:
    """
    Convert individual image files to zarr format.

    Handles multiple scenes and mosaic tiles, processing each independently.

    Args:
        input_path: Input file path or ArrayManager
        output_path: Output directory path
        **kwargs: Configuration parameters including:
            - max_concurrency: Maximum concurrent operations (default: 4)
            - scene_index: Scene(s) to process (default: 'all')
            - mosaic_tile_index: Tile(s) to extract (optional)
            - squeeze: Remove singleton dimensions (default: False)
            - Various chunking, scaling, and metadata parameters
    """
    manager = await _load_input_manager(input_path, kwargs)

    # Setup concurrency control
    max_concurrency = kwargs.get('max_concurrency', 4)
    sem = asyncio.Semaphore(max_concurrency)

    # Determine if suffixes needed
    add_scene = manager.img.n_scenes > 1

    # Process all loaded scenes/tiles
    tasks = []
    if manager.loaded_scenes is None:
        raise ValueError("At least one scene must be available.")
    elif manager.loaded_scenes is not None and manager.loaded_tiles is None:
        for man in manager.loaded_scenes.values():
            # Process scene directly
            out_path = _generate_output_path(
                output_path,
                man.series_path,
                man.series if add_scene else None
            )
            tasks.append(asyncio.create_task(
                _process_single_scene(man, out_path, kwargs, sem)
            ))
    elif len(manager.loaded_scenes) == 1 and manager.loaded_tiles is not None:
        # Process each tile
        n_tiles = manager.img.n_tiles or 1
        add_tile = n_tiles > 1
        for tile in manager.loaded_tiles.values():
            out_path = _generate_output_path(
                output_path,
                tile.series_path,
                tile.mosaic_tile_index if add_tile else None
            )
            tasks.append(asyncio.create_task(
                _process_single_scene(tile, out_path, kwargs, sem)
            ))
    else:
        logger.error(f"Having both multiple scenes and multiple tiles is currently not supported.")
        raise Exception(f"Having both multiple scenes and multiple tiles is not currently supported.")

    await asyncio.gather(*tasks)


async def aggregative_worker(manager: ArrayManager,
                             output_path: str,
                             **kwargs) -> None:
    """
    Convert aggregated image data to zarr format.

    Processes a single scene from a pre-configured ArrayManager.

    Args:
        manager: Pre-configured ArrayManager
        output_path: Output file path (without .zarr extension)
        **kwargs: Configuration parameters (see unary_worker)

    Raises:
        TypeError: If series parameter is not an integer
    """
    series = kwargs.get('series', 0)
    if not isinstance(series, int):
        raise TypeError(
            "Aggregative conversion does not support multiple series. "
            "Please specify an integer as a single series index."
        )

    max_concurrency = kwargs.get('max_concurrency', 4)
    sem = asyncio.Semaphore(max_concurrency)

    output_path_full = f"{output_path}.zarr"
    if kwargs.get('verbose', False):
        logger.info(f"The manager array is of type: {type(manager.array)}")
    await _process_single_scene(manager, output_path_full, kwargs, sem)


# # Synchronous wrappers for multiprocessing
# def unary_worker_sync(input_path: Union[str, ArrayManager],
#                       output_path: str,
#                       kwargs: dict) -> None:
#     """Synchronous wrapper for unary_worker."""
#     return asyncio.run(unary_worker(input_path, output_path, **kwargs))
#
#
# def aggregative_worker_sync(input_path: Union[str, ArrayManager],
#                             output_path: str,
#                             kwargs: dict) -> None:
#     """Synchronous wrapper for aggregative_worker."""
#     return asyncio.run(aggregative_worker(input_path, output_path, **kwargs))



# Replace your existing synchronous wrappers at the bottom with these:

@safe_worker_wrapper
def unary_worker_sync(input_path: Union[str, ArrayManager],
                      output_path: str,
                      kwargs: dict) -> dict:
    """
    Synchronous wrapper for unary_worker.
    Safe for multiprocessing with proper exception handling.
    """
    if kwargs.get('verbose', False):
        logger.info(f"[Worker] Processing: {input_path}")

    # Run the async worker
    asyncio.run(unary_worker(input_path, output_path, **kwargs))

    if kwargs.get('verbose', False):
        logger.info(f"[Worker] Completed: {input_path}")

    return {"status": "success", "input": str(input_path), "output": output_path}


@safe_worker_wrapper
def aggregative_worker_sync(manager: ArrayManager,
                            output_path: str,
                            kwargs: dict) -> dict:
    """
    Synchronous wrapper for aggregative_worker.
    Safe for multiprocessing with proper exception handling.
    """
    if kwargs.get('verbose', False):
        logger.info(f"[Worker] Processing aggregative: {output_path}")

    asyncio.run(aggregative_worker(manager, output_path, **kwargs))

    if kwargs.get('verbose', False):
        logger.info(f"[Worker] Completed aggregative: {output_path}")  
    return {"status": "success", "output": output_path}


@safe_worker_wrapper
def metadata_reader_sync(input_path: Union[str, ArrayManager],
                         kwargs: dict) -> dict:
    """
    Synchronous worker for reading metadata from a single image file.
    
    Initializes an ArrayManager for the input file, reads its metadata,
    and returns a structured dictionary of metadata information.
    
    Args:
        input_path: Path to image file
        kwargs: Job parameters (series, skip_dask, metadata_reader, etc.)
    
    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - input_path: Input file path
        - series: Series index
        - axes: Axis order (e.g., 'tczyx')
        - shape: Array shape dictionary
        - scale: Scale factors dictionary
        - units: Units dictionary
        - dtype: Data type
        - channels: List of channel metadata dictionaries
        - error: Error message (if status is "error")
    """
    if kwargs.get('verbose', False):
        logger.info(f"[MetadataReader] Reading metadata: {input_path}")

    try:
        # Extract relevant parameters
        series = kwargs.get('series', 0)
        skip_dask = kwargs.get('skip_dask', False)
        metadata_reader = kwargs.get('metadata_reader', 'bfio')

        # Initialize ArrayManager
        manager = ArrayManager( # TODO: implement a new metadata reader for this functionality. ArrayManager is for conversion and completes missing axes!
            path=input_path,
            series=series,
            skip_dask=skip_dask,
            metadata_reader=metadata_reader,
        )

        # Initialize the manager (load metadata from file)
        asyncio.run(manager.init())
        
        # DEBUG: Check channels after init
        logger.info(f"[DEBUG] After manager.init():")
        logger.info(f"  manager.omemeta = {manager.omemeta}")
        logger.info(f"  manager._channels = {manager._channels}")
        logger.info(f"  manager.channels property = {manager.channels}")
        
        # Fill default metadata
        manager.fill_default_meta()
        manager.fix_bad_channels()

        # Extract metadata
        metadata = {
            "status": "success",
            "input_path": str(input_path),
            "series": series,
            "axes": manager.axes,
            "shape": dict(manager.shapedict),
            "scale": dict(manager.scaledict),
            "units": dict(manager.unitdict),
            "dtype": str(manager.array.dtype),
            "channels": manager.channels,
            "ndim": manager.ndim,
        }

        if kwargs.get('verbose', False):
            logger.info(f"[MetadataReader] Completed: {input_path}")

        return metadata

    except Exception as e:
        logger.error(f"[MetadataReader] Error reading {input_path}: {str(e)}")
        return {
            "status": "error",
            "input_path": str(input_path),
            "error": str(e),
        }