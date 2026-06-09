"""
Conversion workers for image data processing with zarr storage.

This module provides async workers for converting image data to zarr format,
supporting both unary (single-file) and aggregative (multi-file) conversion modes.
"""

import asyncio
import os
import sys
from typing import Any, Dict, Optional, Tuple, Union

import dask.array as da
import pandas as pd
import psutil

# Add these imports at the top of conversion_worker.py
from eubi_bridge.conversion.worker_init import safe_worker_wrapper
from eubi_bridge.core.config_models import ChunkConfig, ConversionJob
from eubi_bridge.core.data_manager import ArrayManager, _compute_tile_shape
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


def parse_chunks(manager: ArrayManager, job: ConversionJob) -> Tuple:
    """Parse chunk sizes for each axis from the job's ConversionConfig."""
    cfg = job.conversion
    return ChunkConfig(
        t=cfg.time_chunk, c=cfg.channel_chunk,
        z=cfg.z_chunk, y=cfg.y_chunk, x=cfg.x_chunk,
    ).as_tuple(manager.axes)


def parse_shard_coefs(manager: ArrayManager, job: ConversionJob) -> Tuple:
    """Parse shard coefficients for each axis from the job's ConversionConfig."""
    cfg = job.conversion
    shard_map = {
        't': cfg.time_shard_coef,
        'c': cfg.channel_shard_coef,
        'z': cfg.z_shard_coef,
        'y': cfg.y_shard_coef,
        'x': cfg.x_shard_coef,
    }
    return tuple(shard_map[ax] for ax in manager.axes if ax in shard_map)


def parse_scales(manager: ArrayManager, job: ConversionJob) -> Tuple:
    """Parse per-axis scale overrides from job.extra, falling back to manager.scaledict."""
    return _parse_axis_params(manager, job.extra, 2, manager.scaledict)


def parse_scale_factors(manager: ArrayManager, job: ConversionJob) -> Tuple:
    """Parse pyramid scale factors from the job's DownscaleConfig."""
    ds = job.downscale
    sf_map = {
        't': ds.time_scale_factor,
        'c': ds.channel_scale_factor,
        'z': ds.z_scale_factor,
        'y': ds.y_scale_factor,
        'x': ds.x_scale_factor,
    }
    return tuple(sf_map[ax] for ax in manager.axes if ax in sf_map)


def parse_smart_scale_factors(manager: ArrayManager, job: ConversionJob) -> Optional[Tuple]:
    """Compute isotropic scale factors when ``apply_smart_downscaling`` is set in job.extra."""
    if not job.extra.get('apply_smart_downscaling', False):
        return None

    from eubi_bridge.core.scale import compute_isotropic_scale_factors

    actual_scales = parse_scales(manager, job)
    axes = manager.axes
    auto_factors = compute_isotropic_scale_factors(
        pixel_sizes=dict(zip(axes, actual_scales)),
        axes=axes,
    )

    for kwarg_name, axis in (
        ('time_smart_scale_factor', 't'),
        ('channel_smart_scale_factor', 'c'),
        ('z_smart_scale_factor', 'z'),
        ('y_smart_scale_factor', 'y'),
        ('x_smart_scale_factor', 'x'),
    ):
        val = job.extra.get(kwarg_name)
        if val is not None:
            try:
                auto_factors[axis] = max(1, int(val))
            except (ValueError, TypeError):
                pass

    result = tuple(auto_factors.get(ax, 1) for ax in axes)
    logger.info(f"[smart_downscale] axes={axes}, smart_scale_factor={result}")
    return result


def parse_units(manager: ArrayManager, job: ConversionJob) -> Tuple:
    """Parse per-axis unit overrides from job.extra, falling back to manager.unitdict."""
    return _parse_axis_params(manager, job.extra, 4, manager.unitdict)


def _extract_cropping_slices(kwargs: Dict) -> Dict:
    """Extract cropping range parameters from kwargs.
    
    Collects all cropping-related keyword arguments that define
    rectangular regions of interest in the dataset, converting
    string ranges to tuples suitable for crop() method.
    
    Parameters
    ----------
    kwargs : Dict
        Keyword arguments containing potential cropping parameters
        (e.g., time_range="0,5", channel_range="1,3").
        
    Returns
    -------
    Dict
        Dictionary mapping crop method parameter names to range tuples.
        For example: {"trange": (0, 5), "crange": (1, 3)}
    """
    # Map from CLI parameter names to crop() method parameter names
    param_mapping = {
        'time_range': 'trange',
        'channel_range': 'crange',
        'z_range': 'zrange',
        'y_range': 'yrange',
        'x_range': 'xrange',
    }
    
    crop_kwargs = {}
    for cli_param, crop_param in param_mapping.items():
        if cli_param in kwargs and kwargs[cli_param] is not None:
            range_str = kwargs[cli_param]
            # Parse string like "0,5" to tuple (0, 5)
            try:
                if isinstance(range_str, str):
                    range_parts = range_str.split(',')
                    range_tuple = tuple(int(x.strip()) for x in range_parts)
                else:
                    # Already a tuple/list
                    range_tuple = tuple(range_str)
                crop_kwargs[crop_param] = range_tuple
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse {cli_param}='{range_str}': {e}")
    
    return crop_kwargs


async def _prepare_manager(manager: ArrayManager, job: ConversionJob) -> None:
    """Apply preprocessing steps to a scene manager before writing."""
    manager.fill_default_meta()
    await manager.sync_pyramid(save_changes=False)

    manager._channels = parse_channels(manager, channel_intensity_limits='from_dtype')
    manager.fix_bad_channels()

    cfg = job.conversion
    if cfg.verbose:
        logger.info(f"The manager array shape before squeezing: {manager.array.shape}")
    if cfg.squeeze:
        manager.squeeze()

    cropping_kwargs = _extract_cropping_slices({
        'time_range':    cfg.time_range,
        'channel_range': cfg.channel_range,
        'z_range':       cfg.z_range,
        'y_range':       cfg.y_range,
        'x_range':       cfg.x_range,
    })
    if cropping_kwargs:
        manager.crop(**cropping_kwargs)



def _maybe_compute_bfio_array(
    manager: ArrayManager,
    max_concurrent_scenes: int,
    tile_mb: float,
) -> None:
    """Optimise bfio-tiled dask arrays before writing.

    bfio's BioReader.setId() costs 8-10 s.  The queue-based writer subdivides
    the dask array into small regions and computes each independently.  If a
    region boundary cuts across a bfio tile, the same tile task is re-executed
    multiple times, each paying setId() again.

    Strategy: use *tile_mb* (= bf_tile_size_mb) as the compute-region size.
    The dask array is already chunked at that size, so rechunking to the same
    size is a no-op — every rechunked chunk maps 1-to-1 with one _read_bfio_tile
    call.  The writer then slices the resulting numpy chunks freely with no
    extra BioReader opens.

    If *tile_mb* exceeds the per-scene RAM budget (available/2/concurrent), it
    is capped and the user is warned to lower --bf_tile_size_mb.

    Small arrays (full array ≤ effective tile size):
        Compute the entire array into a single numpy-backed in-memory chunk.
    Large arrays:
        Rechunk to effective tile size; each chunk computed on demand.
    """
    if not manager._bfio_tiling:
        return
    if not isinstance(manager.array, da.Array):
        return

    try:
        available    = psutil.virtual_memory().available
        budget_bytes = available / 2 / max(1, max_concurrent_scenes)
        tile_bytes   = tile_mb * 1024 ** 2
    except Exception as e:
        logger.debug(f"bfio RAM check failed ({e}); keeping tiled dask array")
        return

    if tile_bytes > budget_bytes:
        effective_mb = budget_bytes / (1024 ** 2)
        logger.warning(
            f"bf_tile_size_mb={tile_mb:.0f} exceeds per-scene RAM budget "
            f"({effective_mb:.0f} MB); capping. "
            f"Consider reducing --bf_tile_size_mb."
        )
    else:
        effective_mb = tile_mb

    arr         = manager.array
    array_bytes = arr.nbytes

    if array_bytes <= effective_mb * 1024 ** 2:
        # ── small: compute fully into one in-memory chunk ──────────────────
        logger.info(
            f"bfio: computing {array_bytes / 1e6:.1f} MB into memory "
            f"(tile_mb={effective_mb:.0f}, "
            f"{max_concurrent_scenes} concurrent scenes)"
        )
        computed = manager.array.compute()
        manager.state.update(
            array=da.from_array(computed, chunks=computed.shape),
            axes=manager.axes,
            units=manager.units,
            scales=manager.scales,
        )
    else:
        # ── large: rechunk to tile-aligned boundaries ──────────────────────
        # tile_mb == bf_tile_size_mb → rechunk is a no-op on the existing
        # bfio dask chunks, so each task calls _read_bfio_tile exactly once.
        sd = dict(zip(manager.axes, arr.shape))
        T = sd.get('t', 1); C = sd.get('c', 1); Z = sd.get('z', 1)
        Y = sd.get('y', 1); X = sd.get('x', 1)
        t_t, c_t, z_t, y_t, x_t = _compute_tile_shape(
            T, C, Z, Y, X, arr.dtype, effective_mb)
        chunk = tuple(
            {'t': t_t, 'c': c_t, 'z': z_t, 'y': y_t, 'x': x_t}[ax]
            for ax in manager.axes
        )
        logger.info(
            f"bfio: rechunking to tile-aligned regions "
            f"chunk={chunk} ({effective_mb:.0f} MB per region)"
        )
        manager.state.update(
            array=arr.rechunk(chunk),
            axes=manager.axes,
            units=manager.units,
            scales=manager.scales,
        )


async def _process_single_scene(manager: ArrayManager, output_path: str,
                                job: ConversionJob, sem: asyncio.Semaphore) -> None:
    """Process a single scene/tile from a typed ConversionJob."""
    async with sem:
        conv = job.conversion
        ds   = job.downscale
        clus = job.cluster

        if conv.verbose:
            logger.info(f"The manager array shape before preparation: "
                        f"{manager.array.shape if manager.array is not None else 'N/A'}")
        await _prepare_manager(manager, job)
        _maybe_compute_bfio_array(manager, clus.max_concurrent_scenes, clus.bf_tile_size_mb)

        channel_meta = parse_channels(manager, channel_intensity_limits='from_dtype')

        if conv.verbose:
            logger.info(f"The manager array shape before storing: "
                        f"{manager.array.shape if manager.array is not None else 'N/A'}")

        import dask
        # bf_read_concurrency caps dask's thread pool → caps peak concurrent open
        # BioReader instances (one per thread via thread-local cache).  Without this,
        # dask defaults to cpu_count threads, potentially opening cpu_count readers
        # per series and exhausting JVM heap.
        dask_kw = {}
        if clus.bf_read_concurrency is not None:
            dask_kw['num_workers'] = clus.bf_read_concurrency
        with dask.config.set(**dask_kw):
            await store_multiscale_async(
                arr=manager.array,
                dtype=conv.dtype,
                output_path=output_path,
                zarr_format=conv.zarr_format,
                axes=manager.axes,
                scales=parse_scales(manager, job),
                units=parse_units(manager, job),
                channel_meta=channel_meta,
                auto_chunk=conv.auto_chunk,
                output_chunks=parse_chunks(manager, job),
                output_shard_coefficients=parse_shard_coefs(manager, job),
                overwrite=conv.overwrite,
                n_layers=ds.n_layers,
                min_dimension_size=ds.min_dimension_size,
                scale_factors=parse_scale_factors(manager, job),
                smart_scale_factor=parse_smart_scale_factors(manager, job),
                max_concurrency=clus.max_concurrency,
                region_size_mb=clus.region_size_mb,
                compute_batch_size=job.extra.get('compute_batch_size', 4),
                queue_size=clus.queue_size,
                memory_limit_per_batch=job.extra.get('memory_limit_per_batch', 1024),
                verbose=conv.verbose,
                compressor=conv.compressor,
                compressor_params=conv.compressor_params,
                downscale_method=ds.downscale_method,
            )

        update_channels = conv.channel_intensity_limits == 'from_array'

        if conv.save_omexml or update_channels:
            out_mgr = ArrayManager(output_path, skip_dask=conv.skip_dask)
            await out_mgr.init()

            if update_channels:
                out_mgr.fill_default_meta()
                await out_mgr.sync_pyramid(save_changes=False)
                if conv.squeeze:
                    out_mgr.squeeze()
                channels = parse_channels(
                    out_mgr,
                    channel_intensity_limits=conv.channel_intensity_limits,
                    dtype=conv.dtype,
                    **{k: v for k, v in job.extra.items()
                       if k in ('channel_labels', 'channel_colors')},
                )
                assert out_mgr.pyr is not None
                meta = out_mgr.pyr.meta
                meta.metadata['omero']['channels'] = channels
                if meta.zarr_group is not None:
                    if 'ome' not in meta.zarr_group.attrs:
                        meta.zarr_group.attrs.update({'omero': []})
                meta._pending_changes = True
                meta.save_changes()

            if conv.save_omexml:
                await out_mgr.save_omexml(output_path, overwrite=True)


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


async def _load_input_manager(job: ConversionJob) -> ArrayManager:
    """Open the input file, load all requested scenes/tiles/views/illuminations."""
    manager = ArrayManager(
        job.input_path,
        metadata_reader=job.conversion.metadata_reader,
        skip_dask=job.conversion.skip_dask,
        reader_tile_size_mb=job.cluster.bf_tile_size_mb,
        force_bioformats=job.readers.force_bioformats,
        as_mosaic=job.readers.as_mosaic,
    )

    # scene_index may be 'all', an int, or a comma-separated string from CSV
    series = job.readers.scene_index
    if isinstance(series, str) and series != 'all' and ',' in series:
        series = [int(x.strip()) for x in series.split(',')]
    elif isinstance(series, str) and series != 'all':
        try:
            series = int(series)
        except ValueError:
            pass

    await manager.load_scenes(scene_indices=series)
    assert manager.loaded_scenes is not None

    if job.conversion.verbose:
        first = next(iter(manager.loaded_scenes.values()), None)
        if first is not None:
            logger.info(f"The manager array is of type: {type(first.array)}")

    mosaic_tile_index = job.readers.mosaic_tile_index
    if isinstance(mosaic_tile_index, str) and mosaic_tile_index != 'all' and ',' in mosaic_tile_index:
        mosaic_tile_index = [int(x.strip()) for x in mosaic_tile_index.split(',')]
    elif isinstance(mosaic_tile_index, str) and mosaic_tile_index != 'all':
        try:
            mosaic_tile_index = int(mosaic_tile_index)
        except ValueError:
            mosaic_tile_index = None

    if mosaic_tile_index is not None and len(manager.loaded_scenes) > 1:
        logger.warning(
            "Currently cannot load multiple scenes and multiple tiles at the same time. "
            "Will load only the first tile.")
    elif mosaic_tile_index is not None:
        await manager.load_tiles(tile_indices=mosaic_tile_index)

    # ── Views / illuminations ────────────────────────────────────────────
    # Resolve view and illumination indices (same multi-value rules as scene_index).
    def _parse_multi(value):
        if value == 'all' or isinstance(value, list):
            return value
        if isinstance(value, str) and ',' in value:
            return [int(x.strip()) for x in value.split(',')]
        try:
            return int(value)
        except (TypeError, ValueError):
            return value

    view_index  = _parse_multi(job.readers.view_index)
    illu_index  = _parse_multi(job.readers.illumination_index)

    # Determine whether multi-view/illumination iteration is needed.
    # A single int value of 0 with no concat flags means the default single-index
    # behaviour — skip the extra loading step entirely for performance.
    need_views = (
        view_index != 0
        or job.readers.concat_views
        or view_index == 'all'
        or isinstance(view_index, list)
    )
    need_illus = (
        illu_index != 0
        or job.readers.concat_illuminations
        or illu_index == 'all'
        or isinstance(illu_index, list)
    )

    if need_views or need_illus:
        # Default to index 0 for the non-iterated dimension when not specified.
        if not need_views:
            view_index = 0
        if not need_illus:
            illu_index = 0
        await manager.load_views_illuminations(
            view_indices=view_index,
            illumination_indices=illu_index,
            concat_views=job.readers.concat_views,
            concat_illuminations=job.readers.concat_illuminations,
        )

    return manager


async def unary_worker(job: ConversionJob) -> None:
    """Convert a single input file to OME-Zarr (all scenes/tiles/views/illuminations in parallel)."""
    manager = await _load_input_manager(job)

    # max_concurrent_scenes: how many scenes write simultaneously.
    # Keeping this ≤ 1 prevents all scenes calling zarr's sync() at once,
    # which would cause spurious ContainsArrayError on fresh paths.
    sem = asyncio.Semaphore(job.cluster.max_concurrent_scenes)
    add_scene = manager._n_scenes > 1

    tasks = []
    if manager.loaded_scenes is None:
        raise ValueError("At least one scene must be available.")

    # ── View / illumination outputs take priority when present ───────────
    if manager.loaded_views_illuminations is not None:
        for man in manager.loaded_views_illuminations.values():
            out_path = _generate_output_path(
                job.output_path, man.series_path,
                man.series if add_scene else None,
            )
            tasks.append(asyncio.create_task(
                _process_single_scene(man, out_path, job, sem)
            ))
    elif manager.loaded_tiles is None:
        for man in manager.loaded_scenes.values():
            out_path = _generate_output_path(
                job.output_path, man.series_path,
                man.series if add_scene else None,
            )
            tasks.append(asyncio.create_task(
                _process_single_scene(man, out_path, job, sem)
            ))
    elif len(manager.loaded_scenes) == 1 and manager.loaded_tiles is not None:
        n_tiles  = manager._n_tiles or 1
        add_tile = n_tiles > 1
        for tile in manager.loaded_tiles.values():
            out_path = _generate_output_path(
                job.output_path, tile.series_path,
                tile.mosaic_tile_index if add_tile else None,
            )
            tasks.append(asyncio.create_task(
                _process_single_scene(tile, out_path, job, sem)
            ))
    else:
        raise Exception(
            "Having both multiple scenes and multiple tiles is not currently supported.")

    await asyncio.gather(*tasks)


async def aggregative_worker(manager: ArrayManager,
                             output_path: str,
                             **kwargs) -> None:
    """Convert aggregated (pre-concatenated) image data to OME-Zarr."""
    series = kwargs.get('series', 0)
    if not isinstance(series, int):
        raise TypeError(
            "Aggregative conversion does not support multiple series. "
            "Please specify an integer as a single series index."
        )

    output_path_full = f"{output_path}.zarr"
    if kwargs.get('verbose', False):
        logger.info(f"The manager array is of type: {type(manager.array)}")

    # Build a typed ConversionJob so _process_single_scene can access typed fields.
    job = ConversionJob.from_kwargs(
        input_path=manager.path or '',
        output_path=output_path_full,
        kwargs=kwargs,
    )
    sem = asyncio.Semaphore(job.cluster.max_concurrency)
    await _process_single_scene(manager, output_path_full, job, sem)


@safe_worker_wrapper
def unary_worker_sync(job: ConversionJob) -> dict:
    """Synchronous entry point for multiprocessing workers.

    Dask distributed workers run with an active event loop, so asyncio.run()
    cannot be called directly.  We escape by running the coroutine in a
    dedicated thread that has no event loop of its own.
    """
    import concurrent.futures

    if job.conversion.verbose:
        logger.info(f"[Worker] Processing: {job.input_path}")

    def _run_in_fresh_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(unary_worker(job))
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(_run_in_fresh_thread).result()

    if job.conversion.verbose:
        logger.info(f"[Worker] Completed: {job.input_path}")

    return {"status": "success", "input": str(job.input_path), "output": job.output_path}


@safe_worker_wrapper
def aggregative_worker_sync(manager: ArrayManager,
                            output_path: str,
                            kwargs: dict) -> dict:
    """
    Synchronous wrapper for aggregative_worker.
    Safe for multiprocessing with proper exception handling.
    """
    import concurrent.futures

    if kwargs.get('verbose', False):
        logger.info(f"[Worker] Processing aggregative: {output_path}")

    def _run_in_fresh_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                aggregative_worker(manager, output_path, **kwargs)
            )
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(_run_in_fresh_thread).result()

    if kwargs.get('verbose', False):
        logger.info(f"[Worker] Completed aggregative: {output_path}")
    return {"status": "success", "output": output_path}


async def _aggregative_group_pipeline(
    file_paths: list,
    output_path: str,
    job_kwargs: dict,
) -> None:
    """Async pipeline for one pre-grouped file set: read → digest → write zarr.

    Called from ``aggregative_worker_from_paths`` inside a fresh event loop.
    ``output_path`` is the explicit planned path — it overrides whatever name
    ``digest()`` would derive internally.
    """
    from eubi_bridge.conversion.aggregative_conversion_base import AggregativeConverter

    scene_index            = job_kwargs.get('scene_index', 0)
    override_channel_names = job_kwargs.get('override_channel_names', False)
    metadata_reader        = job_kwargs.get('metadata_reader', 'bfio')
    common_dir             = os.path.commonpath(file_paths)

    base = AggregativeConverter(
        series=scene_index,
        override_channel_names=override_channel_names,
    )
    base.filepaths = file_paths

    await base.read_dataset(
        readers_params={'aszarr': job_kwargs.get('skip_dask', False)}
    )
    await base.digest(
        time_tag=job_kwargs.get('time_tag'),
        channel_tag=job_kwargs.get('channel_tag'),
        z_tag=job_kwargs.get('z_tag'),
        y_tag=job_kwargs.get('y_tag'),
        x_tag=job_kwargs.get('x_tag'),
        axes_of_concatenation=job_kwargs.get('concatenation_axes'),
        metadata_reader=metadata_reader,
        output_path=common_dir,
    )

    if not base.managers:
        raise RuntimeError(f"digest() produced no output groups for: {file_paths}")

    # Pre-grouped files → exactly one output manager.  Use the planned path.
    manager = next(iter(base.managers.values()))
    await aggregative_worker(manager, output_path, **job_kwargs)


@safe_worker_wrapper
def aggregative_worker_from_paths(
    file_paths: list,
    output_path: str,
    job_kwargs: dict,
) -> dict:
    """Full aggregative pipeline for one pre-grouped file set.

    Runs entirely within this process so only picklable data crosses the
    process boundary, enabling true CPU-core parallelism via ProcessPool.
    Each subprocess starts its own JVM instance — no shared reader state.
    """
    import concurrent.futures
    import dask

    soft_start_jvm()

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with dask.config.set(scheduler='synchronous'):
                return loop.run_until_complete(
                    _aggregative_group_pipeline(file_paths, output_path, job_kwargs)
                )
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(_run).result()

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