"""
Conversion workers: the entry points a worker process runs for one input file
(``unary_worker_sync``), one joined output (``aggregative_worker_from_paths``) or one
file's metadata (``metadata_reader_sync``), and the per-job helpers the conversion
paths share (``pyramid_path``, ``aggregative_path``).

eubi-bridge 0.1.x converted through ``ArrayManager`` and its own writer here; that route
is retired (kept for reference in ``legacy_0_1/``).
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from eubi_bridge.conversion.worker_init import safe_worker_wrapper
from eubi_bridge.core.config_models import ChunkConfig, ConversionJob, parse_scale
from eubi_bridge.core.output_store import store_exists
from eubi_bridge.utils.jvm_manager import soft_start_jvm
from eubi_bridge.utils.logging_config import get_logger

logger = get_logger(__name__)

AXIS_PARAM_MAP = {
    't': ('time_chunk', 'time_shard_coef', 'time_scale', 'time_scale_factor', 'time_unit'),
    # No scale or unit for channels: a channel has no physical extent.
    'c': ('channel_chunk', 'channel_shard_coef', None, 'channel_scale_factor', None),
    'z': ('z_chunk', 'z_shard_coef', 'z_scale', 'z_scale_factor', 'z_unit'),
    'y': ('y_chunk', 'y_shard_coef', 'y_scale', 'y_scale_factor', 'y_unit'),
    'x': ('x_chunk', 'x_shard_coef', 'x_scale', 'x_scale_factor', 'x_unit'),
}

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


def _parse_axis_params(manager, kwargs: Dict,
                       param_idx: int, default_dict: Dict) -> Tuple:
    """
    Generic parser for axis-based parameters.

    Args:
        manager: the image (``axes``, ``scaledict``, ``unitdict``)
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
        if param_name is None:
            # No user-settable parameter for this axis (a channel has no
            # physical scale or unit).  Its scale still belongs in the result,
            # carrying whatever the file itself says -- dropping it would leave
            # the scales tuple shorter than the axes it describes.  Its unit
            # does not: the NGFF writer expects the channel left out of the
            # units and the default for it ('Channel') is no valid NGFF unit,
            # which viewers such as Neuroglancer refuse to open.
            if param_idx == 2:
                output[axis] = default_dict.get(axis)
            continue

        output[axis] = _get_param_value(kwargs, param_name, axis, default_dict)

    return tuple(output[ax] for ax in axes if ax in output)


def parse_chunks(manager, job: ConversionJob) -> Tuple:
    """Parse chunk sizes for each axis from the job's ConversionConfig."""
    cfg = job.conversion
    return ChunkConfig(
        t=cfg.time_chunk, c=cfg.channel_chunk,
        z=cfg.z_chunk, y=cfg.y_chunk, x=cfg.x_chunk,
    ).as_tuple(manager.axes)


def parse_shard_coefs(manager, job: ConversionJob) -> Tuple:
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


def _axis_metadata_kwargs(job: ConversionJob) -> Dict:
    """Scale/unit overrides from the metadata config, overlaid with job.extra.

    These are typed fields on MetadataConfig now, so they no longer arrive as
    unrecognised keys in ``extra``.  ``extra`` is still consulted and still
    wins, because that is where a per-row override from a conversion table
    arrives -- the config value is only the batch-wide default.
    """
    stored = {k: v for k, v in job.metadata.model_dump().items()
              if (k.endswith(('_scale', '_unit')) and v is not None)}
    # extra skips the config model, so its scales get the same one-notation check
    stored.update({k: parse_scale(v, k) if k.endswith('_scale') else v
                   for k, v in job.extra.items()})
    return stored


def parse_scales(manager, job: ConversionJob) -> Tuple:
    """Parse per-axis scale overrides, falling back to manager.scaledict."""
    return _parse_axis_params(
        manager, _axis_metadata_kwargs(job), 2, manager.scaledict)


def parse_scale_factors(manager, job: ConversionJob) -> Tuple:
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


def parse_smart_scale_factors(manager, job: ConversionJob) -> Optional[Tuple]:
    """Compute isotropic scale factors when ``apply_smart_downscaling`` is set.

    Smart-scaling parameters live on ``job.downscale`` (the DownscaleConfig); the
    legacy ``job.extra`` location is still honoured as a fallback.
    """
    ds = job.downscale

    def _smart(name):
        val = getattr(ds, name, None)
        return val if val is not None else job.extra.get(name)

    if not (getattr(ds, 'apply_smart_downscaling', False)
            or job.extra.get('apply_smart_downscaling', False)):
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
        val = _smart(kwarg_name)
        if val is not None:
            try:
                auto_factors[axis] = max(1, int(val))
            except (ValueError, TypeError):
                pass

    result = tuple(auto_factors.get(ax, 1) for ax in axes)
    logger.info(f"[smart_downscale] axes={axes}, smart_scale_factor={result}")
    return result


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


def _threads_per_scene(clus) -> int:
    """A conversion's ``max_threads`` (``"auto"``: core/memory.write_threads), shared
    by the scenes it converts at once (``max_concurrent_scenes``): raising the scenes
    never raises the threads."""
    from eubi_bridge.core import memory
    scenes = max(1, int(clus.max_concurrent_scenes or 1))
    threads = memory.write_threads(clus.max_threads or "auto", _decode_sharers(clus))
    return max(1, threads // scenes)


def _unit_mb(arr) -> Optional[float]:
    """The size of the source's native decode unit in MB (a whole compressed plane,
    tile or strip), or None when it is not known."""
    try:
        from dyna_zarr._read_cost import read_unit
        unit = read_unit(arr)
        if unit is None:
            return None
        return float(np.prod(unit)) * np.dtype(arr.dtype).itemsize / 2 ** 20
    except Exception:                                        # noqa: BLE001
        return None


def _memory_plan_for(array, job: ConversionJob) -> tuple:
    """``(scene write budget, region size)`` in MB for the source `array` of one output,
    from the run's memory budget (core/memory.py): this process's share, minus what it
    already holds and the reader's share (applied to micro-reader here), over the scenes
    at once."""
    from eubi_bridge.core import memory
    clus = job.cluster
    process = clus.process_memory_mb or memory.process_mb(clus)
    micro = job.readers.pixel_reader == 'micro' or job.metadata.metadata_reader == 'micro'
    threads = None
    if micro:
        unit = _unit_mb(array)
        threads = memory.decode_threads(clus.decode_threads, _decode_sharers(clus), process,
                                        unit)
        decode, cache = memory.reader_share(process, clus.decode_memory, threads, unit)
        from eubi_bridge.core.micro_source import set_read_concurrency, set_read_memory
        set_read_concurrency(threads)
        set_read_memory(decode, cache)
        reader = decode + cache
    else:                                                    # Bio-Formats: the JVM heap
        reader = memory.parse_mb(clus.jvm_memory or '1g')
    held = memory.baseline_mb()
    scenes = max(1, int(clus.max_concurrent_scenes or 1))
    scene = memory.scene_mb(process, held, reader, scenes)
    region = memory.region_mb(clus.region_size_mb, scene, _threads_per_scene(clus))
    reading = f"{reader:.0f} MB reading" + (f" ({threads} decode threads)" if threads else "")
    line = (f"Memory: {process:.0f} MB for this conversion = {held:.0f} MB already held + "
            f"{reading} + {scene:.0f} MB writing per scene ({scenes} at once); "
            f"regions of {region:.0f} MB")
    # once per process for each plan: a plate asks for one per field
    (logger.debug if line in _memory_logged else logger.info)(line)
    _memory_logged.add(line)
    return scene, region


_memory_logged: set = set()


def _decode_sharers(clus) -> int:
    """The conversions whose decoding shares this machine's cores: those running at once
    in worker processes (each has its own decode pool); 1 when they run as threads of one
    process (one pool for all of them)."""
    if getattr(clus, 'use_threading', False):
        return 1
    return max(1, int(getattr(clus, 'concurrent_jobs', None) or clus.max_processes or 1))


#: settings of eubi-bridge's own writer (retired in 0.2.0) that do nothing now
_EUBI_WRITER_ONLY = ('queue_size', 'max_concurrent_downscale_layers',
                     'tensorstore_data_copy_concurrency')
_warned_unused: set = set()


def _warn_unused_writer_settings(clus) -> None:
    """Warn once for each setting of eubi-bridge's retired writer set to a non-default
    value (it does nothing now; never fails: retired-settings rule)."""
    for name in _EUBI_WRITER_ONLY:
        default = type(clus).model_fields[name].default
        if getattr(clus, name) != default and name not in _warned_unused:
            _warned_unused.add(name)
            logger.warning(
                f"{name}={getattr(clus, name)} does nothing: it was a setting of "
                f"eubi-bridge's own writer, retired in 0.2.0. ome_zarr_pyramid writes "
                f"every output; its parallelism is max_threads, its memory memory_budget.")


async def _convert_safely(conversion, output_path: str) -> None:
    """Await one output's `conversion`; on failure remove any partial output it
    created and re-raise."""
    target = output_path if output_path.endswith('.zarr') else f"{output_path}.zarr"
    # Whether the output already existed BEFORE this run.  A pre-existing
    # dataset is never ours to delete — only partial output created by this run
    # may be cleaned up on failure.
    # Remote-aware: os.path.exists is always False for a URL, which
    # would make a pre-existing remote store look like our own partial
    # output and hand it to the cleanup below.
    preexisting = store_exists(target)
    try:
        await conversion
    except FileExistsError:
        # Name collision with overwrite=False: surface the error and leave the
        # existing dataset completely untouched.
        raise
    except Exception:
        import shutil
        try:
            if not preexisting and os.path.isdir(target):
                shutil.rmtree(target, ignore_errors=True)
        except Exception:
            pass
        raise


def _generate_output_path(base_path: str, series_path: str,
                          series_idx: Optional[int] = None,
                          tile_idx: Optional[int] = None,
                          resolved_basename: Optional[str] = None) -> str:
    """
    Generate output path with optional scene/tile suffixes.

    Args:
        base_path: Base output directory
        series_path: Source series path
        series_idx: Optional scene index
        tile_idx: Optional tile index
        resolved_basename: Pre-computed basename replacing the one derived from
            *series_path*.  Set when two inputs share a filename and would
            otherwise write to the same store.  Scene/tile suffixes still apply
            on top of it, so multi-scene naming is unaffected.

    Returns:
        Complete output path
    """
    from eubi_bridge.utils.path_utils import output_stem
    basename = resolved_basename or output_stem(series_path)
    suffix = ""

    if series_idx is not None:
        suffix += f"_scene{series_idx}"
    if tile_idx is not None:
        suffix += f"_tile{tile_idx}"

    return f"{base_path}/{basename}{suffix}.zarr"


async def unary_worker(job: ConversionJob) -> None:
    """Convert a single input file to OME-Zarr (all its outputs; conversion/pyramid_path.py)."""
    from eubi_bridge.conversion import pyramid_path
    await pyramid_path.convert(job)


def _report_failures(results: list, job: ConversionJob, total_in_file: int = 0) -> None:
    """Log the outputs that failed; raise when all did."""
    failures = [r for r in results if isinstance(r, BaseException)]
    if failures:
        from eubi_bridge.conversion.pyramid_path import ReaderCapabilityError
        gaps = [exc for exc in failures if isinstance(exc, ReaderCapabilityError)]
        if gaps:
            # the install cannot decode the file: one error, the outputs after the
            # first were not tried (nothing to say per scene)
            raise gaps[0]
        for exc in failures:
            logger.warning(f"Skipped a scene that could not be converted: {exc}")
        if len(failures) == len(results):
            raise failures[0]
        written = len(results) - len(failures)
        total_in_file = total_in_file or len(results)
        unreadable = max(0, total_in_file - written)
        reasons = "; ".join(sorted({str(exc) for exc in failures}))
        logger.warning(
            "\n" + "=" * 78 + "\n"
            f"PARTIAL CONVERSION: {job.input_path}\n"
            f"Converted {written} of {total_in_file} scene(s) that the file "
            f"contains. {unreadable} scene(s) could NOT be converted and are "
            f"MISSING from the output.\n"
            f"Reason(s): {reasons}\n"
            f"The file itself may still be fine; if the reason names damaged or "
            f"corrupt data, re-export or repair the source file.\n" + "=" * 78
        )


@safe_worker_wrapper
def unary_worker_sync(job: ConversionJob) -> dict:
    """Synchronous entry point for multiprocessing workers.

    Dask distributed workers run with an active event loop, so asyncio.run()
    cannot be called directly.  We escape by running the coroutine in a
    dedicated thread that has no event loop of its own.
    """
    import concurrent.futures
    from eubi_bridge.core.memory import baseline_mb
    baseline_mb()            # what this process holds before it reads anything

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
    from eubi_bridge.core.memory import baseline_mb
    baseline_mb()            # what this process holds before it reads anything

    soft_start_jvm()

    def _run():
        from eubi_bridge.conversion import aggregative_path
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                aggregative_path.convert_group(file_paths, output_path, job_kwargs))
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(_run).result()

    return {"status": "success", "output": output_path}


@safe_worker_wrapper
def metadata_reader_sync(input_path: str, kwargs: dict) -> dict:
    """One file's metadata as a conversion writes it (conversion/metadata_path.py):
    ``{"status": "success", "input_path", "series", "axes", "shape", "scale", "units",
    "dtype", "channels", "ndim"}``, a list of them for several scenes, or
    ``{"status": "error", "input_path", "error"}``."""
    if kwargs.get('verbose', False):
        logger.info(f"[MetadataReader] Reading metadata: {input_path}")
    from eubi_bridge.conversion import metadata_path
    return metadata_path.read_metadata(input_path, kwargs)
