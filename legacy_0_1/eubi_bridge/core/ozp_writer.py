"""Write a conversion with ome_zarr_pyramid instead of eubi-bridge's own writer.

Selected with ``writer='ozp'``, or by the default ``writer='auto'`` wherever it applies
(see :func:`resolve_writer`). Same inputs as ``writers.store_multiscale_async`` /
``store_existing_pyramid_async``, and the same store as output (the write path is
dyna -> ome_zarr_pyramid -> eubi-bridge).

What eubi-bridge decides stays here, so both writers get identical inputs: the chunk shape
(``autocompute_chunk_shape``), the shard coefficients, the channel metadata and the
``eubi_bridge`` attrs block. Everything written - arrays, levels, NGFF metadata - is
ome_zarr_pyramid's (via dyna-zarr).
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from eubi_bridge.utils.array_utils import autocompute_chunk_shape
from eubi_bridge.utils.logging_config import get_logger

logger = get_logger(__name__)


def _require_ozp():
    """ome_zarr_pyramid, or a clear error if the installation lacks it."""
    try:
        import ome_zarr_pyramid
    except ImportError as exc:
        raise ImportError(
            "writer='ozp' needs ome_zarr_pyramid, a dependency of eubi-bridge that "
            "is missing here: pip install ome_zarr_pyramid (or use writer='eubi').") from exc
    return ome_zarr_pyramid


def _is_dask(arr) -> bool:
    """A dask array (subclasses included), without importing dask."""
    return type(arr).__module__.split(".")[0] == "dask" or any(
        c.__module__.split(".")[0] == "dask" for c in type(arr).__mro__)


def _memory_budget(arrays, kwargs) -> Optional[float]:
    """The memory budget for this write, or None where it does not apply.

    dyna-zarr bounds a write by its budget in its own region loop. A dask array is
    written by dask's own scheduler instead (its memory is max_threads x the output's
    write unit), and dyna refuses a budget for it rather than ignore it. eubi-bridge sets
    the budget for every conversion, so for a dask source (the standard readers) it is
    left out here: such a write behaves as with eubi-bridge 0.1.3's writer. eubi-bridge
    0.1.4 passed it anyway, so every conversion read through dask failed."""
    budget = kwargs.get('memory_budget_mb')
    if budget is not None and any(_is_dask(a) for a in arrays):
        logger.info("memory_budget does not apply to a dask-read source: dask's scheduler "
                    "writes it (memory about max_threads x the output's write unit)")
        return None
    return budget


class _SerialisedReads:
    """A resource-backed dask array read one region at a time. bioio-nd2's array (and
    a Bio-Formats-backed dask array) reads through one shared handle that is not
    thread-safe: read from several threads at once it can return wrong pixels without
    an error, or crash. eubi-bridge's own writer locks such reads (writers._read_region);
    the ozp writer hands dask arrays to dask's scheduler, so the lock is put here."""

    def __init__(self, array):
        self._array = array
        self.shape, self.dtype, self.ndim = tuple(array.shape), array.dtype, array.ndim

    def __getitem__(self, key):
        return np.asarray(self._array[key].compute(scheduler="synchronous"))


def _read_safely(arr):
    """`arr` as it is, or, when it is resource-backed, a dask array over the same chunks
    whose reads run one at a time (dask's per-array read lock); writes stay parallel."""
    from eubi_bridge.core.writers import _is_resource_backed
    if not _is_resource_backed(arr):
        return arr
    import dask.array as da
    return da.from_array(_SerialisedReads(arr), chunks=arr.chunks, lock=True,
                         meta=np.empty((0,) * arr.ndim, dtype=arr.dtype))


def _level_keys(pyr):
    from natsort import natsorted
    return natsorted(pyr.layers.keys())


def _numbered_levels(pyr) -> bool:
    """ozp writes resolution levels as 0..n-1; eubi's writer keeps any source names."""
    keys = _level_keys(pyr)
    return keys == [str(i) for i in range(len(keys))]


def resolve_writer(requested: str, existing_pyr=None) -> str:
    """The writer a conversion uses: 'eubi' or 'ozp'. ``'auto'`` is 'ozp', except where
    it cannot write the output yet - kept resolution levels (``existing_pyr``) not named
    0..n-1 - which go to eubi-bridge's writer. An explicit choice is used as given (and
    'ozp' refuses that case loudly in store_existing_pyramid_ozp)."""
    if requested != 'auto':
        return requested
    if existing_pyr is not None and not _numbered_levels(existing_pyr):
        logger.info(f"writer='auto': eubi-bridge's writer for this output: it keeps the "
                    f"source's resolution levels {_level_keys(existing_pyr)}, and ozp "
                    f"writes levels as 0..n-1.")
        return 'eubi'
    return 'ozp'


def _ozp_target_args(output_path):
    """``(path, storage_options)`` for ome_zarr_pyramid.

    eubi-bridge addresses object storage as ``https://<endpoint>/<bucket>/<key>`` and
    reads/writes it anonymously (s3fs ``anon=True``). ome_zarr_pyramid's dyna engine
    takes ``s3://<bucket>/<key>`` plus obstore options: the same endpoint, unsigned
    requests (anonymous) and path-style addressing (bucket in the path, as in the URL).
    Local paths pass through.
    """
    text = str(output_path)
    if text.startswith("https://"):
        host, _, key = text[len("https://"):].partition("/")
        return (f"s3://{key}", {"endpoint": f"https://{host}", "skip_signature": True,
                                "virtual_hosted_style_request": False})
    return text, None


def _ozp_version(zarr_format: int, ome_zarr_version: Optional[str]) -> str:
    """The OME-Zarr version, as store_multiscale_async decides it."""
    return ome_zarr_version or ('0.5' if zarr_format == 3 else '0.4')


def _chunks(arr, axes, auto_chunk, output_chunks, dtype, target_chunk_mb):
    """The level-0 chunk shape, exactly as store_multiscale_async computes it."""
    if auto_chunk or output_chunks is None:
        chunks = autocompute_chunk_shape(arr.shape, axes=axes,
                                         target_chunk_mb=target_chunk_mb, dtype=dtype)
    else:
        chunks = output_chunks
    return tuple(int(c) for c in np.minimum(chunks, arr.shape).tolist())


def _eubi_attrs(acquisition_metadata: Optional[dict]) -> dict:
    """The namespaced ``eubi_bridge`` attrs block (version stamp + acquisition)."""
    from eubi_bridge.ngff.multiscales import _eubi_bridge_stamp
    block = dict(_eubi_bridge_stamp())
    if acquisition_metadata:
        block['acquisition'] = acquisition_metadata
    return block


def _decorate(pyr, *, translation, channel_meta, acquisition_metadata):
    """Translation (every level), omero channels and the eubi_bridge block."""
    meta = pyr.meta
    if translation is not None:
        for path in meta.resolution_paths:
            meta.update_translation(path, translation)
    if isinstance(channel_meta, list):
        meta.metadata.setdefault('omero', {})['channels'] = channel_meta
    meta.metadata['eubi_bridge'] = _eubi_attrs(acquisition_metadata)
    meta._pending_changes = True
    return pyr


async def store_existing_pyramid_ozp(
    pyr,
    output_path: str,
    axes: Sequence[str],
    units: Sequence[str],
    channel_meta: Any = None,
    zarr_format: int = 2,
    auto_chunk: bool = True,
    output_chunks=None,
    output_shard_coefficients=None,
    overwrite: bool = False,
    translation: Optional[Sequence[float]] = None,
    *,
    max_threads: Optional[int] = None,
    region_size_mb: float = 8.0,
    **kwargs,
) -> None:
    """``store_existing_pyramid_async``, written by ome_zarr_pyramid: the source's own
    resolution levels, as they are (no downscaling), each with its own chunk shape."""
    import asyncio
    await asyncio.to_thread(
        _store_existing_sync, pyr, output_path, axes, units, channel_meta, zarr_format,
        auto_chunk, output_chunks, output_shard_coefficients, overwrite, translation,
        max_threads, region_size_mb, kwargs)


def _store_existing_sync(src, output_path, axes, units, channel_meta, zarr_format,
                         auto_chunk, output_chunks, output_shard_coefficients, overwrite,
                         translation, max_threads, region_size_mb, kwargs):
    _require_ozp()
    from ome_zarr_pyramid import IO, Pyramid

    keys = _level_keys(src)
    if not _numbered_levels(src):
        raise NotImplementedError(
            f"writer='ozp' writes resolution levels as 0..n-1; this source names them "
            f"{keys}. Use writer='auto' or 'eubi' for it.")
    arrays = [_read_safely(src.layers[k]) for k in keys]
    dtype = np.dtype(kwargs.get('dtype') or arrays[0].dtype)
    version = _ozp_version(zarr_format, kwargs.get('ome_zarr_version'))
    axes = ''.join(axes)
    target_mb = kwargs.get('target_chunk_mb', 1)
    chunks = {i: _chunks(a, axes, auto_chunk, output_chunks, dtype, target_mb)
              for i, a in enumerate(arrays)}

    pyr = Pyramid().from_arrays(arrays, axis_order=axes, unit_list=list(units),
                                scales=[list(src.meta.get_scale(k)) for k in keys],
                                version=version, name="Series 0")
    _decorate(pyr, translation=translation, channel_meta=channel_meta,
              acquisition_metadata=None)
    logger.info(f"writer='ozp': {output_path} (the source's {len(keys)} levels)")
    target, storage_options = _ozp_target_args(output_path)
    IO().write_pyramid(
        pyr, target, storage_options=storage_options, overwrite=overwrite,
        max_threads=int(max_threads or 4), region_size_mb=float(region_size_mb),
        chunk_shape=chunks,
        shard_coefficients=(tuple(output_shard_coefficients)
                            if output_shard_coefficients is not None and zarr_format == 3
                            else None),
        compressor=kwargs.get('compressor', 'blosc'),
        compressor_params=kwargs.get('compressor_params') or {},
        dtype=dtype if dtype != np.dtype(arrays[0].dtype) else None,
        window_from_data=bool(kwargs.get('window_from_data', False)),
        windows=kwargs.get('windows', 'keep'),
        io_backend=kwargs.get('io_backend', 'tensorstore'),
        memory_budget_mb=_memory_budget(arrays, kwargs),
    )


async def store_multiscale_ozp(
    arr,
    output_path: str,
    axes: Sequence[str],
    scales: Sequence[float],
    units: Sequence[str],
    translation: Optional[Sequence[float]] = None,
    acquisition_metadata: Optional[dict] = None,
    zarr_format: int = 2,
    auto_chunk: bool = True,
    output_chunks=None,
    output_shard_coefficients=None,
    overwrite: bool = False,
    channel_meta: Any = None,
    *,
    scale_factors=None,
    n_layers=None,
    min_dimension_size=None,
    downscale_method: str = 'simple',
    smart_scale_factor=None,
    max_threads: Optional[int] = None,
    region_size_mb: float = 8.0,
    **kwargs,
) -> None:
    """``store_multiscale_async``, written by ome_zarr_pyramid.

    ome_zarr_pyramid's API is synchronous and runs its own event loops, so it runs on a
    worker thread: eubi-bridge calls this from inside a running loop."""
    import asyncio
    await asyncio.to_thread(
        _store_multiscale_sync, arr, output_path, axes, scales, units, translation,
        acquisition_metadata, zarr_format, auto_chunk, output_chunks,
        output_shard_coefficients, overwrite, channel_meta, scale_factors, n_layers,
        min_dimension_size, downscale_method, smart_scale_factor, max_threads,
        region_size_mb, kwargs)


def _store_multiscale_sync(arr, output_path, axes, scales, units, translation,
                           acquisition_metadata, zarr_format, auto_chunk, output_chunks,
                           output_shard_coefficients, overwrite, channel_meta,
                           scale_factors, n_layers, min_dimension_size, downscale_method,
                           smart_scale_factor, max_threads, region_size_mb, kwargs):
    _require_ozp()
    from ome_zarr_pyramid import IO, Pyramid

    arr = _read_safely(arr)
    dtype = kwargs.get('dtype') or arr.dtype
    dtype = np.dtype(dtype)
    version = _ozp_version(zarr_format, kwargs.get('ome_zarr_version'))
    axes = ''.join(axes)
    chunks = _chunks(arr, axes, auto_chunk, output_chunks, dtype,
                     kwargs.get('target_chunk_mb', 1))

    pyr = Pyramid().from_arrays([arr], axis_order=axes, unit_list=list(units),
                                scales=[list(scales)], version=version, name="Series 0")
    if scale_factors is not None:
        pyr = pyr.downscale(scale_factor=tuple(scale_factors), n_layers=n_layers,
                            min_dimension_size=min_dimension_size,
                            smart_scale_factor=smart_scale_factor,
                            downscale_method=downscale_method)
    _decorate(pyr, translation=translation, channel_meta=channel_meta,
              acquisition_metadata=acquisition_metadata)
    logger.info(f"writer='ozp': {output_path} ({pyr.nlayers} levels, chunks {chunks})")
    target, storage_options = _ozp_target_args(output_path)
    IO().write_pyramid(
        pyr, target, storage_options=storage_options, overwrite=overwrite,
        max_threads=int(max_threads or 4), region_size_mb=float(region_size_mb),
        chunk_shape=chunks,
        shard_coefficients=(tuple(output_shard_coefficients)
                            if output_shard_coefficients is not None and zarr_format == 3
                            else None),
        compressor=kwargs.get('compressor', 'blosc'),
        compressor_params=kwargs.get('compressor_params') or {},
        dtype=dtype if dtype != np.dtype(arr.dtype) else None,
        # 'from_array' channel windows: measured while level 0 is written
        window_from_data=bool(kwargs.get('window_from_data', False)),
        windows=kwargs.get('windows', 'keep'),
        # dyna's bytes backend: 'tensorstore' or 'zarrista'
        io_backend=kwargs.get('io_backend', 'tensorstore'),
        # the whole write's memory (zarrista: big shards tiled in y/x within it);
        # not for a dask source, which dask's scheduler writes
        memory_budget_mb=_memory_budget([arr], kwargs),
    )
