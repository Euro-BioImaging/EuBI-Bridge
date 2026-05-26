"""Workflow executor — concurrent wave scheduling and lazy region I/O.

Wave-level concurrency
----------------------
All waves are launched as concurrent ``asyncio`` tasks.  Each task waits on an
``asyncio.Event`` for every one of its input heaves before starting, then sets
the event for its output heave when done.  This means:

* Independent branches run in parallel automatically.
* Fan-in waves block until ALL their inputs are ready.
* Sequential chains enforce ordering without any topological sort.

Lazy processing and I/O
-----------------------
``processor.process(dask_array, params)`` builds a **lazy dask graph** — no
pixels are read or computed at this point.  The lazy result is passed directly
to ``store_multiscale_async``, which partitions it into spatial regions
internally and streams reads → writes through its queue-based writer.  No
explicit ``.compute()`` or materialisation occurs in this module.

The ``client`` parameter is accepted for future cluster-level wave dispatch
(e.g. submitting each wave as a dask future to a SLURM cluster).  For now,
wave parallelism is handled by asyncio tasks; the ``store_multiscale_async``
writer uses its own thread pool for concurrent I/O.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from dask.distributed import Client
    from eubi_flow.models import FlowSpec, WaveSpec


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def execute_flow(
    flow: "FlowSpec",
    client: Optional["Client"] = None,
    region_size_mb: float = 256.0,
    n_layers: int = 5,
    scale_factor: tuple[int, ...] = (1, 1, 2, 2, 2),
    downscale_method: str = "simple",
    overwrite: bool = False,
    on_wave_status: Optional[Callable[[str, str], None]] = None,
    on_wave_error:  Optional[Callable[[str, str], None]] = None,
) -> "FlowSpec":
    """Execute all waves in the flow with full DAG concurrency.

    Parameters
    ----------
    flow : FlowSpec
        The flow to execute (wave statuses updated in-place).
    client : dask.distributed.Client, optional
        Connected dask client.  Reserved for future cluster-level wave
        dispatch; not used for I/O (handled by store_multiscale_async).
    region_size_mb : float
        Target region size forwarded to ``store_multiscale_async``.
    n_layers : int
        Number of pyramid levels to generate per output heave.
    scale_factor : tuple[int, ...]
        Per-axis downscale factors in tczyx order applied at each level.
    downscale_method : str
        TensorStore downsampling method passed to the writer.
    """
    from eubi_flow.validation import lint_flow, FlowValidationError
    errors = lint_flow(flow)
    if errors:
        raise FlowValidationError(errors)

    # One Event per heave — set when that heave is ready to read.
    heave_ready: dict[str, asyncio.Event] = {
        hid: asyncio.Event() for hid in flow.heaves
    }
    # Input heaves (not produced by any wave) are available immediately.
    produced = {w.output_heave_id for w in flow.waves}
    for hid in flow.heaves:
        if hid not in produced:
            heave_ready[hid].set()

    async def run_wave(wave: "WaveSpec") -> None:
        await asyncio.gather(*[heave_ready[hid].wait()
                                for hid in wave.input_heave_ids])
        wave.status = "running"
        if on_wave_status:
            on_wave_status(wave.wave_id, "running")
        try:
            await _execute_wave(
                flow, wave,
                region_size_mb, n_layers, scale_factor, downscale_method, overwrite,
            )
            wave.status = "done"
        except Exception:
            wave.status = "failed"
            if on_wave_error:
                import traceback as _tb
                on_wave_error(wave.wave_id, _tb.format_exc())
            raise
        finally:
            if on_wave_status:
                on_wave_status(wave.wave_id, wave.status)
            heave_ready[wave.output_heave_id].set()

    await asyncio.gather(*[asyncio.create_task(run_wave(w)) for w in flow.waves],
                         return_exceptions=True)
    flow.status = "completed" if all(w.status == "done" for w in flow.waves) else "failed"
    return flow


# ---------------------------------------------------------------------------
# Single-wave execution
# ---------------------------------------------------------------------------

async def _execute_wave(
    flow: "FlowSpec",
    wave: "WaveSpec",
    region_size_mb: float,
    n_layers: int,
    scale_factor: tuple[int, ...],
    downscale_method: str,
    overwrite: bool = False,
) -> None:
    """Read input pyramid → apply processor lazily → write output pyramid."""
    from eubi_bridge.core.writers import store_multiscale_async
    from eubi_bridge.core.pyramid_reader import read_pyramid
    from eubi_flow.registry import get_processor

    # MVP: first input heave only (fan-in processing deferred)
    in_heave  = flow.heaves[wave.input_heave_ids[0]]
    out_heave = flow.heaves[wave.output_heave_id]

    # --- Read input pyramid, always expand to 5D (tczyx) ---
    reader   = read_pyramid(in_heave.path)
    pyr5d    = reader.pyr.to5D()
    base_arr = pyr5d.base_array            # lazy dask array

    # --- Derive output metadata via processor descriptors ---
    processor  = get_processor(wave.name)
    in_axes    = pyr5d.axes
    in_scales  = pyr5d.meta.scaledict.get("0", {ax: 1.0 for ax in in_axes})
    in_units   = pyr5d.meta.unit_dict

    # Resolve effective params: add_wave params < engine config defaults
    effective_params = flow.effective_wave_params(wave)

    out_axes   = processor.output_axes(in_axes, effective_params)
    out_shape  = processor.output_shape(tuple(base_arr.shape), effective_params)
    out_scales = processor.output_scales(in_scales, effective_params)
    out_units  = processor.output_units(in_units, effective_params)

    # --- Resolve effective wave params (add_wave < engine config) ---
    effective_params = flow.effective_wave_params(wave)

    # --- Apply processing lazily using effective params ---
    processed = processor.process(base_arr, effective_params)

    # --- Determine pyramid scale factors ---
    has_spatial = any(ax in out_axes for ax in "zyx")
    if n_layers > 1 and has_spatial:
        sf        = tuple(_align_scale_factor(scale_factor, out_axes))
        use_layers = n_layers
    else:
        sf        = None
        use_layers = 0

    # --- Write full OME-Zarr pyramid via store_multiscale_async ---
    # store_multiscale_async reads the lazy dask graph region-by-region
    # through its internal queue-based writer — no external .compute() needed.
    axis_list   = list(out_axes)
    base_scales = [float(out_scales.get(ax, 1.0)) for ax in axis_list]
    unit_list   = [str(out_units.get(ax, "")) for ax in axis_list]

    await store_multiscale_async(
        arr=processed,
        output_path=str(out_heave.path),
        axes=axis_list,
        scales=tuple(base_scales),     # flat per-axis pixel sizes, same format as parse_scales()
        units=unit_list,
        scale_factors=sf,
        n_layers=use_layers,
        downscale_method=downscale_method,
        region_size_mb=region_size_mb,
        overwrite=overwrite,
    )

    # --- Persist output metadata in HeaveSpec ---
    out_heave.axes   = out_axes
    out_heave.scales = out_scales
    out_heave.units  = out_units
    out_heave.dtype  = str(processed.dtype)
    out_heave.shape  = list(out_shape)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _align_scale_factor(
    scale_factor: tuple[int, ...],
    axes: str,
) -> list[int]:
    """Return a per-axis scale-factor list aligned to ``axes``.

    ``scale_factor`` is always provided in tczyx order (5 values).
    For output arrays with fewer axes the corresponding values are selected.
    """
    sf_map = {ax: sf for ax, sf in zip("tczyx", scale_factor)}
    return [sf_map.get(ax, 1) for ax in axes]
