"""Batch execution utilities for EuBI-Bridge workflows.

A flow is always defined for a single OME-Zarr.  Batch execution means
running the same flow independently against every OME-Zarr found in the
input path, in parallel up to ``max_concurrent`` at a time.

For each input OME-Zarr the engine:
1. Deep-copies the flow template.
2. Points ``heave_000`` at that specific OME-Zarr.
3. Sets a per-file workdir:  ``<template_workdir>/<zarr_stem>/``.
4. Recalculates all output heave paths under the per-file workdir.
5. Runs ``execute_flow`` concurrently via an asyncio semaphore.

Public API
----------
scan_ome_zarrs(input_path, includes, excludes) -> list[str]
    Discover all valid OME-Zarr stores under input_path.
run_batch(template_flow, zarr_paths, client, **kwargs) -> list[FlowSpec]
    Execute the template flow once per zarr_path.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from dask.distributed import Client
    from eubi_flow.models import FlowSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OME-Zarr discovery
# ---------------------------------------------------------------------------

def scan_ome_zarrs(
    input_path: str,
    includes: Optional[str] = None,
    excludes: Optional[str] = None,
) -> list[str]:
    """Return every valid OME-Zarr store reachable from *input_path*.

    Resolution rules (mirroring ``take_filepaths_from_path``):

    * **Single OME-Zarr directory** — returned as-is after validation.
    * **Plain directory** — scanned recursively; ``.zarr`` boundaries are
      treated as atomic units (descent stops at ``.zarr``).
    * **Glob pattern** — expanded via ``sensitive_glob``.

    Each discovered path is validated with ``is_ome_zarr`` before being
    included.  Paths that fail validation are logged as warnings and skipped.

    Parameters
    ----------
    input_path : str
        File-system path, directory, or glob pattern.
    includes : str, optional
        Comma-separated substrings; only paths containing at least one are kept.
    excludes : str, optional
        Comma-separated substrings; paths containing any are removed.
    """
    from eubi_bridge.utils.path_utils import take_filepaths_from_path, is_ome_zarr

    # Fast path: input is itself an OME-Zarr
    if Path(input_path).is_dir() and is_ome_zarr(input_path):
        return [str(input_path)]

    # Scan via the existing path utility (handles dirs, globs, includes/excludes)
    try:
        candidates = take_filepaths_from_path(
            input_path, includes=includes, excludes=excludes
        )
    except ValueError:
        return []

    # Filter to .zarr paths that carry valid OME-NGFF metadata
    valid: list[str] = []
    for p in candidates:
        if not p.endswith(".zarr"):
            continue
        if is_ome_zarr(p):
            valid.append(p)
        else:
            logger.warning(
                "Skipping '%s' — directory ends in .zarr but contains no "
                "OME-NGFF metadata ('ome' or 'multiscales' attribute).", p
            )

    return valid


# ---------------------------------------------------------------------------
# Per-file flow preparation
# ---------------------------------------------------------------------------

def _prepare_file_flow(template_flow: "FlowSpec", zarr_path: str) -> "FlowSpec":
    """Return a deep copy of *template_flow* configured for *zarr_path*.

    Output layout uses **heave-first** organisation so every heave maps to
    one directory that collects results across all input zarrs::

        workdir/
        ├── heave_001/          (or a custom name like "blurred")
        │   ├── dataset_a.zarr
        │   └── dataset_b.zarr
        └── heave_002/
            ├── dataset_a.zarr
            └── dataset_b.zarr

    Changes made to the copy:
    * ``heave_000.path`` → *zarr_path*
    * ``heave_000`` metadata → read from *zarr_path* (best-effort)
    * Each output heave path → ``<template_workdir>/<heave_id>/<zarr_stem>.zarr``
    * All wave / flow statuses → reset to ``"pending"``
    """
    flow = template_flow.model_copy(deep=True)
    stem = Path(zarr_path).stem

    # heave_000 — point at this file and refresh metadata
    h0 = flow.heaves["heave_000"]
    h0.path = zarr_path
    try:
        from eubi_bridge.core.pyramid_reader import read_pyramid
        reader = read_pyramid(zarr_path)
        pyr5d  = reader.pyr.to5D()
        h0.axes   = pyr5d.axes
        h0.scales = pyr5d.meta.scaledict.get("0", {})
        h0.units  = pyr5d.meta.unit_dict
        h0.shape  = list(pyr5d.base_array.shape)
        h0.dtype  = str(pyr5d.base_array.dtype)
    except Exception as exc:
        logger.warning("Could not read metadata from '%s': %s", zarr_path, exc)

    # Output heave paths: template_workdir / heave_id / zarr_stem.zarr
    for hid, heave in flow.heaves.items():
        if hid != "heave_000":
            heave_dir = Path(template_flow.workdir) / hid
            heave_dir.mkdir(parents=True, exist_ok=True)
            heave.path = str(heave_dir / f"{stem}.zarr")

    # Reset execution state
    for wave in flow.waves:
        wave.status = "pending"
    flow.status = "pending"

    return flow


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

async def run_batch(
    template_flow: "FlowSpec",
    zarr_paths: list[str],
    client: Optional["Client"],
    region_size_mb: float,
    n_layers: int,
    scale_factor: tuple[int, ...],
    downscale_method: str,
    max_concurrent: int,
    overwrite: bool = False,
) -> list["FlowSpec"]:
    """Execute *template_flow* once per entry in *zarr_paths*, in parallel.

    Concurrency is bounded by *max_concurrent* (an asyncio ``Semaphore``).
    Each file runs an independent ``execute_flow`` call; a failure for one
    file is caught, logged, and recorded without affecting the others.

    Parameters
    ----------
    template_flow : FlowSpec
        The flow definition to replicate for each file.
    zarr_paths : list[str]
        OME-Zarr paths discovered by ``scan_ome_zarrs``.
    client : dask.distributed.Client or None
        Dask client passed through to ``execute_flow``.
    max_concurrent : int
        Maximum number of files processed simultaneously.
    """
    from eubi_flow.executor import execute_flow

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _run_one(zarr_path: str) -> "FlowSpec":
        async with semaphore:
            flow = _prepare_file_flow(template_flow, zarr_path)
            try:
                logger.info("Starting flow for '%s'", zarr_path)
                result = await execute_flow(
                    flow, client,
                    region_size_mb=region_size_mb,
                    n_layers=n_layers,
                    scale_factor=scale_factor,
                    downscale_method=downscale_method,
                    overwrite=overwrite,
                )
                logger.info(
                    "Finished flow for '%s' — status: %s", zarr_path, result.status
                )
                return result
            except Exception as exc:
                logger.error("Flow failed for '%s': %s", zarr_path, exc)
                flow.status = "failed"
                return flow

    results = await asyncio.gather(*[asyncio.create_task(_run_one(p)) for p in zarr_paths])
    return list(results)
