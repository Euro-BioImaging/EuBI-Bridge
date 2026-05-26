"""EuBIFlow — entry class for the workflow subsystem.

Flow files are stored in ``~/.eubi_bridge/.flows/`` as ``{name}.json``,
mirroring the way ``ConfigManager`` stores named configs under
``~/.eubi_bridge/``.  Heave output data (OME-Zarr) is written to
``workdir``, which is stored inside the flow JSON.

Configuration follows the same three-level merge as EuBIBridge::

    stored config  <  configure_* commands  <  run-time CLI flags

CLI surface::

    # Create
    eubi flow create myflow --input_path /data/input.zarr --workdir /data/work/myflow

    # Configure flow-wide settings (persisted in the flow JSON)
    eubi flow configure_cluster  myflow --max_workers 8
    eubi flow configure_downscale myflow --n_layers 3 --z_scale_factor 1

    # Edit DAG (all node-level ops go through select)
    eubi flow select myflow add_wave gaussian_filter --output_heave blurred --z_sigma 2.0
    eubi flow select myflow add_wave threshold_otsu  --input_heave blurred --output_heave binary_mask
    eubi flow select myflow update_wave wave_000 --z_sigma 4.0
    eubi flow select myflow update_wave wave_001 --wave_name threshold_fixed --threshold 128
    eubi flow select myflow update_wave wave_001 --output_heave final_mask
    eubi flow select myflow list_waves
    eubi flow select myflow show

    # Execute (CLI flags override stored config for this run only)
    eubi flow run myflow
    eubi flow run myflow --max_workers 16

    # Manage
    eubi flow list_flows
    eubi flow show_flow   myflow
    eubi flow delete_flow myflow
    eubi flow list_waves        # registered processors
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

from typing import Callable


# ---------------------------------------------------------------------------
# Core async execution — single source of truth for CLI and GUI
# ---------------------------------------------------------------------------

async def _run_flow_async(
    name: str,
    flow: "FlowSpec",
    *,
    input_path: Optional[str] = None,
    includes: Optional[str] = None,
    excludes: Optional[str] = None,
    client=None,
    max_concurrent: int = 4,
    region_size_mb: float = 256.0,
    n_layers: int = 5,
    scale_factor: tuple = (1, 1, 2, 2, 2),
    downscale_method: str = "simple",
    overwrite: bool = False,
    on_log: Optional[Callable[[str], None]] = None,
    on_wave_status: Optional[Callable[[str, str], None]] = None,
    on_wave_error: Optional[Callable[[str, str], None]] = None,
) -> list:
    """Execute a flow against one or more OME-Zarrs.  Shared by CLI and GUI.

    Handles in order:
    1. Per-flow workdir creation (``workdir / name /``)
    2. OME-Zarr discovery via ``scan_ome_zarrs``
    3. Per-file flow preparation (``_prepare_file_flow``)
    4. Concurrent execution via ``execute_flow``

    Returns a list of completed ``FlowSpec`` objects (one per input zarr).
    Progress is reported through the optional callbacks; pass ``print`` for
    ``on_log`` to get CLI-style output.
    """
    import asyncio as _asyncio
    from eubi_flow.executor import execute_flow
    from eubi_flow.batch import scan_ome_zarrs, _prepare_file_flow

    def _log(msg: str) -> None:
        if on_log:
            on_log(msg)

    # Deep-copy so the stored JSON is never mutated by a run.
    flow = flow.model_copy(deep=True)

    # Per-flow workdir: workdir / name /
    flow_workdir = Path(flow.workdir) / name
    flow_workdir.mkdir(parents=True, exist_ok=True)
    flow.workdir = str(flow_workdir)
    for hid, heave in flow.heaves.items():
        if hid != "heave_000":
            heave.path = flow.heave_path(hid)

    # Discover input zarrs
    src = input_path or flow.heaves["heave_000"].path
    zarr_paths = scan_ome_zarrs(src, includes=includes, excludes=excludes)
    if not zarr_paths:
        raise ValueError(
            f"No valid OME-Zarr stores found at '{src}'. "
            "Check the path or use --includes / --excludes to filter."
        )
    _log(f"Found {len(zarr_paths)} OME-Zarr(s) to process:")
    for p in zarr_paths:
        _log(f"  {p}")

    exec_kwargs = dict(
        client=client,
        region_size_mb=region_size_mb,
        n_layers=n_layers,
        scale_factor=scale_factor,
        downscale_method=downscale_method,
        overwrite=overwrite,
        on_wave_status=on_wave_status,
        on_wave_error=on_wave_error,
    )

    sem = _asyncio.Semaphore(max_concurrent)

    async def _run_one(zarr_path: str):
        async with sem:
            file_flow = _prepare_file_flow(flow, zarr_path)
            stem = Path(zarr_path).stem
            _log(f"Starting '{stem}'…")
            try:
                result = await execute_flow(file_flow, **exec_kwargs)
                _log(f"Finished '{stem}' — {result.status}")
                return result
            except Exception as exc:
                _log(f"Failed  '{stem}': {exc}")
                file_flow.status = "failed"
                return file_flow

    results = list(await _asyncio.gather(
        *[_asyncio.create_task(_run_one(p)) for p in zarr_paths]
    ))
    return results


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _flows_dir() -> Path:
    """Return ``~/.eubi_bridge/.flows/``, creating it if needed."""
    p = Path(os.path.expanduser("~")) / ".eubi_bridge" / ".flows"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _flow_path(name: str) -> Path:
    return _flows_dir() / f"{name}.json"


def _validate_name(name: str) -> None:
    if not re.match(r"^[A-Za-z0-9_\-]+$", name):
        raise ValueError(
            f"Flow name '{name}' must contain only letters, digits, "
            "hyphens, and underscores."
        )


def _load(name: str):
    """Load a named flow or raise KeyError with a helpful message."""
    from eubi_flow.serialization import load_flow
    _validate_name(name)
    path = _flow_path(name)
    if not path.exists():
        available = sorted(p.stem for p in _flows_dir().glob("*.json"))
        raise KeyError(
            f"Flow '{name}' not found in {_flows_dir()}. "
            f"Available: {available or ['(none)']}"
        )
    return load_flow(path)


# ---------------------------------------------------------------------------
# FlowEditor — DAG editing sub-namespace
# ---------------------------------------------------------------------------

class FlowEditor:
    """Sub-namespace returned by ``EuBIFlow.select(name)``.

    Supports Python Fire method-chaining::

        eubi flow select myflow add_wave gaussian_filter --z_sigma 2.0
        eubi flow select myflow list_waves
        eubi flow select myflow show
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._path = _flow_path(name)
        self._flow = _load(name)

    def add_wave(
        self,
        wave_name: str,
        input_heave: Optional[str] = None,
        input_heaves: Optional[str] = None,
        output_heave: Optional[str] = None,
        **params,
    ) -> None:
        """Append a processing wave.

        Parameters
        ----------
        wave_name : str
            Registered processor key (e.g. ``gaussian_filter``).
        input_heave : str, optional
            Name of the single input heave (default: last heave in the flow).
        input_heaves : str, optional
            Comma-separated heave names for fan-in, e.g. ``"heave_000,blurred"``.
        output_heave : str, optional
            Name for the output heave.  Defaults to the next auto-generated
            ``heave_NNN``.  Must be unique within the flow.

        Fan-out is implicit: two waves that share the same input heave run
        concurrently.  Extra keyword arguments are stored as wave parameters.

        Examples::

            eubi flow select myflow add_wave gaussian_filter
            eubi flow select myflow add_wave gaussian_filter --output_heave blurred
            eubi flow select myflow add_wave threshold_otsu \\
                --input_heave blurred --output_heave binary_mask
        """
        from eubi_flow.registry import get_processor
        from eubi_flow.models import WaveSpec, HeaveSpec
        from eubi_flow.serialization import save_flow

        get_processor(wave_name)   # validate name exists in registry

        # --- resolve input heave(s) ---
        if input_heaves is not None:
            input_ids = [h.strip() for h in input_heaves.split(",")]
        elif input_heave is not None:
            input_ids = [input_heave]
        else:
            input_ids = [self._flow.last_heave_id()]

        for hid in input_ids:
            if hid not in self._flow.heaves:
                raise ValueError(
                    f"Input heave '{hid}' does not exist. "
                    f"Available: {sorted(self._flow.heaves)}"
                )

        # --- resolve output heave name ---
        out_id = output_heave if output_heave is not None else self._flow.next_heave_id()
        if out_id in self._flow.heaves:
            raise ValueError(
                f"Heave '{out_id}' already exists. "
                "Choose a different --output_heave name."
            )

        out_path = self._flow.heave_path(out_id)
        self._flow.heaves[out_id] = HeaveSpec(heave_id=out_id, path=out_path)
        new_wave_id = self._flow.next_wave_id()
        self._flow.waves.append(WaveSpec(
            wave_id=new_wave_id,
            name=wave_name,
            input_heave_ids=input_ids,
            output_heave_id=out_id,
            params=dict(params),
        ))
        save_flow(self._flow, self._path)
        print(f"Added {new_wave_id} ({wave_name}) {input_ids} → {out_id}  [{out_path}]")

    def remove_wave(self, wave_id: str) -> None:
        """Remove a wave and its output heave.

        Raises ``ValueError`` if the output heave is still consumed by another
        wave downstream — remove those first.
        """
        from eubi_flow.serialization import save_flow

        matches = [w for w in self._flow.waves if w.wave_id == wave_id]
        if not matches:
            raise ValueError(
                f"Wave '{wave_id}' not found. "
                f"Available: {sorted(w.wave_id for w in self._flow.waves)}"
            )
        wave = matches[0]
        downstream = [
            w for w in self._flow.waves
            if wave.output_heave_id in w.input_heave_ids and w.wave_id != wave_id
        ]
        if downstream:
            raise ValueError(
                f"Cannot remove {wave_id}: heave '{wave.output_heave_id}' "
                f"is consumed by {[w.wave_id for w in downstream]}. "
                "Remove downstream waves first."
            )
        del self._flow.heaves[wave.output_heave_id]
        self._flow.waves.remove(wave)
        save_flow(self._flow, self._path)
        print(f"Removed {wave_id} and its output heave {wave.output_heave_id}.")

    def list_waves(self) -> None:
        """Print all waves in the flow with their effective parameters."""
        if not self._flow.waves:
            print("No waves defined yet.")
            return
        header = (
            f"{'Wave':<12} {'Processor':<22} {'Input(s)':<20} "
            f"{'Output':<14} {'Status':<10} Effective params"
        )
        print(header)
        print("-" * len(header))
        for w in self._flow.waves:
            eff    = self._flow.effective_wave_params(w)
            inputs = ",".join(w.input_heave_ids)
            print(
                f"{w.wave_id:<12} {w.name:<22} {inputs:<20} "
                f"{w.output_heave_id:<14} {w.status:<10} {eff}"
            )

    def update_wave(
        self,
        wave_id: str,
        wave_name: Optional[str] = None,
        input_heave: Optional[str] = None,
        input_heaves: Optional[str] = None,
        output_heave: Optional[str] = None,
        **params,
    ) -> None:
        """Update an existing wave — processor, heave wiring, or parameters.

        Parameters
        ----------
        wave_id : str
            Wave identifier, e.g. ``wave_002``.
        wave_name : str, optional
            Swap the processor (e.g. ``threshold_otsu``).  When given,
            ``params`` replaces the existing parameter set entirely.
            When omitted, ``params`` is merged into the existing set.
        input_heave : str, optional
            Rewire the wave to read from a different single input heave.
        input_heaves : str, optional
            Rewire to multiple input heaves (comma-separated, for fan-in).
        output_heave : str, optional
            Rename the output heave.  Updates the dict key, ``heave_id``,
            the derived path, and any downstream ``input_heave_ids`` that
            reference it.  The old path is **not** renamed on disk.
        **params
            Wave-specific parameters, e.g. ``--threshold 128``.

        Examples::

            eubi flow select myflow update_wave wave_002 --threshold 128
            eubi flow select myflow update_wave wave_002 --wave_name threshold_otsu
            eubi flow select myflow update_wave wave_002 --output_heave binary_mask
            eubi flow select myflow update_wave wave_002 --input_heave blurred
        """
        from eubi_flow.registry import get_processor
        from eubi_flow.serialization import save_flow

        ids = {w.wave_id for w in self._flow.waves}
        if wave_id not in ids:
            raise ValueError(
                f"Wave '{wave_id}' not found. Available: {sorted(ids)}"
            )
        wave = next(w for w in self._flow.waves if w.wave_id == wave_id)

        # --- processor / params ---
        if wave_name is not None:
            get_processor(wave_name)
            wave.name   = wave_name
            wave.params = dict(params)
        elif params:
            wave.params.update(params)

        # --- rewire inputs ---
        if input_heaves is not None or input_heave is not None:
            new_inputs = (
                [h.strip() for h in input_heaves.split(",")]
                if input_heaves is not None
                else [input_heave]
            )
            for hid in new_inputs:
                if hid not in self._flow.heaves:
                    raise ValueError(
                        f"Input heave '{hid}' does not exist. "
                        f"Available: {sorted(self._flow.heaves)}"
                    )
            wave.input_heave_ids = new_inputs

        # --- rename output heave ---
        if output_heave is not None:
            old_id = wave.output_heave_id
            if output_heave != old_id and output_heave in self._flow.heaves:
                raise ValueError(
                    f"Heave '{output_heave}' already exists. "
                    "Choose a different name."
                )
            heave_spec = self._flow.heaves.pop(old_id)
            heave_spec.heave_id = output_heave
            heave_spec.path     = self._flow.heave_path(output_heave)
            self._flow.heaves[output_heave] = heave_spec
            wave.output_heave_id = output_heave
            # patch any downstream waves that read the old heave id
            for w in self._flow.waves:
                w.input_heave_ids = [
                    output_heave if hid == old_id else hid
                    for hid in w.input_heave_ids
                ]

        save_flow(self._flow, self._path)
        print(f"Updated {wave_id}:")
        print(f"  processor   : {wave.name}")
        print(f"  inputs      : {wave.input_heave_ids}")
        print(f"  output      : {wave.output_heave_id}")
        print(f"  params      : {wave.params}")

    def show(self) -> None:
        """Pretty-print the full FlowSpec as JSON."""
        print(self._flow.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# EuBIFlow — top-level entry class
# ---------------------------------------------------------------------------

class EuBIFlow:
    """Entry point for EuBI-Bridge workflow operations.

    Flow files live in ``~/.eubi_bridge/.flows/``.
    Wired into the CLI via the ``flow`` property on ``EuBIBridge``.
    """

    # ── flow lifecycle ────────────────────────────────────────────────────

    def create(
        self,
        name: str,
        input_path: str,
        workdir: str,
        description: str = "",
    ) -> None:
        """Create a named flow.

        The flow JSON is saved to ``~/.eubi_bridge/.flows/{name}.json``.
        Output heave data will be written inside ``workdir`` at run time.

        ``input_path`` may be:

        * A **single OME-Zarr directory** — metadata is read immediately
          and stored in ``heave_000``.
        * A **directory of OME-Zarr stores** or **glob pattern** — the path
          is stored as-is; metadata will be read per file at run time.

        Parameters
        ----------
        name : str
            Short identifier (letters, digits, hyphens, underscores).
        input_path : str
            OME-Zarr path, directory of OME-Zarrs, or glob pattern.
        workdir : str
            Root directory for output heave ``.zarr`` files.  A per-file
            subdirectory is created at run time for each input OME-Zarr.
        description : str, optional
            Human-readable note stored in the flow JSON.
        """
        import uuid
        from eubi_flow.models import FlowSpec, HeaveSpec
        from eubi_flow.serialization import save_flow
        from eubi_bridge.utils.path_utils import is_ome_zarr

        _validate_name(name)
        flow_path = _flow_path(name)

        # Read metadata only when input_path is a single OME-Zarr
        axes = scales = units = shape = dtype = None
        single_zarr = Path(input_path).is_dir() and is_ome_zarr(str(input_path))
        if single_zarr:
            try:
                from eubi_bridge.core.pyramid_reader import read_pyramid
                reader = read_pyramid(str(input_path))
                pyr5d  = reader.pyr.to5D()
                axes   = pyr5d.axes
                scales = pyr5d.meta.scaledict.get("0", {})
                units  = pyr5d.meta.unit_dict
                shape  = list(pyr5d.base_array.shape)
                dtype  = str(pyr5d.base_array.dtype)
            except Exception as exc:
                import warnings
                warnings.warn(
                    f"Could not read metadata from {input_path}: {exc}.",
                    stacklevel=2,
                )

        flow = FlowSpec(
            flow_id=str(uuid.uuid4())[:8],
            description=description,
            workdir=str(workdir),
            heaves={"heave_000": HeaveSpec(
                heave_id="heave_000", path=str(input_path),
                axes=axes, scales=scales, units=units, shape=shape, dtype=dtype,
            )},
        )
        Path(workdir).mkdir(parents=True, exist_ok=True)
        save_flow(flow, flow_path)
        print(f"Created flow '{name}' → {flow_path}")
        print(f"  input  : {input_path}")
        print(f"  workdir: {workdir}")
        if single_zarr and axes:
            print(f"  axes={axes}  shape={shape}  dtype={dtype}")
        elif not single_zarr:
            print("  (batch input — per-file metadata will be read at run time)")

    # ── configuration commands ────────────────────────────────────────────

    def configure_cluster(self, name: str, **kwargs) -> None:
        """Update cluster / concurrency parameters for a named flow.

        Omitted arguments keep their current values.  Changes are persisted
        immediately to ``~/.eubi_bridge/.flows/{name}.json``.

        Parameters (same names as EuBIBridge's configure_cluster)
        ----------------------------------------------------------
        max_workers, region_size_mb, use_threading, on_local_cluster,
        on_slurm, memory_per_worker, slurm_partition, slurm_account,
        slurm_time, slurm_worker_timeout
        """
        from eubi_flow.serialization import save_flow

        flow = _load(name)
        _update_model(flow.config.cluster, kwargs)
        save_flow(flow, _flow_path(name))
        print(f"Updated cluster config for flow '{name}'.")
        _show_model(flow.config.cluster)

    def configure_downscale(self, name: str, **kwargs) -> None:
        """Update pyramid / downscale parameters for a named flow.

        Omitted arguments keep their current values.

        Parameters (same names as EuBIBridge's configure_downscale)
        -----------------------------------------------------------
        n_layers, min_dimension_size, time_scale_factor, channel_scale_factor,
        z_scale_factor, y_scale_factor, x_scale_factor, downscale_method
        """
        from eubi_flow.serialization import save_flow

        flow = _load(name)
        _update_model(flow.config.downscale, kwargs)
        save_flow(flow, _flow_path(name))
        print(f"Updated downscale config for flow '{name}'.")
        _show_model(flow.config.downscale)

    def show_config(self, name: str) -> None:
        """Display the stored config (cluster + downscale + engine) for a flow."""
        flow = _load(name)
        print(flow.config.model_dump_json(indent=2))

    # ── pre-flight validation ─────────────────────────────────────────────

    def lint(self, name: str) -> None:
        """Validate flow parameters without running anything.

        Resolves effective parameters for every wave and checks them against
        the wave's ``Params`` model.  Reports all issues upfront so nothing
        is written to disk when the flow is misconfigured.

        Use this before ``run`` to catch missing required fields (e.g. a
        ``threshold_fixed`` wave without a ``threshold`` value) or out-of-range
        values (e.g. ``percentile=150``) early.

        Example::

            eubi flow lint myflow
        """
        from eubi_flow.validation import lint_flow

        flow   = _load(name)
        errors = lint_flow(flow)
        if errors:
            print(f"Flow '{name}' has {len(errors)} validation error(s):")
            for e in errors:
                print(f"  • {e}")
        else:
            print(f"Flow '{name}' is valid — all wave parameters OK.")

    # ── execution ─────────────────────────────────────────────────────────

    def run(
        self,
        name: str,
        # ── input override ────────────────────────────────────────────────
        input_path: Optional[str] = None,
        includes: Optional[str] = None,
        excludes: Optional[str] = None,
        # ── cluster overrides (None → use stored config) ──────────────────
        max_workers: Optional[int] = None,
        region_size_mb: Optional[float] = None,
        on_local_cluster: Optional[bool] = None,
        on_slurm: Optional[bool] = None,
        scheduler_address: Optional[str] = None,
        # ── downscale overrides ───────────────────────────────────────────
        n_layers: Optional[int] = None,
        z_scale_factor: Optional[int] = None,
        y_scale_factor: Optional[int] = None,
        x_scale_factor: Optional[int] = None,
        downscale_method: Optional[str] = None,
        # ── output ───────────────────────────────────────────────────────
        overwrite: bool = False,
    ) -> None:
        """Execute a named flow against one or more OME-Zarr stores.

        The input is resolved as follows:

        1. ``--input_path`` (if given) overrides the path stored in the flow.
        2. The resolved path is scanned with ``scan_ome_zarrs``; every valid
           OME-Zarr found is processed.
        3. Each OME-Zarr runs the same flow independently in parallel,
           writing its heaves to ``workdir/<zarr_stem>/``.

        Stored ``config.cluster`` and ``config.downscale`` are used as
        defaults; any argument supplied here overrides them for this run only.

        Parameters
        ----------
        name : str
            Flow to run (must exist in ``~/.eubi_bridge/.flows/``).
        input_path : str, optional
            OME-Zarr, directory, or glob that overrides the stored input.
        includes : str, optional
            Comma-separated substrings — only matching paths are processed.
        excludes : str, optional
            Comma-separated substrings — matching paths are skipped.
        """
        import asyncio
        from dask.distributed import Client, LocalCluster
        from eubi_flow.validation import lint_flow, FlowValidationError

        flow = _load(name)

        # ── pre-flight validation ─────────────────────────────────────────
        errors = lint_flow(flow)
        if errors:
            raise FlowValidationError(errors)

        # ── resolve effective config ──────────────────────────────────────
        cc = flow.config.cluster
        dc = flow.config.downscale
        eff_max_workers    = max_workers      if max_workers      is not None else cc.max_workers
        eff_region_size_mb = region_size_mb   if region_size_mb   is not None else cc.region_size_mb
        eff_on_local       = on_local_cluster if on_local_cluster is not None else cc.on_local_cluster
        eff_on_slurm       = on_slurm         if on_slurm         is not None else cc.on_slurm
        eff_n_layers       = n_layers         if n_layers         is not None else dc.n_layers
        eff_z_sf           = z_scale_factor   if z_scale_factor   is not None else dc.z_scale_factor
        eff_y_sf           = y_scale_factor   if y_scale_factor   is not None else dc.y_scale_factor
        eff_x_sf           = x_scale_factor   if x_scale_factor   is not None else dc.x_scale_factor
        eff_method         = downscale_method if downscale_method is not None else dc.downscale_method
        eff_scale_factor   = (dc.time_scale_factor, dc.channel_scale_factor,
                              eff_z_sf, eff_y_sf, eff_x_sf)

        # ── spin up dask cluster ──────────────────────────────────────────
        cluster = None
        if scheduler_address:
            client = Client(scheduler_address)
        elif eff_on_local and not eff_on_slurm:
            cluster = LocalCluster(n_workers=eff_max_workers, threads_per_worker=1)
            client  = Client(cluster)
        else:
            client = Client()

        # ── execute via shared async core ─────────────────────────────────
        try:
            results = asyncio.run(_run_flow_async(
                name, flow,
                input_path=input_path,
                includes=includes,
                excludes=excludes,
                client=client,
                max_concurrent=eff_max_workers,
                region_size_mb=eff_region_size_mb,
                n_layers=eff_n_layers,
                scale_factor=eff_scale_factor,
                downscale_method=eff_method,
                overwrite=overwrite,
                on_log=print,
            ))
        finally:
            client.close()
            if cluster is not None:
                try:
                    cluster.close()
                except Exception:
                    pass

        # ── report ────────────────────────────────────────────────────────
        n_done   = sum(1 for r in results if r.status == "completed")
        n_failed = sum(1 for r in results if r.status == "failed")
        print(f"\nBatch complete: {n_done}/{len(results)} succeeded, "
              f"{n_failed} failed.")
        for result in results:
            wave_summary = ", ".join(f"{w.wave_id}:{w.status}" for w in result.waves)
            print(f"  [{result.status:<10}] ({wave_summary})")

    # ── DAG editing ───────────────────────────────────────────────────────

    def select(self, name: str) -> FlowEditor:
        """Return a ``FlowEditor`` for DAG editing (Fire sub-namespace).

        Example::

            eubi flow select myflow add_wave gaussian_filter --z_sigma 2.0
        """
        return FlowEditor(name)

    # ── flow management ───────────────────────────────────────────────────

    def list_flows(self) -> None:
        """List all flows in ``~/.eubi_bridge/.flows/``."""
        flows_dir = _flows_dir()
        entries   = sorted(flows_dir.glob("*.json"))
        if not entries:
            print(f"No flows found in {flows_dir}.")
            return
        from eubi_flow.serialization import load_flow
        print(f"Flows in {flows_dir}:")
        print(f"  {'Name':<24} {'ID':<10} {'Status':<12} {'Waves':<6} Workdir")
        print("  " + "-" * 72)
        for p in entries:
            try:
                f = load_flow(p)
                print(
                    f"  {p.stem:<24} {f.flow_id:<10} {f.status:<12} "
                    f"{len(f.waves):<6} {f.workdir}"
                )
            except Exception as exc:
                print(f"  {p.stem:<24} (unreadable: {exc})")

    def show_flow(self, name: str) -> None:
        """Pretty-print the full JSON of a named flow."""
        print(_load(name).model_dump_json(indent=2))

    def delete_flow(self, name: str) -> None:
        """Delete the flow JSON from ``~/.eubi_bridge/.flows/``.

        Heave data in ``workdir`` is **not** removed.
        """
        _validate_name(name)
        path = _flow_path(name)
        if not path.exists():
            raise KeyError(f"Flow '{name}' not found.")
        path.unlink()
        print(f"Deleted flow '{name}' ({path}).")
        print("Note: heave data in workdir was not removed.")

    # ── processor registry ────────────────────────────────────────────────

    def list_waves(self) -> None:
        """Print all registered wave processor names and types."""
        from eubi_flow.registry import list_processors
        procs = list_processors()
        if not procs:
            print("No processors registered.")
            return
        print(f"{'Name':<26} {'Type':<14} Module")
        print("-" * 60)
        for pname, cls in sorted(procs.items()):
            inst   = cls()
            wtype  = inst.wave_type().value
            module = cls.__module__.split(".")[-1]
            print(f"{pname:<26} {wtype:<14} {module}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _update_model(model, updates: dict) -> None:
    """Apply non-None updates to a Pydantic model in-place."""
    for key, val in updates.items():
        if val is None:
            continue
        if not hasattr(model, key):
            raise ValueError(
                f"Unknown parameter '{key}' for {type(model).__name__}. "
                f"Valid: {list(model.model_fields)}"
            )
        setattr(model, key, val)


def _show_model(model) -> None:
    """Print a Pydantic model as indented JSON."""
    print(model.model_dump_json(indent=2))
