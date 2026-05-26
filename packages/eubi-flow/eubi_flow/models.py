"""Pydantic data models for EuBI-Bridge workflows."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Per-section config models (mirror EuBIBridge's ClusterConfig / DownscaleConfig)
# ---------------------------------------------------------------------------

class FlowClusterConfig(BaseModel):
    """Cluster / concurrency parameters for flow execution.

    Mirrors ``ClusterConfig`` from ``core/config_models.py`` so that the same
    mental model and parameter names apply across conversion and workflow runs.
    """

    max_workers: int = 4
    region_size_mb: float = 256.0
    use_threading: bool = False
    on_local_cluster: bool = True
    on_slurm: bool = False
    memory_per_worker: str = "4GB"
    slurm_partition: Optional[str] = None
    slurm_account: Optional[str] = None
    slurm_time: str = "24:00:00"
    slurm_worker_timeout: int = 300


class FlowDownscaleConfig(BaseModel):
    """Pyramid / downscale parameters for output heaves.

    Mirrors ``DownscaleConfig`` from ``core/config_models.py`` — per-axis
    scale factors, number of levels, and the TensorStore downsampling method.
    """

    n_layers: int = 5
    min_dimension_size: int = 64
    time_scale_factor: int = 1
    channel_scale_factor: int = 1
    z_scale_factor: int = 2
    y_scale_factor: int = 2
    x_scale_factor: int = 2
    downscale_method: Literal[
        "simple", "mean", "median", "min", "max", "mode"
    ] = "simple"

    def as_scale_factor_tuple(self) -> tuple[int, ...]:
        """Return scale factors as a ``(t, c, z, y, x)`` tuple."""
        return (
            self.time_scale_factor,
            self.channel_scale_factor,
            self.z_scale_factor,
            self.y_scale_factor,
            self.x_scale_factor,
        )


class FlowConfig(BaseModel):
    """Top-level config block embedded in ``FlowSpec``.

    Analogous to the cluster / conversion / downscale separation in
    EuBIBridge.  Updated via ``eubi flow configure_cluster NAME`` and
    ``eubi flow configure_downscale NAME``.

    Wave-specific parameters live directly on each ``WaveSpec`` — there is no
    separate engine section.
    """

    cluster:  FlowClusterConfig   = Field(default_factory=FlowClusterConfig)
    downscale: FlowDownscaleConfig = Field(default_factory=FlowDownscaleConfig)


# ---------------------------------------------------------------------------
# Core DAG models
# ---------------------------------------------------------------------------

class HeaveSpec(BaseModel):
    """One OME-Zarr pyramid artifact in the pipeline.

    ``heave_000`` is always the flow input.  Each wave produces the next
    heave.  Metadata fields are populated when the heave is first written.
    """

    heave_id: str
    path: str
    axes: Optional[str] = None       # e.g. "tczyx"
    scales: Optional[dict] = None    # {axis: physical_pixel_size}
    units: Optional[dict] = None     # {axis: unit_string}
    dtype: Optional[str] = None
    shape: Optional[list[int]] = None


class WaveSpec(BaseModel):
    """One processing step in the flow.

    ``params`` is the single source of truth for wave parameters.  Values
    supplied at ``add_wave`` time are stored here; ``configure_engine`` updates
    them in-place.
    """

    wave_id: str
    name: str                              # registered processor key
    input_heave_ids: list[str]
    output_heave_id: str
    params: dict = Field(default_factory=dict)
    status: Literal["pending", "running", "done", "failed"] = "pending"


class FlowSpec(BaseModel):
    """The complete DAG workflow, serialisable to / from JSON."""

    flow_id: str
    description: str = ""
    workdir: str = ""                     # directory for heave .zarr output files
    config: FlowConfig = Field(default_factory=FlowConfig)
    waves: list[WaveSpec] = Field(default_factory=list)
    heaves: dict[str, HeaveSpec] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: Literal["pending", "running", "completed", "failed"] = "pending"

    # ── convenience helpers ───────────────────────────────────────────────

    def next_heave_id(self) -> str:
        i = 0
        while f"heave_{i:03d}" in self.heaves:
            i += 1
        return f"heave_{i:03d}"

    def next_wave_id(self) -> str:
        return f"wave_{len(self.waves):03d}"

    def last_heave_id(self) -> str:
        if not self.heaves:
            raise ValueError("Flow has no heaves yet.")
        return max(self.heaves.keys())

    def heave_path(self, heave_id: str) -> str:
        """Return the output path for a heave zarr inside workdir."""
        import pathlib
        if not self.workdir:
            raise ValueError(
                "FlowSpec.workdir is not set. "
                "Provide --workdir when creating the flow."
            )
        return str(pathlib.Path(self.workdir) / f"{heave_id}.zarr")

    def effective_wave_params(self, wave: WaveSpec) -> dict:
        """Return the wave's parameters (single source of truth)."""
        return dict(wave.params)
