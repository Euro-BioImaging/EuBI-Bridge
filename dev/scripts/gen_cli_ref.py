"""Build-time CLI reference generator for EuBI-Bridge.

Runs via the mkdocs-gen-files plugin before every `mkdocs build`.
Introspects EuBIBridge + Pydantic config models and writes
docs/cli_reference.md with collapsible command cards.

Can also be run standalone for debugging:
    python docs/scripts/gen_cli_ref.py
"""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path
from typing import Any, Union, get_args, get_origin

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so local imports work both inside
# mkdocs-gen-files and when run standalone.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eubi_bridge.ebridge import EuBIBridge, ConfigureGroup
from eubi_bridge.core.config_models import (
    ClusterConfig,
    ConversionConfig,
    ConcatenationConfig,
    DownscaleConfig,
    ReaderConfig,
    SUPPORTED_COMPRESSORS_V2,
    SUPPORTED_COMPRESSORS_V3,
)

# ---------------------------------------------------------------------------
# Command grouping — controls page section order and membership
# ---------------------------------------------------------------------------

COMMAND_GROUPS: dict[str, list[str]] = {
    "Conversion": ["to_zarr", "validate_aggregative"],
    "Metadata": ["show_pixel_meta", "update_pixel_meta", "update_channel_meta"],
    "Configuration": [
        "configure.cluster",
        "configure.conversion",
        "configure.downscale",
        "configure.readers",
        "configure.concatenation",
    ],
    "Named Configs": [
        "with_config",
        "save_as",
        "update_config",
        "list_configs",
        "show_configs",
        "delete_config",
    ],
    "Display & Info": ["show_config", "show_root_defaults", "version"],
    "Reset": ["reset_config"],
}

# Maps each command to the override sections that are actually relevant for it.
# Only sections listed here will be shown in that command's card.
COMMAND_KWARGS_SECTIONS: dict[str, list[str]] = {
    "to_zarr": [
        "Cluster overrides", "Conversion overrides", "Downscale overrides",
        "Reader overrides", "Concatenation overrides",
    ],
    "validate_aggregative": [
        "Cluster overrides", "Reader overrides", "Concatenation overrides",
    ],
}

# All available sections and their Pydantic models.
KWARGS_SECTIONS: dict[str, Any] = {
    "Cluster overrides":       ClusterConfig,
    "Conversion overrides":    ConversionConfig,
    "Downscale overrides":     DownscaleConfig,
    "Reader overrides":        ReaderConfig,
    "Concatenation overrides": ConcatenationConfig,
}

# All field names covered by config models — used to exclude duplicates from
# the "CLI-only options" section (they already appear in the override sections).
_ALL_CONFIG_FIELDS: frozenset[str] = frozenset(
    fname
    for model in KWARGS_SECTIONS.values()
    for fname in model.model_fields
)

# Human-readable descriptions for Pydantic model fields not covered by
# any method docstring (supplementary layer).
_FIELD_HINTS: dict[str, str] = {
    # ClusterConfig
    "max_workers": "Number of parallel file-level worker processes.",
    "queue_size": "Internal write-queue depth per worker.",
    "region_size_mb": "Region size in MB for spatial partitioning.",
    "memory_per_worker": "Memory limit for SLURM / LocalCluster workers.",
    "max_concurrency": "TensorStore write concurrency per worker.",
    "max_concurrent_scenes": "Parallel scenes per file (default 1).",
    "tensorstore_data_copy_concurrency": "TensorStore internal copy threads.",
    "max_retries": "Retries on a broken worker process.",
    "on_slurm": "Submit to a SLURM cluster.",
    "on_local_cluster": "Use a Dask LocalCluster backend.",
    "use_threading": "Use ThreadPool instead of ProcessPool.",
    "slurm_time": "SLURM wall-clock limit, e.g. '24:00:00'.",
    "slurm_account": "SLURM account name.",
    "slurm_partition": "SLURM partition / queue.",
    "slurm_worker_timeout": "Seconds to wait for SLURM workers to start.",
    "slurm_sif_path": "Path to an Apptainer/Singularity `.sif` image to run SLURM workers inside (optional).",
    "max_concurrent_downscale_layers": "How many pyramid levels are downscaled in parallel per file (default 1 = sequential, lowest memory).",
    # ConversionConfig
    "ome_zarr_version": "OME-Zarr (NGFF) version to write: `0.4` (backed by Zarr v2) or `0.5` (backed by Zarr v3, which enables sharding). This is the preferred control and supersedes `zarr_format` — the underlying zarr container format is derived from it.",
    "zarr_format": "**Deprecated** — use `ome_zarr_version` instead. Zarr container version: `2` = OME-Zarr 0.4 (classic chunk-based); `3` = OME-Zarr 0.5 (sharding support).",
    "skip_dask": "Read TIFF files via zarr's native aszarr backend instead of dask. Faster for large TIFFs; ignored for non-TIFF formats.",
    "auto_chunk": "Auto-compute chunk shape to approximate `target_chunk_mb`. When `False`, the manual `*_chunk` values below are used.",
    "target_chunk_mb": "Target uncompressed chunk size in MB when `auto_chunk=True`.",
    "time_chunk": "Chunk size along the time axis. Applies only when `auto_chunk=False`.",
    "channel_chunk": "Chunk size along the channel axis. Applies only when `auto_chunk=False`.",
    "z_chunk": "Chunk size along the z axis. Applies only when `auto_chunk=False`.",
    "y_chunk": "Chunk size along the y axis. Applies only when `auto_chunk=False`.",
    "x_chunk": "Chunk size along the x axis. Applies only when `auto_chunk=False`.",
    "time_shard_coef": "Shard size = chunk × coef for the time axis. Zarr v3 only — ignored for v2.",
    "channel_shard_coef": "Shard size = chunk × coef for the channel axis. Zarr v3 only — ignored for v2.",
    "z_shard_coef": "Shard size = chunk × coef for the z axis. Zarr v3 only — ignored for v2.",
    "y_shard_coef": "Shard size = chunk × coef for the y axis. Zarr v3 only — ignored for v2.",
    "x_shard_coef": "Shard size = chunk × coef for the x axis. Zarr v3 only — ignored for v2.",
    "time_range": 'Crop the time axis before writing. Pass as a `"start,stop"` string (e.g. `"0,10"` to keep frames 0–9).',
    "channel_range": 'Crop the channel axis. Same format as `time_range`, e.g. `"0,2"` keeps the first two channels.',
    "z_range": 'Crop the z axis. Same format as `time_range`, e.g. `"5,50"` keeps z-slices 5–49.',
    "y_range": 'Crop the y axis. Same format as `time_range`.',
    "x_range": 'Crop the x axis. Same format as `time_range`.',
    "compressor": "Compression codec.",
    "compressor_params": "Codec-specific parameters dict. When omitted, sensible defaults are used (e.g. for blosc: `{'cname': 'lz4', 'clevel': 5, 'shuffle': 1}`).",
    "dimension_order": "Axis order string for the output array (default `tczyx`).",
    "overwrite": "Overwrite existing output zarr if it already exists.",
    "override_channel_names": "Replace output channel labels with the `channel_tag` values. For aggregative conversions with a tuple `channel_tag` only.",
    "channel_intensity_limits": "Strategy for OMERO display-window limits.",
    "metadata_reader": "Backend used to read OME-XML pixel metadata.",
    "save_omexml": "Write a companion OME-XML sidecar file alongside the zarr.",
    "squeeze": "Remove length-1 dimensions before writing.",
    "dtype": "`auto` preserves the source dtype; pass any NumPy dtype string (e.g. `uint16`, `float32`) to cast on write.",
    "verbose": "Log verbose per-chunk progress output.",
    # DownscaleConfig
    "time_scale_factor": "Downscale factor for the time axis per pyramid level.",
    "channel_scale_factor": "Downscale factor for the channel axis.",
    "z_scale_factor": "Downscale factor for the z axis.",
    "y_scale_factor": "Downscale factor for the y axis.",
    "x_scale_factor": "Downscale factor for the x axis.",
    "n_layers": "Number of pyramid levels (None = auto).",
    "min_dimension_size": "Stop downscaling when smallest dimension reaches this size.",
    "downscale_method": "`simple` = stride/nearest (fastest). `mean` / `median` / `min` / `max` / `mode` = aggregation methods passed to TensorStore.",
    "keep_existing_resolutions": "If the input already carries its own multiscale pyramid (e.g. `.ims`, `.zarr`), write each existing resolution level straight to the output OME-Zarr instead of rebuilding the pyramid (default False).",
    "apply_smart_downscaling": "Choose per-axis downscale factors automatically from the pixel anisotropy so the pyramid approaches isotropy, instead of using the fixed `*_scale_factor` values (default False).",
    "time_smart_scale_factor": "Override the smart-downscaling factor for the time axis (used when `apply_smart_downscaling=True`).",
    "z_smart_scale_factor": "Override the smart-downscaling factor for the z axis (used when `apply_smart_downscaling=True`).",
    "y_smart_scale_factor": "Override the smart-downscaling factor for the y axis (used when `apply_smart_downscaling=True`).",
    "x_smart_scale_factor": "Override the smart-downscaling factor for the x axis (used when `apply_smart_downscaling=True`).",
    # ReaderConfig
    "scene_index": (
        "Scene / series index to read. Pass an integer for a single scene, "
        "`all` to convert each scene to a separate zarr group, "
        "or comma-separated integers (e.g. `0,2,4`) for a subset of scenes."
    ),
    "mosaic_tile_index": (
        "Mosaic tile(s) to read when **not** stitching. Pass an integer, `all`, or "
        "comma-separated integers; each selected tile becomes a separate OME-Zarr "
        "(named `_tile{N}`). Use `--as_mosaic` instead to stitch tiles into one output. "
        "Composes with scene / view / illumination selection (cartesian product)."
    ),
    "as_mosaic": "Stitch all mosaic tiles into a single full field-of-view output (instead of one OME-Zarr per tile).",
    "view_index": (
        "View(s) to read. Pass an integer, `all`, or comma-separated integers; each "
        "selected view becomes a separate OME-Zarr (named `_view{N}`) unless "
        "`--concat_views` stacks them along the channel axis."
    ),
    "phase_index": "Phase index.",
    "illumination_index": (
        "Illumination(s) to read. Pass an integer, `all`, or comma-separated integers; "
        "each selected illumination becomes a separate OME-Zarr (named `_illu{N}`) unless "
        "`--concat_illuminations` stacks them along the channel axis."
    ),
    "rotation_index": "Rotation index.",
    "sample_index": "Sample index.",
    "force_bioformats": "Force the Java Bio-Formats reader even for formats EuBI-Bridge reads natively (CZI, ND2, LIF, IMS…). Useful as a fallback when a native read fails.",
    "concat_views": "When reading multiple views, stack them along the channel axis into one output (cartesian product with existing channels) instead of writing one OME-Zarr per view.",
    "concat_illuminations": "When reading multiple illuminations, stack them along the channel axis into one output (cartesian product with existing channels) instead of writing one OME-Zarr per illumination.",
    # ConcatenationConfig
    "concatenation_axes": (
        "Axes along which files are concatenated. Pass a string of axis letters "
        "(`t`, `c`, `z`, `y`, `x`), e.g. `c` for channel, `zc` for z + channel. "
        "Each axis listed requires a corresponding `*_tag` argument."
    ),
    "time_tag": (
        "Substring(s) in filenames that identify the time axis. "
        "Pass a single string to match one file per time point, "
        "or comma-separated strings to assign labels in order, e.g. `t001,t002,t003`."
    ),
    "channel_tag": (
        "Substring(s) in filenames that identify the channel axis. "
        "Pass a single string or comma-separated strings, e.g. `raw,mask`."
    ),
    "z_tag": (
        "Substring(s) in filenames that identify the z axis. "
        "Pass a single string or comma-separated strings."
    ),
    "y_tag": "Substring(s) in filenames that identify the y axis.",
    "x_tag": "Substring(s) in filenames that identify the x axis.",
}

# ---------------------------------------------------------------------------
# Examples — base dict (to_zarr context) + per-command overrides
# ---------------------------------------------------------------------------

# Base examples used when a parameter appears inside to_zarr override sections.
# These all show "eubi to_zarr ..." which is correct in that context.
_EXAMPLES: dict[str, str | list[str]] = {
    # ── to_zarr explicit params ───────────────────────────────────────────────
    "input_path": "eubi to_zarr /data/input /data/output",
    "output_path": "eubi to_zarr /data/input /data/output",
    "includes": [
        '# Include only files whose name contains "EMPIAR"',
        'eubi to_zarr /data/input /data/output --includes "EMPIAR"',
        "",
        "# Multiple patterns — comma-separated, no spaces around commas",
        'eubi to_zarr /data/input /data/output --includes "EMPIAR,scan_001,confocal"',
    ],
    "excludes": [
        "# Exclude all files in a subdirectory named 'backup'",
        'eubi to_zarr /data/input /data/output --excludes "backup"',
        "",
        "# Exclude multiple patterns at once",
        'eubi to_zarr /data/input /data/output --excludes "backup,temp,preview"',
    ],
    "plan": [
        "# Step 1 — validate without converting to inspect the plan",
        "eubi validate_aggregative /data/input /data/output --concatenation_axes c --channel_tag raw,mask",
        "",
        "# Step 2 — pass the plan object to skip re-planning on the actual run",
        "eubi to_zarr /data/input /data/output --concatenation_axes c --channel_tag raw,mask --plan <plan>",
    ],
    "output_file": "eubi show_pixel_meta /data/input --output_file report.html",
    "name": "eubi show_config --name hpc",
    "create": "eubi update_config hpc --create",
    "series": "eubi show_pixel_meta /data/input --series 2",
    # ── ClusterConfig (to_zarr context) ──────────────────────────────────────
    "max_workers": "eubi to_zarr /data/input /data/output --max_workers 8",
    "max_retries": "eubi to_zarr /data/input /data/output --max_retries 3",
    "queue_size": "eubi to_zarr /data/input /data/output --queue_size 8",
    "region_size_mb": "eubi to_zarr /data/input /data/output --region_size_mb 512",
    "max_concurrency": "eubi to_zarr /data/input /data/output --max_concurrency 8",
    "max_concurrent_scenes": "eubi to_zarr /data/input /data/output --max_concurrent_scenes 2",
    "use_threading": "eubi to_zarr /data/input /data/output --use_threading",
    "on_slurm": [
        "# Submit all file workers to SLURM",
        "eubi to_zarr /data/input /data/output --on_slurm --slurm_account myaccount --slurm_partition gpu",
    ],
    "on_local_cluster": "eubi to_zarr /data/input /data/output --on_local_cluster",
    "memory_per_worker": "eubi to_zarr /data/input /data/output --memory_per_worker 8GB",
    "tensorstore_data_copy_concurrency": "eubi to_zarr /data/input /data/output --tensorstore_data_copy_concurrency 8",
    "slurm_time": "eubi to_zarr /data/input /data/output --on_slurm --slurm_time 48:00:00",
    "slurm_account": "eubi to_zarr /data/input /data/output --on_slurm --slurm_account myaccount",
    "slurm_partition": "eubi to_zarr /data/input /data/output --on_slurm --slurm_partition gpu",
    "slurm_worker_timeout": "eubi to_zarr /data/input /data/output --on_slurm --slurm_worker_timeout 600",
    "slurm_sif_path": "eubi to_zarr /data/input /data/output --on_slurm --slurm_sif_path /apps/eubi.sif",
    "max_concurrent_downscale_layers": "eubi to_zarr /data/input /data/output --max_concurrent_downscale_layers 2",
    # ── ConversionConfig (to_zarr context) ───────────────────────────────────
    "ome_zarr_version": [
        "# Write OME-Zarr 0.4 (Zarr v2, the default)",
        "eubi to_zarr /data/input /data/output --ome_zarr_version 0.4",
        "",
        "# Write OME-Zarr 0.5 (Zarr v3 — required for sharding)",
        "eubi to_zarr /data/input /data/output --ome_zarr_version 0.5",
    ],
    "zarr_format": "eubi to_zarr /data/input /data/output --zarr_format 3",
    "skip_dask": "eubi to_zarr /data/input /data/output --skip_dask",
    "auto_chunk": "eubi to_zarr /data/input /data/output --auto_chunk False --z_chunk 64 --y_chunk 256 --x_chunk 256",
    "target_chunk_mb": "eubi to_zarr /data/input /data/output --target_chunk_mb 4.0",
    "time_chunk": "eubi to_zarr /data/input /data/output --auto_chunk False --time_chunk 1",
    "channel_chunk": "eubi to_zarr /data/input /data/output --auto_chunk False --channel_chunk 1",
    "z_chunk": "eubi to_zarr /data/input /data/output --auto_chunk False --z_chunk 64",
    "y_chunk": "eubi to_zarr /data/input /data/output --auto_chunk False --y_chunk 256",
    "x_chunk": "eubi to_zarr /data/input /data/output --auto_chunk False --x_chunk 256",
    "time_shard_coef": "eubi to_zarr /data/input /data/output --ome_zarr_version 0.5 --time_shard_coef 1",
    "channel_shard_coef": "eubi to_zarr /data/input /data/output --ome_zarr_version 0.5 --channel_shard_coef 1",
    "z_shard_coef": "eubi to_zarr /data/input /data/output --ome_zarr_version 0.5 --z_shard_coef 5",
    "y_shard_coef": "eubi to_zarr /data/input /data/output --ome_zarr_version 0.5 --y_shard_coef 5",
    "x_shard_coef": "eubi to_zarr /data/input /data/output --ome_zarr_version 0.5 --x_shard_coef 5",
    "time_range": 'eubi to_zarr /data/input /data/output --time_range "0,10"',
    "channel_range": 'eubi to_zarr /data/input /data/output --channel_range "0,2"',
    "z_range": 'eubi to_zarr /data/input /data/output --z_range "5,50"',
    "y_range": 'eubi to_zarr /data/input /data/output --y_range "0,512"',
    "x_range": 'eubi to_zarr /data/input /data/output --x_range "0,512"',
    "compressor": "eubi to_zarr /data/input /data/output --compressor zstd",
    "compressor_params": 'eubi to_zarr /data/input /data/output --compressor blosc --compressor_params \'{"cname": "lz4", "clevel": 9}\'',
    "overwrite": "eubi to_zarr /data/input /data/output --overwrite",
    "dtype": "eubi to_zarr /data/input /data/output --dtype uint8",
    "squeeze": "eubi to_zarr /data/input /data/output --squeeze False",
    "channel_intensity_limits": "eubi to_zarr /data/input /data/output --channel_intensity_limits from_array",
    "metadata_reader": "eubi to_zarr /data/input /data/output --metadata_reader bioformats",
    "save_omexml": "eubi to_zarr /data/input /data/output --save_omexml False",
    "override_channel_names": "eubi to_zarr /data/input /data/output --concatenation_axes c --channel_tag raw,mask --override_channel_names",
    "verbose": "eubi to_zarr /data/input /data/output --verbose",
    # ── DownscaleConfig (to_zarr context) ────────────────────────────────────
    "n_layers": "eubi to_zarr /data/input /data/output --n_layers 5",
    "min_dimension_size": "eubi to_zarr /data/input /data/output --min_dimension_size 32",
    "downscale_method": [
        "# Use mean aggregation (recommended for fluorescence images)",
        "eubi to_zarr /data/input /data/output --downscale_method mean",
        "",
        "# Use nearest-neighbour striding (fastest, recommended for label images)",
        "eubi to_zarr /data/input /data/output --downscale_method simple",
    ],
    "z_scale_factor": [
        "# Isotropic downscaling on all spatial axes",
        "eubi to_zarr /data/input /data/output --z_scale_factor 2 --y_scale_factor 2 --x_scale_factor 2",
        "",
        "# Anisotropic — skip z downscaling for thick sections",
        "eubi to_zarr /data/input /data/output --z_scale_factor 1 --y_scale_factor 2 --x_scale_factor 2",
    ],
    "y_scale_factor": "eubi to_zarr /data/input /data/output --y_scale_factor 2",
    "x_scale_factor": "eubi to_zarr /data/input /data/output --x_scale_factor 2",
    "time_scale_factor": "eubi to_zarr /data/input /data/output --time_scale_factor 1",
    "channel_scale_factor": "eubi to_zarr /data/input /data/output --channel_scale_factor 1",
    "keep_existing_resolutions": [
        "# Copy an .ims / .zarr input's own pyramid levels instead of rebuilding them",
        "eubi to_zarr /data/input /data/output --keep_existing_resolutions",
    ],
    "apply_smart_downscaling": [
        "# Let EuBI-Bridge pick per-axis factors from the voxel anisotropy",
        "eubi to_zarr /data/input /data/output --apply_smart_downscaling",
    ],
    "z_smart_scale_factor": "eubi to_zarr /data/input /data/output --apply_smart_downscaling --z_smart_scale_factor 1",
    "y_smart_scale_factor": "eubi to_zarr /data/input /data/output --apply_smart_downscaling --y_smart_scale_factor 2",
    "x_smart_scale_factor": "eubi to_zarr /data/input /data/output --apply_smart_downscaling --x_smart_scale_factor 2",
    "time_smart_scale_factor": "eubi to_zarr /data/input /data/output --apply_smart_downscaling --time_smart_scale_factor 1",
    # ── ReaderConfig (to_zarr context) ───────────────────────────────────────
    "scene_index": [
        "# Convert only scene 2 from a multi-scene file",
        "eubi to_zarr /data/input /data/output --scene_index 2",
        "",
        "# Convert every scene to a separate OME-Zarr group",
        "eubi to_zarr /data/input /data/output --scene_index all",
        "",
        "# Convert scenes 0, 2 and 4 only",
        "eubi to_zarr /data/input /data/output --scene_index 0,2,4",
    ],
    "as_mosaic": [
        "# Stitch all mosaic tiles into a single full field-of-view output",
        "eubi to_zarr /data/input /data/output --as_mosaic",
    ],
    "mosaic_tile_index": [
        "# Write every tile as its own OME-Zarr (default — each tile separate)",
        "eubi to_zarr /data/input /data/output --mosaic_tile_index all",
        "",
        "# Write only tiles 0 and 2 (each as a separate output)",
        "eubi to_zarr /data/input /data/output --mosaic_tile_index 0,2",
        "",
        "# Stitch all tiles into one mosaic instead of writing them separately",
        "eubi to_zarr /data/input /data/output --as_mosaic",
    ],
    "view_index": [
        "# Write each view as its own OME-Zarr",
        "eubi to_zarr /data/input /data/output --view_index all",
        "",
        "# Concatenate all views along the channel axis into one output",
        "eubi to_zarr /data/input /data/output --view_index all --concat_views",
    ],
    "concat_views": [
        "# Stack every view onto the channel axis (cartesian with existing channels)",
        "eubi to_zarr /data/input /data/output --view_index all --concat_views",
    ],
    "phase_index": "eubi to_zarr /data/input /data/output --phase_index 1",
    "illumination_index": [
        "# Write each illumination as its own OME-Zarr",
        "eubi to_zarr /data/input /data/output --illumination_index all",
        "",
        "# Concatenate all illuminations along the channel axis into one output",
        "eubi to_zarr /data/input /data/output --illumination_index all --concat_illuminations",
    ],
    "concat_illuminations": [
        "# Stack every illumination onto the channel axis (cartesian with existing channels)",
        "eubi to_zarr /data/input /data/output --illumination_index all --concat_illuminations",
    ],
    "rotation_index": "eubi to_zarr /data/input /data/output --rotation_index 0",
    "sample_index": "eubi to_zarr /data/input /data/output --sample_index 0",
    "force_bioformats": "eubi to_zarr /data/input /data/output --force_bioformats",
    # ── ConcatenationConfig (to_zarr context) ────────────────────────────────
    "concatenation_axes": [
        "# Concatenate files along the channel axis",
        "eubi to_zarr /data/input /data/output --concatenation_axes c --channel_tag raw,mask",
        "",
        "# Concatenate along both z and channel axes simultaneously",
        "eubi to_zarr /data/input /data/output --concatenation_axes zc --z_tag slices --channel_tag raw,mask",
    ],
    "channel_tag": [
        "# Two channels — raw fluorescence and segmentation mask",
        "eubi to_zarr /data/input /data/output --concatenation_axes c --channel_tag raw,mask",
        "",
        "# Single channel — assign a label without concatenating",
        "eubi to_zarr /data/input /data/output --concatenation_axes c --channel_tag DAPI",
    ],
    "time_tag": [
        "# Three time points identified by substrings in each filename",
        "eubi to_zarr /data/input /data/output --concatenation_axes t --time_tag t001,t002,t003",
    ],
    "z_tag": [
        "# Multiple z-slices acquired as separate files",
        "eubi to_zarr /data/input /data/output --concatenation_axes z --z_tag slice001,slice002,slice003",
    ],
    "y_tag": "eubi to_zarr /data/input /data/output --concatenation_axes y --y_tag row_top,row_bottom",
    "x_tag": "eubi to_zarr /data/input /data/output --concatenation_axes x --x_tag col_left,col_right",
}

# Per-command example overrides.  Checked before _EXAMPLES so commands get
# context-correct examples instead of the to_zarr defaults above.
_CMD_EXAMPLES: dict[str, dict[str, str | list[str]]] = {

    # ── show_pixel_meta ───────────────────────────────────────────────────────
    "show_pixel_meta": {
        "input_path": "eubi show_pixel_meta /data/input",
        "includes": [
            '# Show metadata only for files whose name contains "EMPIAR"',
            'eubi show_pixel_meta /data/input --includes "EMPIAR"',
            "",
            "# Multiple patterns — comma-separated",
            'eubi show_pixel_meta /data/input --includes "EMPIAR,scan_001,confocal"',
        ],
        "excludes": [
            "# Skip preview files",
            'eubi show_pixel_meta /data/input --excludes "preview"',
            "",
            "# Skip multiple patterns",
            'eubi show_pixel_meta /data/input --excludes "preview,temp"',
        ],
        "series": "eubi show_pixel_meta /data/input --series 2",
        "output_file": [
            "# Save as an HTML report you can open in a browser",
            "eubi show_pixel_meta /data/input --output_file report.html",
            "",
            "# Save as plain text",
            "eubi show_pixel_meta /data/input --output_file report.txt",
        ],
    },

    # ── update_pixel_meta ─────────────────────────────────────────────────────
    "update_pixel_meta": {
        "input_path": "eubi update_pixel_meta /data/output",
        "includes": [
            '# Update only files whose name contains "EMPIAR"',
            'eubi update_pixel_meta /data/output --includes "EMPIAR"',
        ],
        "excludes": [
            "# Skip files whose name contains 'backup'",
            'eubi update_pixel_meta /data/output --excludes "backup"',
        ],
        "time_scale": "eubi update_pixel_meta /data/output --time_scale 1.0 --time_unit second",
        "z_scale": [
            "# Set z-step size and unit together",
            "eubi update_pixel_meta /data/output --z_scale 0.5 --z_unit micrometer",
        ],
        "y_scale": "eubi update_pixel_meta /data/output --y_scale 0.25 --y_unit micrometer",
        "x_scale": "eubi update_pixel_meta /data/output --x_scale 0.25 --x_unit micrometer",
        "time_unit": "eubi update_pixel_meta /data/output --time_scale 1.0 --time_unit second",
        "z_unit": "eubi update_pixel_meta /data/output --z_scale 0.5 --z_unit micrometer",
        "y_unit": "eubi update_pixel_meta /data/output --y_scale 0.25 --y_unit micrometer",
        "x_unit": "eubi update_pixel_meta /data/output --x_scale 0.25 --x_unit micrometer",
        "max_workers": [
            "# Update 8 OME-Zarr stores in parallel",
            "eubi update_pixel_meta /data/output --max_workers 8",
        ],
    },

    # ── update_channel_meta ───────────────────────────────────────────────────
    "update_channel_meta": {
        "input_path": "eubi update_channel_meta /data/output",
        "includes": [
            '# Update only files whose name contains "EMPIAR"',
            'eubi update_channel_meta /data/output --includes "EMPIAR"',
        ],
        "excludes": [
            "# Skip files whose name contains 'backup'",
            'eubi update_channel_meta /data/output --excludes "backup"',
        ],
        "channel_labels": [
            "# Rename channels 0 and 1",
            'eubi update_channel_meta /data/output --channel_labels "0,DAPI;1,GFP"',
            "",
            "# Three channels",
            'eubi update_channel_meta /data/output --channel_labels "0,DAPI;1,GFP;2,mCherry"',
        ],
        "channel_colors": [
            "# Set display colours in hex (blue DAPI, green GFP)",
            'eubi update_channel_meta /data/output --channel_colors "0,0000FF;1,00FF00"',
            "",
            "# Three channels — blue, green, red",
            'eubi update_channel_meta /data/output --channel_colors "0,0000FF;1,00FF00;2,FF0000"',
        ],
        "channel_intensity_limits": [
            "# Compute per-channel min/max from the pixel data",
            "eubi update_channel_meta /data/output --channel_intensity_limits from_array",
            "",
            "# Let the viewer decide (OMERO-style auto-scaling)",
            "eubi update_channel_meta /data/output --channel_intensity_limits auto",
        ],
        "max_workers": [
            "# Update 8 OME-Zarr stores in parallel",
            "eubi update_channel_meta /data/output --max_workers 8",
        ],
    },

    # ── validate_aggregative ──────────────────────────────────────────────────
    "validate_aggregative": {
        "input_path": "eubi validate_aggregative /data/input /data/output --concatenation_axes c --channel_tag raw,mask",
        "output_path": "eubi validate_aggregative /data/input /data/output --concatenation_axes c --channel_tag raw,mask",
        "includes": [
            '# Plan the conversion but only include files matching "scan"',
            'eubi validate_aggregative /data/input /data/output --concatenation_axes c --channel_tag raw,mask --includes "scan"',
        ],
        "excludes": [
            "# Exclude preview files from the plan",
            'eubi validate_aggregative /data/input /data/output --concatenation_axes c --channel_tag raw,mask --excludes "preview"',
        ],
        "concatenation_axes": [
            "# Plan a channel concatenation",
            "eubi validate_aggregative /data/input /data/output --concatenation_axes c --channel_tag raw,mask",
            "",
            "# Plan a z + channel concatenation",
            "eubi validate_aggregative /data/input /data/output --concatenation_axes zc --z_tag slices --channel_tag raw,mask",
        ],
        "time_tag": [
            "# Plan concatenation of three time points",
            "eubi validate_aggregative /data/input /data/output --concatenation_axes t --time_tag t001,t002,t003",
        ],
        "channel_tag": [
            "# Two channels — raw fluorescence and segmentation mask",
            "eubi validate_aggregative /data/input /data/output --concatenation_axes c --channel_tag raw,mask",
        ],
        "z_tag": [
            "# Multiple z-slices acquired as separate files",
            "eubi validate_aggregative /data/input /data/output --concatenation_axes z --z_tag slice001,slice002,slice003",
        ],
        "y_tag": "eubi validate_aggregative /data/input /data/output --concatenation_axes y --y_tag row_top,row_bottom",
        "x_tag": "eubi validate_aggregative /data/input /data/output --concatenation_axes x --x_tag col_left,col_right",
    },

    # ── configure_cluster ─────────────────────────────────────────────────────
    "configure.cluster": {
        "max_workers": [
            "# Set the default number of parallel worker processes",
            "eubi configure cluster --max_workers 8",
        ],
        "queue_size": [
            "# Allow each worker to buffer more write tasks",
            "eubi configure cluster --queue_size 8",
        ],
        "region_size_mb": [
            "# Process larger spatial regions per task (reduces overhead on large arrays)",
            "eubi configure cluster --region_size_mb 512",
        ],
        "memory_per_worker": [
            "# Allocate 8 GB per worker (relevant for SLURM / LocalCluster)",
            "eubi configure cluster --memory_per_worker 8GB",
        ],
        "max_concurrency": [
            "# Allow TensorStore to issue 8 write operations concurrently per worker",
            "eubi configure cluster --max_concurrency 8",
        ],
        "max_concurrent_scenes": [
            "# Convert 2 scenes within a single file simultaneously",
            "eubi configure cluster --max_concurrent_scenes 2",
        ],
        "on_local_cluster": [
            "# Use a Dask LocalCluster as the scheduler",
            "eubi configure cluster --on_local_cluster",
        ],
        "on_slurm": [
            "# Submit workers to SLURM and set partition + account",
            "eubi configure cluster --on_slurm --slurm_account myaccount --slurm_partition gpu",
        ],
        "use_threading": [
            "# Switch from ProcessPool to ThreadPool (useful when GIL is not a bottleneck)",
            "eubi configure cluster --use_threading",
        ],
        "tensorstore_data_copy_concurrency": [
            "# Allow TensorStore to copy data with 8 internal threads",
            "eubi configure cluster --tensorstore_data_copy_concurrency 8",
        ],
        "max_retries": [
            "# Retry failed workers up to 5 times before aborting",
            "eubi configure cluster --max_retries 5",
        ],
        "slurm_time": [
            "# Set SLURM wall-clock limit to 48 hours",
            "eubi configure cluster --on_slurm --slurm_time 48:00:00",
        ],
        "slurm_account": [
            "# Charge SLURM jobs to 'myproject'",
            "eubi configure cluster --on_slurm --slurm_account myproject",
        ],
        "slurm_partition": [
            "# Submit to the 'gpu' partition",
            "eubi configure cluster --on_slurm --slurm_partition gpu",
        ],
        "slurm_worker_timeout": [
            "# Wait up to 10 minutes for SLURM workers to start",
            "eubi configure cluster --on_slurm --slurm_worker_timeout 600",
        ],
    },

    # ── configure_conversion ──────────────────────────────────────────────────
    "configure.conversion": {
        "ome_zarr_version": [
            "# Default to OME-Zarr 0.5 (Zarr v3, enables sharding)",
            "eubi configure conversion --ome_zarr_version 0.5",
            "",
            "# Default to OME-Zarr 0.4 (Zarr v2)",
            "eubi configure conversion --ome_zarr_version 0.4",
        ],
        "zarr_format": [
            "# Deprecated — prefer 'ome_zarr_version'. Switch to Zarr v3 (supports sharding)",
            "eubi configure conversion --zarr_format 3",
        ],
        "skip_dask": [
            "# Use zarr's native TIFF backend instead of dask (faster for large TIFFs)",
            "eubi configure conversion --skip_dask",
        ],
        "auto_chunk": [
            "# Disable auto-chunking and set chunk sizes manually",
            "eubi configure conversion --auto_chunk False --z_chunk 64 --y_chunk 256 --x_chunk 256",
        ],
        "target_chunk_mb": [
            "# Target 4 MB uncompressed chunks when auto_chunk is on",
            "eubi configure conversion --target_chunk_mb 4.0",
        ],
        "time_chunk": [
            "# Process one time point per chunk",
            "eubi configure conversion --auto_chunk False --time_chunk 1",
        ],
        "channel_chunk": [
            "# Process one channel per chunk",
            "eubi configure conversion --auto_chunk False --channel_chunk 1",
        ],
        "z_chunk": [
            "# Set manual z chunk size (requires auto_chunk False)",
            "eubi configure conversion --auto_chunk False --z_chunk 64",
        ],
        "y_chunk": [
            "# Set manual y chunk size (requires auto_chunk False)",
            "eubi configure conversion --auto_chunk False --y_chunk 256",
        ],
        "x_chunk": [
            "# Set manual x chunk size (requires auto_chunk False)",
            "eubi configure conversion --auto_chunk False --x_chunk 256",
        ],
        "time_shard_coef": [
            "# Shard = 1 × time chunk (no sharding on time axis)",
            "eubi configure conversion --ome_zarr_version 0.5 --time_shard_coef 1",
        ],
        "channel_shard_coef": [
            "# Shard = 1 × channel chunk",
            "eubi configure conversion --ome_zarr_version 0.5 --channel_shard_coef 1",
        ],
        "z_shard_coef": [
            "# Shard = 5 × z chunk on the z axis",
            "eubi configure conversion --ome_zarr_version 0.5 --z_shard_coef 5",
        ],
        "y_shard_coef": [
            "# Shard = 5 × y chunk on the y axis",
            "eubi configure conversion --ome_zarr_version 0.5 --y_shard_coef 5",
        ],
        "x_shard_coef": [
            "# Shard = 5 × x chunk on the x axis",
            "eubi configure conversion --ome_zarr_version 0.5 --x_shard_coef 5",
        ],
        "time_range": [
            '# Keep only time frames 0–9 (start inclusive, stop exclusive)',
            'eubi configure conversion --time_range "0,10"',
        ],
        "channel_range": [
            "# Keep only the first two channels",
            'eubi configure conversion --channel_range "0,2"',
        ],
        "z_range": [
            "# Keep z-slices 5 through 49",
            'eubi configure conversion --z_range "5,50"',
        ],
        "y_range": [
            "# Crop y to the first 512 pixels",
            'eubi configure conversion --y_range "0,512"',
        ],
        "x_range": [
            "# Crop x to the first 512 pixels",
            'eubi configure conversion --x_range "0,512"',
        ],
        "compressor": [
            "# Use zstd compression (good balance of speed and ratio)",
            "eubi configure conversion --compressor zstd",
            "",
            "# Disable compression entirely",
            "eubi configure conversion --compressor none",
        ],
        "compressor_params": [
            "# Use blosc with lz4 codec at compression level 9",
            'eubi configure conversion --compressor blosc --compressor_params \'{"cname": "lz4", "clevel": 9}\'',
        ],
        "overwrite": [
            "# Allow overwriting an existing OME-Zarr at the output path",
            "eubi configure conversion --overwrite",
        ],
        "override_channel_names": [
            "# Replace channel labels with the channel_tag values from filenames",
            "eubi configure conversion --override_channel_names",
        ],
        "channel_intensity_limits": [
            "# Compute display window limits from actual pixel data",
            "eubi configure conversion --channel_intensity_limits from_array",
            "",
            "# Let the viewer compute limits automatically",
            "eubi configure conversion --channel_intensity_limits auto",
        ],
        "metadata_reader": [
            "# Use the Java BioFormats backend for metadata (more formats supported)",
            "eubi configure conversion --metadata_reader bioformats",
        ],
        "save_omexml": [
            "# Disable OME-XML sidecar file generation",
            "eubi configure conversion --save_omexml False",
        ],
        "squeeze": [
            "# Keep length-1 dimensions instead of removing them",
            "eubi configure conversion --squeeze False",
        ],
        "dtype": [
            "# Cast pixel values to uint8 on write (reduces file size)",
            "eubi configure conversion --dtype uint8",
            "",
            "# Keep the source dtype",
            "eubi configure conversion --dtype auto",
        ],
        "verbose": [
            "# Enable verbose per-chunk progress logging",
            "eubi configure conversion --verbose",
        ],
    },

    # ── configure_downscale ───────────────────────────────────────────────────
    "configure.downscale": {
        "n_layers": [
            "# Generate 5 pyramid levels",
            "eubi configure downscale --n_layers 5",
        ],
        "min_dimension_size": [
            "# Stop downscaling when the smallest spatial dimension is below 32 px",
            "eubi configure downscale --min_dimension_size 32",
        ],
        "downscale_method": [
            "# Use mean aggregation (recommended for fluorescence images)",
            "eubi configure downscale --downscale_method mean",
            "",
            "# Use nearest-neighbour striding (fastest, recommended for label images)",
            "eubi configure downscale --downscale_method simple",
        ],
        "z_scale_factor": [
            "# Isotropic downscaling on all spatial axes",
            "eubi configure downscale --z_scale_factor 2 --y_scale_factor 2 --x_scale_factor 2",
            "",
            "# Anisotropic — skip z downscaling for thick sections",
            "eubi configure downscale --z_scale_factor 1 --y_scale_factor 2 --x_scale_factor 2",
        ],
        "y_scale_factor": "eubi configure downscale --y_scale_factor 2",
        "x_scale_factor": "eubi configure downscale --x_scale_factor 2",
        "time_scale_factor": [
            "# Disable time-axis downscaling (value 1 = no downscaling)",
            "eubi configure downscale --time_scale_factor 1",
        ],
        "channel_scale_factor": [
            "# Disable channel-axis downscaling",
            "eubi configure downscale --channel_scale_factor 1",
        ],
        "keep_existing_resolutions": [
            "# Copy a source pyramid (.ims / .zarr) verbatim instead of rebuilding it",
            "eubi configure downscale --keep_existing_resolutions",
        ],
        "apply_smart_downscaling": [
            "# Pick per-axis downscale factors automatically from voxel anisotropy",
            "eubi configure downscale --apply_smart_downscaling",
        ],
    },

    # ── configure_readers ─────────────────────────────────────────────────────
    "configure.readers": {
        "as_mosaic": [
            "# Stitch tiled acquisitions into a single mosaic array on read",
            "eubi configure readers --as_mosaic",
        ],
        "scene_index": [
            "# Always read scene 2 from multi-scene files",
            "eubi configure readers --scene_index 2",
            "",
            "# Convert each scene to a separate OME-Zarr group",
            "eubi configure readers --scene_index all",
            "",
            "# Convert only scenes 0, 2 and 4",
            "eubi configure readers --scene_index 0,2,4",
        ],
        "mosaic_tile_index": [
            "# Write every tile separately (default)",
            "eubi configure readers --mosaic_tile_index all",
            "",
            "# Write only tiles 0 and 2",
            "eubi configure readers --mosaic_tile_index 0,2",
        ],
        "view_index": [
            "# Write each view separately",
            "eubi configure readers --view_index all",
            "",
            "# Select only the second view",
            "eubi configure readers --view_index 1",
        ],
        "phase_index": [
            "# Select phase index 1",
            "eubi configure readers --phase_index 1",
        ],
        "illumination_index": [
            "# Write each illumination separately",
            "eubi configure readers --illumination_index all",
            "",
            "# Select only the second illumination",
            "eubi configure readers --illumination_index 1",
        ],
        "concat_views": [
            "# Always stack views onto the channel axis",
            "eubi configure readers --concat_views",
        ],
        "concat_illuminations": [
            "# Always stack illuminations onto the channel axis",
            "eubi configure readers --concat_illuminations",
        ],
        "force_bioformats": [
            "# Always route reads through the Java Bio-Formats reader",
            "eubi configure readers --force_bioformats",
        ],
        "rotation_index": [
            "# Select rotation index 0",
            "eubi configure readers --rotation_index 0",
        ],
        "sample_index": [
            "# Select sample index 0",
            "eubi configure readers --sample_index 0",
        ],
    },

    # ── configure_concatenation ───────────────────────────────────────────────
    "configure.concatenation": {
        "concatenation_axes": [
            "# Concatenate files along the channel axis",
            "eubi configure concatenation --concatenation_axes c --channel_tag raw,mask",
            "",
            "# Concatenate along both z and channel axes simultaneously",
            "eubi configure concatenation --concatenation_axes zc --z_tag slices --channel_tag raw,mask",
        ],
        "time_tag": [
            "# Identify three time-point files by substrings in their names",
            "eubi configure concatenation --concatenation_axes t --time_tag t001,t002,t003",
        ],
        "channel_tag": [
            "# Two channels — raw fluorescence and segmentation mask",
            "eubi configure concatenation --concatenation_axes c --channel_tag raw,mask",
            "",
            "# Single channel — DAPI only",
            "eubi configure concatenation --concatenation_axes c --channel_tag DAPI",
        ],
        "z_tag": [
            "# Multiple z-slices acquired as separate files",
            "eubi configure concatenation --concatenation_axes z --z_tag slice001,slice002,slice003",
        ],
        "y_tag": [
            "# Two rows of a tile scan",
            "eubi configure concatenation --concatenation_axes y --y_tag row_top,row_bottom",
        ],
        "x_tag": [
            "# Two columns of a tile scan",
            "eubi configure concatenation --concatenation_axes x --x_tag col_left,col_right",
        ],
    },

    # ── with_config ───────────────────────────────────────────────────────────
    "with_config": {
        "name": [
            "# Use the 'hpc' profile for one conversion run",
            "eubi with_config hpc to_zarr /data/input /data/output",
            "",
            "# Combine a named config with an inline override",
            "eubi with_config hpc to_zarr /data/input /data/output --ome_zarr_version 0.5",
        ],
    },

    # ── save_as ───────────────────────────────────────────────────────────────
    "save_as": {
        "name": [
            "# Save the current configuration as a profile named 'hpc'",
            "eubi save_as hpc",
        ],
    },

    # ── update_config ─────────────────────────────────────────────────────────
    "update_config": {
        "name_or_path": [
            "# Propagate edits from the active config into the 'hpc' profile",
            "eubi update_config hpc",
            "",
            "# Write to an explicit file path instead of a named profile",
            "eubi update_config /data/configs/hpc.json",
        ],
        "create": [
            "# Create the target profile if it does not exist yet",
            "eubi update_config hpc --create",
        ],
    },

    # ── delete_config ─────────────────────────────────────────────────────────
    "delete_config": {
        "name": [
            "# Permanently remove the 'hpc' profile",
            "eubi delete_config hpc",
        ],
    },

    # ── show_config ───────────────────────────────────────────────────────────
    "show_config": {
        "name": [
            "# Inspect the 'hpc' profile without activating it",
            "eubi show_config --name hpc",
        ],
    },
}


def _get_example(pname: str, cmd_name: str | None = None) -> str | list[str] | None:
    """Return the most context-appropriate example for pname in cmd_name."""
    if cmd_name and cmd_name in _CMD_EXAMPLES:
        ex = _CMD_EXAMPLES[cmd_name].get(pname)
        if ex is not None:
            return ex
    return _EXAMPLES.get(pname)


# ---------------------------------------------------------------------------
# Type / constraint helpers
# ---------------------------------------------------------------------------

def _split_generic_args(s: str) -> list[str]:
    """Split 'A, B, C' at top-level commas, respecting nested brackets."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in s:
        if ch in ("[", "("):
            depth += 1
            current.append(ch)
        elif ch in ("]", ")"):
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


_NONE_NAMES = frozenset(("None", "NoneType", "type(None)"))


def _parse_str_annotation(s: str) -> str:
    """Parse a string type annotation (from __future__ annotations or forward-ref)."""
    s = s.strip().strip("'\"")

    # 'AggregativePlan | None' — PEP 604 union syntax
    if "|" in s:
        parts = [p.strip() for p in s.split("|")]
        non_none = [p for p in parts if p not in _NONE_NAMES]
        has_none = any(p in _NONE_NAMES for p in parts)
        pieces = [f"`{p}`" for p in non_none]
        if has_none:
            pieces.append("`None`")
        return " or ".join(pieces) if pieces else "any"

    # Union[A, B, ...]
    m = re.match(r"^(?:typing\.)?Union\[(.+)\]$", s)
    if m:
        inner = _split_generic_args(m.group(1))
        non_none = [p.strip() for p in inner if p.strip() not in _NONE_NAMES]
        has_none = any(p.strip() in _NONE_NAMES for p in inner)
        pieces = [f"`{p}`" for p in non_none]
        if has_none:
            pieces.append("`None`")
        return " or ".join(pieces) if pieces else "any"

    # Optional[A]
    m = re.match(r"^(?:typing\.)?Optional\[(.+)\]$", s)
    if m:
        return f"`{m.group(1).strip()}` or `None`"

    # Simple name
    return f"`{s}`"


def _type_str(annotation: Any) -> str:
    """Return a short, readable type label for a parameter annotation."""
    if annotation is inspect.Parameter.empty:
        return "any"
    if annotation is type(None):
        return "`None`"

    # String annotation (from __future__ import annotations or bare forward-ref)
    if isinstance(annotation, str):
        return _parse_str_annotation(annotation)

    origin = get_origin(annotation)

    # Literal[2, 3] → `2` or `3`
    try:
        from typing import Literal as _Lit
        if origin is _Lit:
            return " or ".join(f"`{a}`" for a in get_args(annotation))
    except ImportError:
        pass

    if origin is Union:
        args = get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        has_none = type(None) in args
        pieces = [_type_str(a) for a in non_none]
        if has_none:
            pieces.append("`None`")
        return " or ".join(pieces) if pieces else "any"

    name = getattr(annotation, "__name__", None) or str(annotation)
    name = name.replace("typing.", "").replace("NoneType", "None")
    return f"`{name}`"


def _constraints_str(field_info: Any) -> str:
    """Extract ge/le/gt/lt constraints from a Pydantic FieldInfo as a string."""
    parts: list[str] = []
    for meta in getattr(field_info, "metadata", []):
        ge = getattr(meta, "ge", None)
        le = getattr(meta, "le", None)
        gt = getattr(meta, "gt", None)
        lt = getattr(meta, "lt", None)
        if ge is not None:
            parts.append(f"≥ {ge}")
        if le is not None:
            parts.append(f"≤ {le}")
        if gt is not None:
            parts.append(f"> {gt}")
        if lt is not None:
            parts.append(f"< {lt}")
    return ", ".join(parts) if parts else "—"


def _default_str(field_info: Any) -> str:
    """Return the Pydantic field default as a Markdown string."""
    from pydantic_core import PydanticUndefined
    default = field_info.default
    if default is PydanticUndefined:
        return "*(auto)*" if field_info.default_factory is not None else "*required*"
    if default is None:
        return "—"
    if default == "":
        return '`""`'
    return f"`{default}`"


# ---------------------------------------------------------------------------
# Docstring parser (Google style)
# ---------------------------------------------------------------------------

def _parse_args_section(docstring: str | None) -> dict[str, str]:
    """Return {param_name: description} from the Args: block of a Google docstring."""
    if not docstring:
        return {}
    result: dict[str, str] = {}
    in_args = False
    current: str | None = None
    for line in docstring.splitlines():
        stripped = line.strip()
        if stripped in ("Args:", "Arguments:"):
            in_args = True
            continue
        if in_args:
            # A new top-level section starts (no leading indent)
            if stripped and not line[0].isspace():
                break
            # Match "    param_name[^:]*: description" — [^:]* skips optional type hints
            m = re.match(r"^\s{4,8}(\w+)[^:]*:\s*(.*)", line)
            if m:
                current = m.group(1)
                result[current] = m.group(2).strip()
            elif current and stripped:
                # Stop continuation on **kwargs / *args lines
                if re.match(r"^\*+\w+", stripped):
                    current = None
                else:
                    result[current] = result[current] + " " + stripped
    return result


# ---------------------------------------------------------------------------
# Per-section kwargs table builder
# ---------------------------------------------------------------------------

def _constraints_for(fname: str, finfo: Any) -> str:
    """Return constraints string for a field, with special cases for known enums."""
    if fname == "compressor":
        v2 = ", ".join(f"`{c}`" for c in sorted(SUPPORTED_COMPRESSORS_V2) if c)
        v3 = ", ".join(f"`{c}`" for c in sorted(SUPPORTED_COMPRESSORS_V3) if c)
        return f"v2: {v2}; v3: {v3}"
    if fname == "channel_intensity_limits":
        return "`from_dtype` · `from_array` · `auto`"
    if fname == "metadata_reader":
        return "`bfio` · `bioformats`"
    if fname == "downscale_method":
        return "`simple` · `mean` · `median` · `min` · `max` · `mode`"
    return _constraints_str(finfo)


def _bool_valid_values(pname: str) -> str:
    return f"`--{pname}` to enable &nbsp;·&nbsp; `--{pname} False` to disable"


def _build_kwargs_collapsibles(
    model: Any, extra_desc: dict[str, str], cmd_name: str | None = None
) -> list[str]:
    """Return per-flag collapsible blocks for all fields of a Pydantic model."""
    blocks: list[str] = []
    for fname, finfo in model.model_fields.items():
        type_s = _type_str(finfo.annotation)
        default_s = _default_str(finfo)
        valid_s = _constraints_for(fname, finfo)
        if type_s == "`bool`":
            type_s = "boolean flag"
            valid_s = _bool_valid_values(fname)
        desc = extra_desc.get(fname) or _FIELD_HINTS.get(fname, "—")
        examples = _get_example(fname, cmd_name)
        blocks += _flag_collapsible(fname, type_s, default_s, valid_s, desc, examples)
    return blocks


# ---------------------------------------------------------------------------
# Collapsible section helper
# ---------------------------------------------------------------------------

def _details_block(title: str, table_lines: list[str]) -> list[str]:
    """Wrap table rows in a <details>/<summary> block at the admonition body indent."""
    return [
        "    <details>",
        f"    <summary>{title}</summary>",
        "",
        *table_lines,
        "",
        "    </details>",
        "",
    ]


def _flag_collapsible(
    pname: str,
    typ: str,
    default: str,
    valid_values: str,
    desc: str,
    examples: str | list[str] | None,
) -> list[str]:
    """Render one per-flag collapsible (pure HTML — no Markdown dependency)."""
    lines = [
        "    <details>",
        f"    <summary><code>--{pname}</code></summary>",
        f"    <p><strong>Type:</strong>&nbsp; {typ}</p>",
        f"    <p><strong>Default:</strong>&nbsp; {default}</p>",
    ]
    if valid_values and valid_values != "—":
        lines.append(f"    <p><strong>Valid values:</strong>&nbsp; {valid_values}</p>")
    lines.append(f"    <p>{_rst_inline(desc)}</p>")
    if examples:
        if isinstance(examples, str):
            examples = [examples]
        # Collect non-empty blocks separated by empty strings
        blocks: list[str] = []
        current: list[str] = []
        for ex in examples:
            if ex == "":
                if current:
                    blocks.append("\n".join(current))
                    current = []
            else:
                current.append(ex)
        if current:
            blocks.append("\n".join(current))
        for block in blocks:
            safe = block.replace("<", "&lt;").replace(">", "&gt;")
            ex_lines = safe.splitlines()
            lines.append("    <pre><code>" + ex_lines[0])
            for ln in ex_lines[1:]:
                lines.append("    " + ln)
            lines.append("    </code></pre>")
    lines += ["    </details>", ""]
    return lines


# ---------------------------------------------------------------------------
# Single command card renderer
# ---------------------------------------------------------------------------

def _resolve_method(name: str) -> Any:
    """Resolve a (possibly dotted) command name to its method object.

    ``"to_zarr"`` → ``EuBIBridge.to_zarr``;
    ``"configure.cluster"`` → ``ConfigureGroup.cluster``.
    """
    if "." in name:
        group, _, sub = name.partition(".")
        group_cls = _GROUP_CLASSES.get(group)
        return getattr(group_cls, sub, None) if group_cls else None
    return getattr(EuBIBridge, name, None)


# Group-prefix → class holding that group's subcommand methods.
_GROUP_CLASSES: dict[str, Any] = {
    "configure": ConfigureGroup,
}


_SECTION_RE = re.compile(
    r"^(Args|Arguments|Returns|Raises|Note|Notes|Example|Examples|Yields):"
)


def _rst_inline(text: str) -> str:
    """Convert reStructuredText ``inline code`` to Markdown `inline code`."""
    return re.sub(r"``([^`]+)``", r"`\1`", text)


def _render_doc_body(raw_doc: str) -> list[str]:
    """Render the description between the summary line and the first section
    header, preserving paragraph breaks and reStructuredText ``::`` literal
    blocks (rendered as fenced ``shell`` code) instead of collapsing everything
    into a single run-on line.

    Returns admonition-body lines, indented four spaces.
    """
    out: list[str] = []
    para: list[str] = []
    code: list[str] = []
    in_code = False

    def flush_para() -> None:
        if para:
            text = _rst_inline(" ".join(s.strip() for s in para)).strip()
            if text:
                out.extend(["    " + text, ""])
        para.clear()

    def flush_code() -> None:
        # Drop blank lines that bracket the literal block.
        while code and code[0] == "":
            code.pop(0)
        while code and code[-1] == "":
            code.pop()
        if code:
            out.append("    ```shell")
            out.extend("    " + c for c in code)
            out.extend(["    ```", ""])
        code.clear()

    for raw in raw_doc.splitlines()[1:]:        # skip the one-line summary
        if _SECTION_RE.match(raw.strip()):
            break
        if in_code:
            if raw.strip() == "" or raw[:1].isspace():
                code.append(raw.strip())        # blank or still-indented → code
                continue
            flush_code()                        # dedented prose resumes
            in_code = False
        stripped = raw.strip()
        if stripped == "":
            flush_para()
        elif stripped.endswith("::"):           # RST literal-block introducer
            para.append(stripped[:-1])          # "chaining::" → "chaining:"
            flush_para()
            in_code = True
        else:
            para.append(stripped)
    flush_para()
    flush_code()
    return out


def _render_command(name: str, method: Any) -> str:
    """Render one collapsible admonition card for a command.

    *name* may be dotted (e.g. ``configure.cluster``); the CLI form replaces the
    dot with a space (``eubi configure cluster``).
    """
    display = name.replace(".", " ")
    sig = inspect.signature(method)
    raw_doc = inspect.getdoc(method) or ""
    summary = raw_doc.split("\n")[0].rstrip(".")
    arg_desc = _parse_args_section(raw_doc)

    params = [
        (pname, p)
        for pname, p in sig.parameters.items()
        if pname not in ("self", "args", "kwargs")
    ]

    required = [(n, p) for n, p in params if p.default is inspect.Parameter.empty]
    optional = [(n, p) for n, p in params if p.default is not inspect.Parameter.empty]
    has_kwargs = "kwargs" in sig.parameters and name in COMMAND_KWARGS_SECTIONS

    # Usage line
    req_str = " ".join(n.upper() for n, _ in required)
    usage_parts = [f"eubi {display}", req_str, "[OPTIONS]"]
    usage = " ".join(p for p in usage_parts if p)

    lines: list[str] = [
        f'??? command "**`{display}`**&ensp;—&ensp;{summary}"',
        "",
        "    **Usage:**",
        "    ```shell",
        f"    {usage}",
    ]
    if name in ("to_zarr", "show_pixel_meta", "update_pixel_meta",
                "update_channel_meta", "validate_aggregative"):
        lines.append(f"    eubi with_config NAME {name} {req_str} [OPTIONS]")
    lines += ["    ```", ""]

    # Extended description (between summary line and Args: section) — keeps
    # paragraph breaks and reStructuredText literal blocks intact.
    lines += _render_doc_body(raw_doc)

    # ── Required args ──
    if required:
        req_blocks: list[str] = []
        for pname, p in required:
            typ = _type_str(p.annotation)
            desc = arg_desc.get(pname) or _FIELD_HINTS.get(pname, "—")
            examples = _get_example(pname, name)
            req_blocks += _flag_collapsible(pname, typ, "*required*", "—", desc, examples)
        lines += _details_block("Required arguments", req_blocks)

    # ── Optional explicit args ──
    # For kwargs-expanded commands, only show params with no config model field
    # — the rest are already covered by the override sections below.
    if has_kwargs:
        visible_optional = [(n, p) for n, p in optional if n not in _ALL_CONFIG_FIELDS]
        section_label = "CLI-only options"
    else:
        visible_optional = optional
        section_label = "Optional arguments"

    if visible_optional:
        flag_blocks: list[str] = []
        for pname, p in visible_optional:
            typ = _type_str(p.annotation)
            default = "—" if p.default is None else f"`{p.default}`"
            if p.default is False:
                default = "`False`"
            elif p.default is True:
                default = "`True`"
            elif p.default == "":
                default = '`""`'
            valid = "—"
            if typ == "`bool`":
                typ = "boolean flag"
                valid = _bool_valid_values(pname)
            desc = arg_desc.get(pname) or _FIELD_HINTS.get(pname, "—")
            examples = _get_example(pname, name)
            flag_blocks += _flag_collapsible(pname, typ, default, valid, desc, examples)
        lines += _details_block(section_label, flag_blocks)

    # ── Kwargs sections (only those relevant to this command) ──
    if has_kwargs:
        for section_name in COMMAND_KWARGS_SECTIONS[name]:
            model = KWARGS_SECTIONS[section_name]
            blocks = _build_kwargs_collapsibles(model, arg_desc, cmd_name=name)
            if blocks:
                lines += _details_block(section_name, blocks)

    return "\n".join(lines) + "\n\n"


# ---------------------------------------------------------------------------
# Full page generator
# ---------------------------------------------------------------------------

_SECTION_INTRO = """\
Click any command card below to expand its full parameter reference.

!!! attention "Info"
    Note that all cards below are auto-generated from `EuBIBridge` methods and Pydantic config models. Types, defaults, and valid ranges are extracted directly from the source.

"""

_CONVERSION_INTRO = """\
Click any command card below to expand its full parameter reference.

!!! attention "Info"
    Note that all cards below are auto-generated from `EuBIBridge` methods and Pydantic config models. Types, defaults, and valid ranges are extracted directly from the source.

!!! tip "Tip: Named-config chaining"
    Prefix any command with `with_config NAME` to use a saved config profile:
    ```shell
    eubi with_config hpc to_zarr /data/input /data/output --ome_zarr_version 0.5
    ```

"""

# Slug used for directory and file names
_SLUG: dict[str, str] = {
    "Conversion":    "conversion",
    "Metadata":      "metadata",
    "Configuration": "configuration",
    "Named Configs": "named_configs",
    "Display & Info":"display_info",
    "Reset":         "reset",
}


def generate_section(group: str, commands: list[str]) -> str:
    """Generate Markdown content for one section page."""
    intro = _CONVERSION_INTRO if group == "Conversion" else _SECTION_INTRO
    parts = [f"# {group}\n\n", intro]
    for cmd in commands:
        method = _resolve_method(cmd)
        if method is None:
            continue
        parts.append(_render_command(cmd, method))
    return "".join(parts)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def _write_standalone() -> None:
    """Write one file per section into docs/cli_reference/."""
    out_dir = _ROOT / "docs" / "cli_reference"
    out_dir.mkdir(exist_ok=True)
    for group, commands in COMMAND_GROUPS.items():
        slug = _SLUG[group]
        path = out_dir / f"{slug}.md"
        path.write_text(generate_section(group, commands), encoding="utf-8")
        print(f"Written to {path}")


if __name__ == "__main__":
    _write_standalone()
