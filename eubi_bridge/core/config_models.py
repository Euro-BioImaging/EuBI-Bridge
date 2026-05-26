"""Pydantic configuration models for EuBI-Bridge.

All public ``configure_*`` methods and ``to_zarr()`` keep their existing
signatures unchanged; these models are used internally for validation and
normalisation only.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Compression constants
# ---------------------------------------------------------------------------

# Codecs present in numcodecs but explicitly rejected by TensorStore.
# lz4/lzma pass numcodecs but fail at ts.open() time — caught here first.
REJECTED_COMPRESSORS: frozenset[str] = frozenset({"lz4", "lzma"})

# TensorStore zarr v2 driver: supported codec ids in the "compressor" field.
# zlib/pcodec/zfpy exist in numcodecs but TensorStore does NOT handle them.
SUPPORTED_COMPRESSORS_V2: frozenset[str] = frozenset(
    {"blosc", "bz2", "gzip", "zstd", "none", ""}
)

# TensorStore zarr3 driver: codec names via zarr.codecs API.
SUPPORTED_COMPRESSORS_V3: frozenset[str] = frozenset(
    {"blosc", "gzip", "sharding", "zstd", "crc32ccodec", "none", ""}
)

# Union — used when zarr_format is not known at validation time.
SUPPORTED_COMPRESSORS_ANY: frozenset[str] = SUPPORTED_COMPRESSORS_V2 | SUPPORTED_COMPRESSORS_V3

BLOSC_CNAMES: frozenset[str] = frozenset(
    {"lz4", "lz4hc", "blosclz", "snappy", "zlib", "zstd"}
)

# numcodecs class names for zarr v2
_CODEC_CLASS_V2: dict = {
    "blosc": "Blosc",
    "bz2":   "BZ2",
    "gzip":  "GZip",
    "zstd":  "Zstd",
}

# zarr.codecs class names for zarr v3
_CODEC_CLASS_V3: dict = {
    "blosc":      "BloscCodec",
    "gzip":       "GzipCodec",
    "zstd":       "ZstdCodec",
    "sharding":   "ShardingCodec",
    "crc32ccodec":"CRC32CCodec",
}

# Default Blosc params used when none are supplied
_BLOSC_DEFAULTS_V2: dict = {"cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0}
_BLOSC_DEFAULTS_V3: dict = {"cname": "lz4", "clevel": 5, "shuffle": "shuffle"}

# Integer shuffle → string enum for zarr v3 BloscCodec
_BLOSC_SHUFFLE_STR: dict = {0: "noshuffle", 1: "shuffle", 2: "bitshuffle"}

_V2_LABEL = sorted(SUPPORTED_COMPRESSORS_V2 - {""})
_V3_LABEL = sorted(SUPPORTED_COMPRESSORS_V3 - {""})
_ANY_LABEL = sorted(SUPPORTED_COMPRESSORS_ANY - {""})


# ---------------------------------------------------------------------------
# ClusterConfig
# ---------------------------------------------------------------------------

class ClusterConfig(BaseModel):
    """Distributed cluster / concurrency parameters."""

    model_config = ConfigDict(extra="ignore")

    on_local_cluster: bool = False
    on_slurm: bool = False
    use_threading: bool = False
    max_workers: int = Field(default=4, ge=1, le=256)
    queue_size: int = Field(default=4, ge=1, le=4096)
    region_size_mb: int = Field(default=256, gt=0)
    max_concurrency: int = Field(default=4, ge=1)
    max_concurrent_scenes: int = Field(default=1, ge=1)
    memory_per_worker: str = "1GB"
    tensorstore_data_copy_concurrency: int = Field(default=4, ge=1)
    max_retries: int = Field(default=10, ge=0, le=100)
    bf_read_concurrency: Optional[int] = Field(default=4, ge=1)
    bf_tile_size_mb: float = Field(default=512.0, gt=0.0)
    jvm_memory: Optional[str] = "2g"

    @field_validator('jvm_memory', mode='before')
    @classmethod
    def _normalize_jvm_memory(cls, v):
        if v is None:
            return v
        s = str(v).strip().lower()
        for suffix_in, suffix_out in (('gb', 'g'), ('mb', 'm'), ('kb', 'k')):
            if s.endswith(suffix_in):
                return s[:-len(suffix_in)] + suffix_out
        return s
    # SLURM-specific (ignored when on_slurm=False)
    slurm_time: str = "24:00:00"
    slurm_account: Optional[str] = None
    slurm_partition: Optional[str] = None
    slurm_worker_timeout: int = Field(default=300, gt=0)


# ---------------------------------------------------------------------------
# ReaderConfig
# ---------------------------------------------------------------------------

class ReaderConfig(BaseModel):
    """Reader / scene-selection parameters."""

    model_config = ConfigDict(extra="ignore")

    as_mosaic: bool = False
    view_index: int = Field(default=0, ge=0)
    phase_index: int = Field(default=0, ge=0)
    illumination_index: int = Field(default=0, ge=0)
    # Accept int, 'all', or comma-separated ints (e.g. '0,2,4')
    scene_index: Union[int, str] = 0
    rotation_index: int = Field(default=0, ge=0)
    mosaic_tile_index: Union[int, str, None] = None
    sample_index: int = Field(default=0, ge=0)
    force_bioformats: bool = False

    @field_validator('scene_index', mode='before')
    @classmethod
    def _validate_scene_index(cls, v):
        if isinstance(v, int) or v == 'all':
            return v
        if isinstance(v, str):
            try:
                parts = [int(x.strip()) for x in v.split(',')]
                return parts[0] if len(parts) == 1 else v
            except ValueError:
                raise ValueError(f"scene_index must be an int, 'all', or comma-separated ints; got {v!r}")
        return v

    @field_validator('mosaic_tile_index', mode='before')
    @classmethod
    def _validate_mosaic_tile_index(cls, v):
        if v is None or isinstance(v, int) or v == 'all':
            return v
        if isinstance(v, str):
            try:
                parts = [int(x.strip()) for x in v.split(',')]
                return parts[0] if len(parts) == 1 else v
            except ValueError:
                raise ValueError(f"mosaic_tile_index must be an int, 'all', or comma-separated ints; got {v!r}")
        return v


# ---------------------------------------------------------------------------
# ConversionConfig
# ---------------------------------------------------------------------------

class ConversionConfig(BaseModel):
    """Zarr conversion parameters: format, chunking, sharding, compression, dtype."""

    model_config = ConfigDict(extra="ignore")

    verbose: bool = False
    zarr_format: Literal[2, 3] = 2
    skip_dask: bool = False
    auto_chunk: bool = True
    target_chunk_mb: float = Field(default=1.0, gt=0.0)
    time_chunk: int = Field(default=1, ge=1)
    channel_chunk: int = Field(default=1, ge=1)
    z_chunk: int = Field(default=96, ge=1)
    y_chunk: int = Field(default=96, ge=1)
    x_chunk: int = Field(default=96, ge=1)
    time_shard_coef: int = Field(default=1, ge=1)
    channel_shard_coef: int = Field(default=1, ge=1)
    z_shard_coef: int = Field(default=3, ge=1)
    y_shard_coef: int = Field(default=3, ge=1)
    x_shard_coef: int = Field(default=3, ge=1)
    time_range: Optional[Any] = None
    channel_range: Optional[Any] = None
    z_range: Optional[Any] = None
    y_range: Optional[Any] = None
    x_range: Optional[Any] = None
    dimension_order: str = "tczyx"
    compressor: str = "blosc"
    compressor_params: dict = Field(default_factory=dict)
    overwrite: bool = False
    override_channel_names: bool = False
    channel_intensity_limits: Literal["from_dtype", "from_array", "auto"] = "from_dtype"
    metadata_reader: str = "bfio"
    save_omexml: bool = True
    squeeze: bool = True
    # 'auto' is the on-disk sentinel; _collect_params converts it to None at
    # runtime — we accept both here so JSON round-trips work unchanged.
    dtype: Optional[str] = "auto"

    @field_validator("compressor")
    @classmethod
    def _validate_compressor(cls, v: str) -> str:
        name = (v or "").lower()
        if name in REJECTED_COMPRESSORS:
            raise ValueError(
                f"Compressor '{v}' is not supported by the tensorstore backend. "
                f"For LZ4-like compression use blosc with cname='lz4'."
            )
        if name not in SUPPORTED_COMPRESSORS_ANY:
            raise ValueError(
                f"Unknown compressor '{v}'. "
                f"Supported across v2+v3: {_ANY_LABEL}."
            )
        return v

    @model_validator(mode="after")
    def _validate_compressor_for_format(self) -> "ConversionConfig":
        """Cross-field check: compressor must be valid for the chosen zarr_format."""
        name = (self.compressor or "").lower()
        if name in ("", "none"):
            return self
        if self.zarr_format == 2 and name not in SUPPORTED_COMPRESSORS_V2:
            raise ValueError(
                f"Compressor '{self.compressor}' is not supported for Zarr v2. "
                f"Supported: {_V2_LABEL}."
            )
        if self.zarr_format == 3 and name not in SUPPORTED_COMPRESSORS_V3:
            raise ValueError(
                f"Compressor '{self.compressor}' is not supported for Zarr v3. "
                f"Supported: {_V3_LABEL}."
            )
        # Blosc param checks
        if name == "blosc" and self.compressor_params:
            cname = self.compressor_params.get("cname", "lz4")
            if cname not in BLOSC_CNAMES:
                raise ValueError(
                    f"Invalid blosc cname '{cname}'. "
                    f"Choose from {sorted(BLOSC_CNAMES)}."
                )
            clevel = self.compressor_params.get("clevel", 5)
            if not (0 <= int(clevel) <= 9):
                raise ValueError(f"blosc clevel must be 0–9, got {clevel}.")
        return self


# ---------------------------------------------------------------------------
# DownscaleConfig
# ---------------------------------------------------------------------------

class DownscaleConfig(BaseModel):
    """Pyramid downscaling parameters."""

    model_config = ConfigDict(extra="ignore")

    time_scale_factor: int = Field(default=1, ge=1)
    channel_scale_factor: int = Field(default=1, ge=1)
    z_scale_factor: int = Field(default=2, ge=1)
    y_scale_factor: int = Field(default=2, ge=1)
    x_scale_factor: int = Field(default=2, ge=1)
    n_layers: Optional[int] = None
    min_dimension_size: int = Field(default=64, gt=0)
    downscale_method: Literal["simple", "mean", "median", "min", "max", "mode"] = "simple"

    @field_validator("n_layers")
    @classmethod
    def _validate_n_layers(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 1:
            raise ValueError(f"n_layers must be ≥ 1, got {v}.")
        return v


# ---------------------------------------------------------------------------
# ConcatenationConfig
# ---------------------------------------------------------------------------

class ConcatenationConfig(BaseModel):
    """Aggregative (concatenation) parameters persisted in the config file.

    All fields default to ``None``, which means "unary conversion" — no
    concatenation, no tags required.  When ``concatenation_axes`` is set to a
    non-``None`` value the corresponding tags must be supplied either here or
    at ``to_zarr()`` call time; the cross-field check is performed by
    :class:`AggregativeConversionJob` when the job is built.
    """

    model_config = ConfigDict(extra="ignore")

    concatenation_axes: Optional[Union[str, int]] = None
    time_tag:    Optional[Union[str, List[str]]] = None
    channel_tag: Optional[Union[str, List[str]]] = None
    z_tag:       Optional[Union[str, List[str]]] = None
    y_tag:       Optional[Union[str, List[str]]] = None
    x_tag:       Optional[Union[str, List[str]]] = None


# ---------------------------------------------------------------------------
# CompressorConfig  (replaces the writers.py dataclass)
# ---------------------------------------------------------------------------

class CompressorConfig(BaseModel):
    """Compressor name + parameter bundle.

    Validates the name against the union of v2 and v3 supported codecs.
    When zarr_format is known, use ConversionConfig for the stricter per-format check.
    """

    model_config = ConfigDict(extra="ignore")

    name: str = "blosc"
    params: dict = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        name = (v or "").lower()
        if name in REJECTED_COMPRESSORS:
            raise ValueError(
                f"Compressor '{v}' is not supported by the tensorstore backend. "
                f"For LZ4-like compression use blosc with cname='lz4'."
            )
        if name not in SUPPORTED_COMPRESSORS_ANY:
            raise ValueError(
                f"Unknown compressor '{v}'. "
                f"Supported across v2+v3: {_ANY_LABEL}."
            )
        return v

    @model_validator(mode="after")
    def _validate_blosc_params(self) -> "CompressorConfig":
        if self.name.lower() == "blosc" and self.params:
            cname = self.params.get("cname", "lz4")
            if cname not in BLOSC_CNAMES:
                raise ValueError(
                    f"Invalid blosc cname '{cname}'. "
                    f"Choose from {sorted(BLOSC_CNAMES)}."
                )
            clevel = self.params.get("clevel", 5)
            if not (0 <= int(clevel) <= 9):
                raise ValueError(f"blosc clevel must be 0–9, got {clevel}.")
        return self

    def build(self, zarr_format: int = 2):
        """Instantiate the codec object for the given zarr format.

        Returns ``None`` when compression is disabled (name is None / '' / 'none').
        Heavy imports (numcodecs / zarr.codecs) are deferred to call time so that
        config_models.py itself stays import-light.
        """
        name = (self.name or "").lower()
        if not name or name == "none":
            return None

        params = dict(self.params)  # work on a copy

        if zarr_format == 2:
            import numcodecs
            cls_name = _CODEC_CLASS_V2.get(name)
            if cls_name is None:
                raise ValueError(
                    f"Unsupported compressor '{name}' for Zarr v2. "
                    f"Supported: {_V2_LABEL}"
                )
            if name == "blosc" and not params:
                params = dict(_BLOSC_DEFAULTS_V2)
            return getattr(numcodecs, cls_name)(**params)

        elif zarr_format == 3:
            from zarr import codecs
            cls_name = _CODEC_CLASS_V3.get(name)
            if cls_name is None:
                raise ValueError(
                    f"Unsupported compressor '{name}' for Zarr v3. "
                    f"Supported: {_V3_LABEL}"
                )
            if name == "blosc" and not params:
                params = dict(_BLOSC_DEFAULTS_V3)
            # Convert integer shuffle to the string enum zarr v3 BloscCodec expects
            if name == "blosc" and isinstance(params.get("shuffle"), int):
                params["shuffle"] = _BLOSC_SHUFFLE_STR.get(
                    params["shuffle"], str(params["shuffle"])
                )
            return getattr(codecs, cls_name)(**params)

        else:
            raise ValueError(f"Unsupported zarr_format: {zarr_format}")


# ---------------------------------------------------------------------------
# EuBIConfig  (root — mirrors root_defaults in ebridge.py)
# ---------------------------------------------------------------------------

class EuBIConfig(BaseModel):
    """Root configuration object; mirrors ``EuBIBridge.root_defaults``."""

    model_config = ConfigDict(extra="ignore")

    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    readers: ReaderConfig = Field(default_factory=ReaderConfig)
    conversion: ConversionConfig = Field(default_factory=ConversionConfig)
    downscale: DownscaleConfig = Field(default_factory=DownscaleConfig)


# ---------------------------------------------------------------------------
# ConversionJob  (built in ebridge.py, consumed by dispatcher.py)
# ---------------------------------------------------------------------------

# Keys owned by the four config models — used by ConversionJob.from_kwargs to
# split the flat merged dict into typed sub-models vs. passthrough extras.
_CONFIG_MODEL_KEYS: frozenset[str] = frozenset(
    set(ClusterConfig.model_fields)
    | set(ReaderConfig.model_fields)
    | set(ConversionConfig.model_fields)
    | set(DownscaleConfig.model_fields)
)


class ConversionJob(BaseModel):
    """A single ready-to-execute conversion job with all parameters validated.

    Built by ``ConversionManager.to_zarr()`` in ``ebridge.py`` — one per input
    file, after the Stage 2 CSV triage merge.  The dispatcher receives a list
    of these and treats them as already valid; no further Pydantic work needed
    on its side.

    ``extra`` carries any kwargs not owned by the four config models (e.g.
    ``y_scale``, ``channel_colors``, ``time_tag``).  They are passed through
    unchanged to the worker.
    """

    model_config = ConfigDict(extra="ignore")

    input_path: str
    output_path: str
    cluster: ClusterConfig    = Field(default_factory=ClusterConfig)
    readers: ReaderConfig     = Field(default_factory=ReaderConfig)
    conversion: ConversionConfig = Field(default_factory=ConversionConfig)
    downscale: DownscaleConfig   = Field(default_factory=DownscaleConfig)
    extra: dict               = Field(default_factory=dict)

    @classmethod
    def from_kwargs(
        cls,
        input_path: str,
        output_path: str,
        kwargs: dict,
    ) -> "ConversionJob":
        """Build a validated ``ConversionJob`` from a flat merged-kwargs dict.

        Each config section is validated independently (``extra='ignore'`` on
        the sub-models silently drops unknown keys).  Keys not belonging to any
        model go into ``extra`` for passthrough to the worker.
        """
        return cls(
            input_path=input_path,
            output_path=output_path,
            cluster=ClusterConfig(**kwargs),
            readers=ReaderConfig(**kwargs),
            conversion=ConversionConfig(**kwargs),
            downscale=DownscaleConfig(**kwargs),
            extra={k: v for k, v in kwargs.items() if k not in _CONFIG_MODEL_KEYS},
        )

    def to_worker_kwargs(self) -> dict:
        """Flatten back to the dict that ``unary_worker_sync`` expects."""
        result: dict = {}
        result.update(self.cluster.model_dump())
        result.update(self.readers.model_dump())
        result.update(self.conversion.model_dump())
        result.update(self.downscale.model_dump())
        result.update(self.extra)
        return result


# Keys owned specifically by AggregativeConversionJob (not in the four base models)
_AGGREGATIVE_KEYS: frozenset[str] = frozenset({
    "concatenation_axes", "time_tag", "channel_tag",
    "z_tag", "y_tag", "x_tag", "includes", "excludes",
})


class AggregativeConversionJob(BaseModel):
    """A validated aggregative (concatenation) conversion job.

    Built by ``ConversionManager.to_zarr()`` when ``concatenation_axes`` is
    set.  Validation of the tag-axis consistency and backend compatibility
    happens here so ``dispatch_aggregative_job`` in ``dispatcher.py`` receives
    an already-verified object.
    """

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    input_path: Union[str, List[str]]
    output_path: str
    cluster: ClusterConfig       = Field(default_factory=ClusterConfig)
    readers: ReaderConfig        = Field(default_factory=ReaderConfig)
    conversion: ConversionConfig = Field(default_factory=ConversionConfig)
    downscale: DownscaleConfig   = Field(default_factory=DownscaleConfig)

    # Aggregative-specific fields
    concatenation_axes: Optional[Union[str, int, tuple]] = None
    time_tag:    Optional[Any] = None
    channel_tag: Optional[Any] = None
    z_tag:       Optional[Any] = None
    y_tag:       Optional[Any] = None
    x_tag:       Optional[Any] = None
    includes:    Optional[Any] = None
    excludes:    Optional[Any] = None

    # Pass-through kwargs not covered by any model field
    extra: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_backend_compat(self) -> "AggregativeConversionJob":
        """on_local_cluster is incompatible with concatenation_axes."""
        if self.cluster.on_local_cluster:
            raise ValueError(
                "on_local_cluster=True is not compatible with concatenation_axes. "
                "Use on_slurm=True or remove on_local_cluster."
            )
        return self

    @model_validator(mode="after")
    def _validate_tags_for_axes(self) -> "AggregativeConversionJob":
        """Each concatenation axis must have a corresponding tag."""
        if self.concatenation_axes is None:
            return self
        if isinstance(self.concatenation_axes, str):
            axes = list(self.concatenation_axes.lower())
        else:
            axes = list(self.concatenation_axes)
        axis_to_tag: dict = {
            "t": self.time_tag, "c": self.channel_tag,
            "z": self.z_tag, "y": self.y_tag, "x": self.x_tag,
        }
        missing = [ax for ax in axes if ax in axis_to_tag and axis_to_tag[ax] is None]
        if missing:
            raise ValueError(
                f"concatenation_axes='{self.concatenation_axes}' requires tags for: "
                f"{', '.join(missing)}."
            )
        return self

    @classmethod
    def from_kwargs(
        cls,
        input_path: str,
        output_path: str,
        kwargs: dict,
    ) -> "AggregativeConversionJob":
        """Build from a flat merged-kwargs dict (mirrors ConversionJob.from_kwargs)."""
        agg_keys = _AGGREGATIVE_KEYS
        return cls(
            input_path=input_path,
            output_path=output_path,
            cluster=ClusterConfig(**kwargs),
            readers=ReaderConfig(**kwargs),
            conversion=ConversionConfig(**kwargs),
            downscale=DownscaleConfig(**kwargs),
            concatenation_axes=kwargs.get("concatenation_axes"),
            time_tag=kwargs.get("time_tag"),
            channel_tag=kwargs.get("channel_tag"),
            z_tag=kwargs.get("z_tag"),
            y_tag=kwargs.get("y_tag"),
            x_tag=kwargs.get("x_tag"),
            includes=kwargs.get("includes"),
            excludes=kwargs.get("excludes"),
            extra={k: v for k, v in kwargs.items()
                   if k not in _CONFIG_MODEL_KEYS and k not in agg_keys},
        )

    def to_conversion_kwargs(self) -> dict:
        """Flatten back to the dict that ``run_conversions_with_concatenation`` expects."""
        result: dict = {}
        result.update(self.cluster.model_dump())
        result.update(self.readers.model_dump())
        result.update(self.conversion.model_dump())
        result.update(self.downscale.model_dump())
        result.update(self.extra)
        return result


# ---------------------------------------------------------------------------
# TensorStoreWriteConfig
# ---------------------------------------------------------------------------

class TensorStoreWriteConfig(BaseModel):
    """Writer-specific parameters for ``write_with_tensorstore_async``.

    ``region_workers > 1`` activates spatial partitioning of a **single** output
    zarr across multiple CPU processes.  It is mutually exclusive with
    file-level parallelism — the caller is responsible for setting it only when
    writing a single output zarr (``file_workers == 1``).
    """

    model_config = ConfigDict(extra="ignore")

    max_concurrency:        int           = Field(default=8,    ge=1)
    compute_batch_size:     int           = Field(default=8,    ge=1)
    memory_limit_per_batch: int           = Field(default=1024, gt=0)
    ts_io_concurrency:      Optional[int] = Field(default=None, ge=1)
    region_workers:         int           = Field(default=1,    ge=1)

    @classmethod
    def from_kwargs(cls, kwargs: dict) -> "TensorStoreWriteConfig":
        return cls(**{k: v for k, v in kwargs.items()
                      if k in cls.model_fields})


# ---------------------------------------------------------------------------
# AggregativePlan  (result of ConversionManager.validate_aggregative)
# ---------------------------------------------------------------------------

class AggregativeOutputInfo(BaseModel):
    """Metadata about one planned output zarr in an aggregative conversion."""

    model_config = ConfigDict(extra="ignore")

    output_path:  str
    source_files: List[str]


class AggregativePlan(BaseModel):
    """Result of ``ConversionManager.validate_aggregative()``.

    Describes what the conversion will produce and how many output zarrs
    will be written in parallel (``file_workers``).
    """

    model_config = ConfigDict(extra="ignore")

    n_outputs:    int
    outputs:      List[AggregativeOutputInfo]
    file_workers: int

    def __str__(self) -> str:
        total_files = sum(len(out.source_files) for out in self.outputs)
        lines = [
            f"AggregativePlan — {self.n_outputs} output group(s), "
            f"file_workers={self.file_workers}",
            f"The number of source files: {total_files}",
            "The output path(s):",
        ]
        for i, out in enumerate(self.outputs, 1):
            lines.append(f"  [{i}] {out.output_path}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ChunkConfig
# ---------------------------------------------------------------------------

class ChunkConfig(BaseModel):
    """Per-axis chunk sizes for ND image arrays.

    All fields default to ``None`` (meaning "inherit from the array or use the
    caller's default").  Use ``from_kwargs`` to build from a flat kwargs dict
    (keys ``time_chunk``, ``channel_chunk``, ``z_chunk``, ``y_chunk``,
    ``x_chunk``), or ``from_array`` to read chunk sizes from any supported
    array type.
    """

    model_config = ConfigDict(extra="ignore")

    t: Optional[int] = Field(default=None, ge=1)
    c: Optional[int] = Field(default=None, ge=1)
    z: Optional[int] = Field(default=None, ge=1)
    y: Optional[int] = Field(default=None, ge=1)
    x: Optional[int] = Field(default=None, ge=1)

    _KWARG_TO_AXIS: ClassVar[Dict[str, str]] = {
        "time_chunk":    "t",
        "channel_chunk": "c",
        "z_chunk":       "z",
        "y_chunk":       "y",
        "x_chunk":       "x",
    }

    @classmethod
    def from_kwargs(cls, **kwargs) -> "ChunkConfig":
        """Build from flat conversion-worker kwargs (e.g. ``z_chunk=96``)."""
        axis_vals: dict = {}
        for kw, ax in cls._KWARG_TO_AXIS.items():
            val = kwargs.get(kw)
            if val is not None:
                try:
                    import math
                    if not math.isnan(float(val)):
                        axis_vals[ax] = int(val)
                except (TypeError, ValueError):
                    pass
        return cls(**axis_vals)

    @classmethod
    def from_array(cls, arr, axes: str) -> "ChunkConfig":
        """Read chunk sizes from an array and map them to TCZYX fields."""
        from eubi_bridge.utils.array_utils import get_array_chunks
        raw = get_array_chunks(arr)
        if raw is None or len(raw) != len(axes):
            return cls()
        axis_vals = {ax: raw[i] for i, ax in enumerate(axes) if ax in "tczyx"}
        return cls(**axis_vals)

    def as_tuple(self, axes: str) -> Tuple[int, ...]:
        """Return a chunk tuple ordered by *axes* (e.g. ``'tczyx'``).

        Axes not present in this config receive a fallback of 1.
        """
        defaults = {"t": 1, "c": 1, "z": 1, "y": 1, "x": 1}
        vals = {ax: (getattr(self, ax) or defaults[ax]) for ax in "tczyx"}
        return tuple(vals[ax] for ax in axes if ax in vals)


# ---------------------------------------------------------------------------
# MetadataUpdateConfig
# ---------------------------------------------------------------------------

def _coerce_range(v) -> Optional[Tuple[int, int]]:
    """Coerce a range value: passthrough None/NaN, validate (start, stop) pair."""
    if v is None:
        return None
    try:
        import math
        if math.isnan(float(v)):
            return None
    except (TypeError, ValueError):
        pass
    try:
        start, stop = int(v[0]), int(v[1])
    except (TypeError, ValueError, IndexError):
        raise ValueError(f"Range must be a (start, stop) pair, got {v!r}")
    if start >= stop:
        raise ValueError(f"Range start {start} must be less than stop {stop}")
    return (start, stop)


class MetadataUpdateConfig(BaseModel):
    """Parameters for pixel-metadata and channel-metadata update operations.

    All scale / unit fields default to ``None`` meaning "keep the value already
    stored in the file".  Range fields accept ``None`` (no crop on that axis) or
    a ``(start, stop)`` int pair validated to have ``start < stop``.
    """

    model_config = ConfigDict(extra="ignore")

    # Pixel sizes — None → keep existing value from file
    time_scale:    Optional[float] = Field(default=None, gt=0)
    channel_scale: Optional[float] = Field(default=None, gt=0)
    z_scale:       Optional[float] = Field(default=None, gt=0)
    y_scale:       Optional[float] = Field(default=None, gt=0)
    x_scale:       Optional[float] = Field(default=None, gt=0)

    # Physical units — None → keep existing value from file
    time_unit: Optional[str] = None
    z_unit:    Optional[str] = None
    y_unit:    Optional[str] = None
    x_unit:    Optional[str] = None

    # Crop ranges as (start, stop) — None → no crop on that axis
    time_range:    Optional[Tuple[int, int]] = None
    channel_range: Optional[Tuple[int, int]] = None
    z_range:       Optional[Tuple[int, int]] = None
    y_range:       Optional[Tuple[int, int]] = None
    x_range:       Optional[Tuple[int, int]] = None

    # Misc update options
    squeeze:    bool = False
    save_omexml: bool = True
    series:      Union[int, str] = "all"

    @field_validator(
        "time_range", "channel_range", "z_range", "y_range", "x_range",
        mode="before",
    )
    @classmethod
    def _validate_range(cls, v):
        return _coerce_range(v)

    def scales_for(self, manager) -> dict:
        """Merge explicit scale overrides with the manager's existing scaledict."""
        field_map = {
            "t": self.time_scale,
            "c": self.channel_scale,
            "z": self.z_scale,
            "y": self.y_scale,
            "x": self.x_scale,
        }
        defaults = manager.scaledict
        return {
            ax: (field_map[ax] if field_map.get(ax) is not None else defaults.get(ax))
            for ax in manager.axes
        }

    def units_for(self, manager) -> dict:
        """Merge explicit unit overrides with the manager's existing unitdict."""
        field_map = {
            "t": self.time_unit,
            "z": self.z_unit,
            "y": self.y_unit,
            "x": self.x_unit,
        }
        defaults = manager.unitdict
        return {
            ax: (field_map[ax] if field_map.get(ax) is not None else defaults.get(ax))
            for ax in manager.axes
            if ax != "c"   # channel axis carries no physical unit
        }

    def crop_slices(self) -> list:
        """Return the list of (start, stop) ranges in tczyx order for manager.crop()."""
        return [
            self.time_range,
            self.channel_range,
            self.z_range,
            self.y_range,
            self.x_range,
        ]
