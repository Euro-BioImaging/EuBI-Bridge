"""Pydantic configuration models for EuBI-Bridge.

All public ``configure_*`` methods and ``to_zarr()`` keep their existing
signatures unchanged; these models are used internally for validation and
normalisation only.
"""
from __future__ import annotations

import math
import re
import warnings
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple, Union

from pydantic import (BaseModel, ConfigDict, Field, ValidationInfo, field_validator,
                      model_validator)

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

# Every name here is written AND read by both writers (TensorStore and zarrista
# bundle their own c-blosc). 'snappy' is the exception for OTHER readers:
# numcodecs (zarr-python, ome-zarr-py, napari-ome-zarr) ships blosc without it
# since 0.9.0 and cannot decode it, so choosing it is warned about loudly.
BLOSC_CNAMES: frozenset[str] = frozenset(
    {"lz4", "lz4hc", "blosclz", "snappy", "zlib", "zstd"}
)

_limited_cname_warned: set = set()


def dyna_writes_limited_cnames() -> bool:
    """Whether the installed dyna-zarr writes a blosc cname that numcodecs lacks
    (with its ``LimitedReaderSupportWarning``, dyna-zarr > 0.1.0). dyna-zarr
    0.1.0 refuses it inside the write, so eubi then refuses it up front."""
    try:
        import dyna_zarr
    except Exception:
        return False
    return hasattr(dyna_zarr, "LimitedReaderSupportWarning")


def _check_blosc_cname(cname: str) -> None:
    """Refuse an unknown blosc cname. A known one that numcodecs lacks (snappy):
    warn once per name and process when dyna-zarr writes it, refuse it when the
    installed dyna-zarr cannot (dyna-zarr warns too, in the write)."""
    if cname not in BLOSC_CNAMES:
        raise ValueError(
            f"Invalid blosc cname '{cname}'. "
            f"Choose from {sorted(BLOSC_CNAMES)}."
        )
    try:
        from numcodecs.blosc import list_compressors
        available = set(list_compressors())
    except Exception:
        return
    if cname not in available and not dyna_writes_limited_cnames():
        raise ValueError(
            f"blosc cname '{cname}' is not available in the installed blosc "
            f"library, and the installed dyna-zarr does not write it. "
            f"Choose from {sorted(available)}."
        )
    if cname not in available and cname not in _limited_cname_warned:
        _limited_cname_warned.add(cname)
        _logger.warning(
            f"COMPRESSOR SUPPORT IS LIMITED: blosc cname '{cname}' is written and "
            f"read by EuBI-Bridge, but many OME-Zarr readers cannot open it: "
            f"zarr-python and the tools built on it (ome-zarr-py, napari-ome-zarr) "
            f"cannot decode it. Use 'lz4' (similar speed) or 'zstd' for outputs "
            f"others will open."
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
# OME-Zarr version <-> zarr container format
# ---------------------------------------------------------------------------

import logging as _logging
_logger = _logging.getLogger(__name__)

# Supported OME-Zarr (NGFF) versions and the zarr container format each requires.
# Add new versions here as they are released (e.g. "0.6": 3).
#: the OME-Zarr versions eubi writes, and the zarr format each one is stored in. "rfc8":
#: NGFF RFC 8 (draft), a collection holding the image as an NGFF 0.5 multiscale
OME_ZARR_VERSIONS: Dict[str, int] = {"0.4": 2, "0.5": 3, "0.6": 3, "rfc8": 3}

_zarr_format_deprecation_warned = False
_bf_tile_size_deprecation_warned = False

#: a physical scale written as text: digits, with '.' as the decimal point
_SCALE_TEXT = re.compile(r"\d+\.?\d*|\.\d+")


def parse_scale(value: Any, name: str = "scale") -> Optional[float]:
    """A physical scale (a pixel size or time step) given by a user, or None when blank.

    One notation only: digits, with '.' as the decimal point (``0.26``, ``1800``,
    ``1000000``). Everything else is refused with an error naming the setting, never
    dropped: a unit or other text (``30 min``; the unit is set separately), a comma
    (``0,5`` or ``1,000,000``: decimal or thousands separator depending on the
    country), spaces inside, underscores, exponents, and values that are not above 0.
    A number (not text) is taken as it is; NaN, a blank table cell, means blank.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name}: {value!r} is not a number")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if not _SCALE_TEXT.fullmatch(text):
            raise ValueError(
                f"{name}: {value!r} is not a plain number. Write digits with '.' as the "
                f"decimal point and nothing else: no unit, spaces, commas or other "
                f"separators (e.g. 0.26, 1800 or 1000000). The unit is chosen separately.")
        number = float(text)
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name}: {value!r} is not a number") from None
        if math.isnan(number):              # a blank cell of a table read by pandas
            return None
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name}: must be a number above 0, got {value!r}")
    return number

#: Settings that no longer have any effect: still accepted, so existing
#: configs, saved batches and scripts keep working, warned about once each,
#: and ignored.  name -> why.
RETIRED_SETTINGS: Dict[str, str] = {
    # micro-reader, the default reader, creates no dask arrays (2026-10-05)
    "skip_dask": "micro-reader reads without dask",
    # the local dask cluster backend (2026-10-05): the process pool does the
    # same job.  (SLURM stays, in the full eubi-bridge.)
    "on_local_cluster": "conversions run in this machine's own worker processes",
}
_retired_warned: set = set()


def warn_retired(name: str, value) -> None:
    """Warn once per retired setting, when it is given a value that would
    have done something (None, False and 'default' would not have)."""
    if value is None or value is False or value == "default" or name in _retired_warned:
        return
    _retired_warned.add(name)
    _logger.warning(
        f"DEPRECATION: '{name}' is no longer used and is ignored: "
        f"{RETIRED_SETTINGS[name]}. Remove it from your config, batch or script; "
        f"it will be rejected in a future release.")


#: Settings renamed to the names the whole ecosystem uses (dyna-zarr, ome-zarr-
#: pyramid, tilewise-ccl, ...; 2026-10-08): old -> new, same meaning. The old names
#: still work in configs, the CLI, batches and scripts: a saved config is updated
#: once, when it is loaded; anywhere else they are warned about once each.
RENAMED_SETTINGS: Dict[str, str] = {
    "max_workers": "max_processes",
    "max_concurrency": "max_threads",
    "micro_read_concurrency": "decode_threads",
    "bf_read_concurrency": "bioformats_threads",
    # memory_budget (2026-10-08) is the whole run's memory on this machine; the old
    # per-process setting lives on as the SLURM worker's memory
    "memory_per_worker": "slurm_worker_memory",
}
#: The series index renamed with micro-reader 0.1.0's index ('scene' became 'series',
#: eubi-bridge 0.3.0): old -> new, same meaning. Given anywhere a user gives settings,
#: the old name still works, with a FutureWarning. Kept for several releases: removed
#: in a planned minor release after the alias has shipped in at least three releases.
RENAMED_INDEX_SETTINGS: Dict[str, str] = {
    "scene_index": "series_index",
}
_renamed_warned: set = set()


def _rename_message(old: str, new: str) -> str:
    if old in RENAMED_INDEX_SETTINGS:
        return f"{old} is deprecated; it is now called {new} (same meaning)"
    return f"'{old}' is now called '{new}' (same meaning); please use the new name"


class SeriesIndexConflictWarning(UserWarning):
    """``scene_index`` and ``series_index`` were both given, with different values:
    ``series_index`` is used. A file both eubi-bridge 0.2.x and 0.3.0 write (a saved
    config, a batch.csv) gets this when 0.2.x changed only its own key."""


def _index_text(value: Any) -> str:
    """An index value as text that compares equal across the forms it is saved in:
    ``1`` / ``1.0`` (a pandas column with gaps) / ``"1"``, ``[0, 2]`` / ``"0,2"``."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, (list, tuple)):
        value = ",".join(_index_text(v) for v in value)
    return str(value).replace(" ", "").strip("[]()").lower()


def same_index(a: Any, b: Any) -> bool:
    """True when two index values name the same selection (``_index_text``)."""
    return _index_text(a) == _index_text(b)


#: what saving does to a pair that differs (``resolve_index_pair(on_save=...)``):
#: a saved config and a batch.csv are written from the series index alone (both
#: names, equal), a batch_config.json as it was loaded
INDEX_SAVE_REPLACES = "replaces"
INDEX_SAVE_KEEPS = "keeps"


def resolve_index_pair(scene: Any, series: Any, where: str = "",
                       on_save: Optional[str] = None) -> Any:
    """The series index from a place holding both names (plan 8.32: eubi-bridge 0.3.0
    writes both, for several releases): *series*. Silent when the two agree; when
    they differ, a SeriesIndexConflictWarning (and the log) names both values, since
    eubi-bridge 0.2.x reads the other one. *where* prefixes the message (a file, a
    batch row). *on_save*, for a file eubi-bridge saves again: INDEX_SAVE_REPLACES
    (the message says the next save writes series_index's value under both names,
    so the differing scene_index is gone) or INDEX_SAVE_KEEPS (saved as it is)."""
    if not same_index(scene, series):
        msg = (f"{where}scene_index={scene!r} and series_index={series!r} differ; "
               f"series_index={series!r} is used. scene_index is the name eubi-bridge "
               "0.2.x reads and writes, so one of them was changed by an older "
               "eubi-bridge. Set the one you want as series_index: eubi-bridge 0.3.0 "
               "writes both.")
        if on_save == INDEX_SAVE_REPLACES:
            msg += (" The next time eubi-bridge 0.3.0 saves this file (any setting "
                    "saved, or a load that updates the file) it writes "
                    f"series_index={series!r} under both names: scene_index={scene!r} "
                    f"is then replaced, and eubi-bridge 0.2.x reads {series!r} too.")
        elif on_save == INDEX_SAVE_KEEPS:
            msg += (" Saving it again in eubi-bridge 0.3.0 keeps both values as they "
                    "are.")
        warnings.warn(msg, SeriesIndexConflictWarning, stacklevel=3)
        _logger.warning(msg)
    return series


def rename_settings(params: dict) -> dict:
    """*params* with the old names of renamed settings replaced by the new ones
    (RENAMED_SETTINGS, RENAMED_INDEX_SETTINGS), warning once per old name in the log;
    the index renames also raise a FutureWarning each time. Given both, the new one
    wins; for the index, both given and equal is silent (a file eubi-bridge 0.3.0
    wrote both into) and both given and different is a SeriesIndexConflictWarning
    (``resolve_index_pair``)."""
    renames = {**RENAMED_SETTINGS, **RENAMED_INDEX_SETTINGS}
    if not isinstance(params, dict) or not renames.keys() & params.keys():
        return params
    out = dict(params)
    for old, new in renames.items():
        if old not in out:
            continue
        value = out.pop(old)
        if new in out and old in RENAMED_INDEX_SETTINGS:
            resolve_index_pair(value, out[new])
            continue
        if new in out:
            msg = f"'{old}' and its new name '{new}' were both given; '{new}' is used"
        else:
            out[new] = value
            msg = _rename_message(old, new)
        if old in RENAMED_INDEX_SETTINGS:
            warnings.warn(msg, FutureWarning, stacklevel=2)
        if old not in _renamed_warned:
            _renamed_warned.add(old)
            _logger.warning(f"DEPRECATION: {msg}.")
    return out


#: The same rename in the GUI's reader settings (its saved config, a batch's
#: batch_config.json): 0.2.x name -> 0.3.0 name. The GUI writes both pairs for several
#: releases (an eubi-bridge 0.2.x GUI reads its own) and reads the new pair first.
RENAMED_UI_INDEX_KEYS: Dict[str, str] = {
    "readAllScenes": "readAllSeries",
    "sceneIndices": "seriesIndices",
}


def series_ui(reader: Any, on_save: Optional[str] = None) -> Any:
    """A GUI reader config with the series keys (``readAllSeries``, ``seriesIndices``):
    taken from the 0.2.x scene keys, with a FutureWarning, when it has only those.
    The scene keys stay, so the result can be written back as both pairs. *on_save*:
    what saving the file it came from does to pairs that differ
    (``resolve_index_pair``)."""
    if not isinstance(reader, dict):
        return reader
    _check_ui_pairs(reader, on_save)
    old = [k for k, new in RENAMED_UI_INDEX_KEYS.items() if k in reader and new not in reader]
    if not old:
        return reader
    out = dict(reader)
    for key in old:
        out[RENAMED_UI_INDEX_KEYS[key]] = out[key]
    msg = (" and ".join(old) + (" are" if len(old) > 1 else " is") + " deprecated; now "
           + " and ".join(RENAMED_UI_INDEX_KEYS[k] for k in old) + " (same meaning)")
    warnings.warn(msg, FutureWarning, stacklevel=2)
    if "ui:series" not in _renamed_warned:
        _renamed_warned.add("ui:series")
        _logger.warning(f"DEPRECATION: {msg}.")
    return out


def _check_ui_pairs(reader: dict, on_save: Optional[str] = None) -> None:
    """Warn (SeriesIndexConflictWarning) when a GUI reader config holds both pairs of
    series keys and they select different series; the series pair is used."""
    if all(k in reader for pair in RENAMED_UI_INDEX_KEYS.items() for k in pair):
        def selection(read_all: str, indices: str) -> Any:
            return "all" if reader[read_all] else (reader[indices] or "0")
        resolve_index_pair(selection("readAllScenes", "sceneIndices"),
                           selection("readAllSeries", "seriesIndices"),
                           where="GUI reader settings (readAllScenes / sceneIndices "
                                 "against readAllSeries / seriesIndices): ",
                           on_save=on_save)
        return
    for old, new in RENAMED_UI_INDEX_KEYS.items():
        if old in reader and new in reader:
            resolve_index_pair(reader[old], reader[new],
                               where=f"GUI reader settings ({old} against {new}): ",
                               on_save=on_save)


def series_ui_pairs(read_all: bool, indices: str) -> dict:
    """The GUI's series setting as both pairs of keys (plan 8.32)."""
    return {"readAllSeries": read_all, "seriesIndices": indices,
            "readAllScenes": read_all, "sceneIndices": indices}


def with_both_index_names(config: dict) -> dict:
    """A config to save: its ``readers`` section with ``scene_index`` beside
    ``series_index``, equal (plan 8.32: for several releases, so an eubi-bridge 0.2.x
    sharing the config folder reads the same series). The config itself is unchanged."""
    readers = config.get("readers") if isinstance(config, dict) else None
    if not isinstance(readers, dict) or "series_index" not in readers:
        return config
    section = {}
    for key, value in readers.items():
        if key == "scene_index":
            continue
        section[key] = value
        if key == "series_index":
            section["scene_index"] = value
    return {**config, "readers": section}


#: the config sections renamed settings live in
_RENAMED_BY_SECTION = {"cluster": RENAMED_SETTINGS, "readers": RENAMED_INDEX_SETTINGS}


def rename_config_keys(config: dict, where: str = "") -> bool:
    """Rename old keys in a config's ``cluster`` and ``readers`` sections in place
    (the value kept); True when something was renamed. A saved config is the user's
    choice from before the rename, so this only logs. A ``readers`` section holding
    both index names (eubi-bridge 0.3.0 saves both, ``with_both_index_names``) keeps
    ``series_index``: silently when the two agree, else with a
    SeriesIndexConflictWarning naming both (*where* names the file) and saying that
    the next save writes ``series_index`` under both names (``with_both_index_names``),
    replacing the differing ``scene_index``."""
    if not isinstance(config, dict):
        return False
    done = []
    for name, renames in _RENAMED_BY_SECTION.items():
        section = config.get(name)
        if not isinstance(section, dict):
            continue
        for old in [old for old in renames if old in section]:
            value = section.pop(old)
            new = renames[old]
            if new not in section:
                section[new] = value
                done.append(f"'{old}' to '{new}'")
            elif name == "readers":
                resolve_index_pair(value, section[new], where=where,
                                   on_save=INDEX_SAVE_REPLACES)
            else:
                done.append(f"'{old}' to '{new}'")      # the new one kept
    if done:
        _logger.info("Config: renamed " + ", ".join(done) + " (same meaning).")
    return bool(done)


def drop_retired(params: dict) -> dict:
    """*params* without the retired settings, warning about any given a value."""
    for name in RETIRED_SETTINGS.keys() & params.keys():
        warn_retired(name, params[name])
    return {k: v for k, v in params.items() if k not in RETIRED_SETTINGS}


def resolve_ome_zarr_target(ome_zarr_version, zarr_format=2, warn: bool = False) -> Tuple[str, int]:
    """Resolve the effective ``(ome_zarr_version, zarr_format)`` for an output.

    ``ome_zarr_version`` supersedes ``zarr_format``: when it names a recognised
    OME-Zarr version it wins, and the zarr container format is derived from it.
    When it is ``None`` or unrecognised, fall back to ``zarr_format`` (2 -> '0.4',
    3 -> '0.5').  When ``warn`` is True a one-time deprecation warning is emitted
    on fallback, steering users to ``ome_zarr_version`` (suppressed for the
    silent normalisation that every ConversionConfig build performs).

    Returns ``(version_str, zarr_format_int)``.
    """
    global _zarr_format_deprecation_warned
    if ome_zarr_version is not None:
        v = str(ome_zarr_version).strip().lstrip("vV").lower()
        if v in OME_ZARR_VERSIONS:
            return v, OME_ZARR_VERSIONS[v]
        # never another version than the one asked for (it fell back to zarr_format,
        # i.e. 0.4, before eubi-bridge 0.3.2)
        raise ValueError(
            f"ome_zarr_version {ome_zarr_version!r}: eubi-bridge writes OME-Zarr "
            f"{', '.join(repr(k) for k in OME_ZARR_VERSIONS)}. Choose one of them, or "
            f"leave ome_zarr_version unset for the zarr_format default.")
    try:
        fmt = int(zarr_format)
    except (TypeError, ValueError):
        fmt = 2
    if fmt not in (2, 3):
        fmt = 2
    version = "0.5" if fmt == 3 else "0.4"
    if warn and not _zarr_format_deprecation_warned:
        _logger.warning(
            "DEPRECATION: 'zarr_format' is deprecated; use 'ome_zarr_version' "
            "instead (e.g. ome_zarr_version='0.4', '0.5', '0.6' or 'rfc8'). This run "
            "resolved to "
            "OME-Zarr %s (zarr v%d).", version, fmt,
        )
        _zarr_format_deprecation_warned = True
    return version, fmt


# ---------------------------------------------------------------------------
# ClusterConfig
# ---------------------------------------------------------------------------

class ClusterConfig(BaseModel):
    """Distributed cluster / concurrency parameters."""

    model_config = ConfigDict(extra="ignore")

    # RETIRED (2026-10-05): accepted and ignored (RETIRED_SETTINGS); the
    # process pool does the same job.
    on_local_cluster: Optional[bool] = None
    # SLURM through dask-jobqueue: the full eubi-bridge only (lite refuses it)
    on_slurm: bool = False
    use_threading: bool = False
    # Conversions at once, one image (output) each, in separate processes
    # (threads with use_threading). Was max_workers (RENAMED_SETTINGS).
    max_processes: int = Field(default=4, ge=1, le=256)
    # Conversions that actually run at once: min(max_processes, the number of
    # jobs / output groups), set by the dispatcher (not a user setting).
    # None: max_processes.  The region budget divides RAM among these only.
    concurrent_jobs: Optional[int] = Field(default=None, ge=1)
    queue_size: int = Field(default=4, ge=1, le=4096)
    # 'auto': a scene's write budget over 3 x its threads, 8..256 MB (core/memory.py)
    region_size_mb: Union[float, str] = "auto"
    # Threads per conversion, each reading and writing one region at a time;
    # scenes converted at once share them. Was max_concurrency.
    max_threads: int = Field(default=4, ge=1)
    max_concurrent_scenes: int = Field(default=1, ge=1)
    # The memory the whole run may use on this machine: "16GB", or "auto" = half the
    # RAM free when it starts. Shared out by core/memory.py (processes, what each
    # already holds, reading, scenes). Replaces memory_per_worker (2026-10-08).
    memory_budget: str = "auto"
    # micro-reader's decoding memory per process: "auto" (what the file's decode unit
    # needs, core/memory.reader_share) or e.g. "1GB"
    decode_memory: str = "auto"
    # SLURM: each worker's (SLURM job's) memory; its conversions share it
    slurm_worker_memory: str = "3GB"
    # One conversion process's memory in MB, resolved once by the dispatcher before
    # the pool starts (not a user setting). None: worked out by the worker.
    process_memory_mb: Optional[float] = Field(default=None, gt=0)
    tensorstore_data_copy_concurrency: int = Field(default=4, ge=1)
    max_retries: int = Field(default=10, ge=0, le=100)
    # Bio-Formats read threads (dask's pool for its tile reads). Was
    # bf_read_concurrency.
    bioformats_threads: Optional[int] = Field(default=4, ge=1)
    # micro-reader's decode threads per worker process when
    # readers.pixel_reader='micro' (its region reads share one pool). Was
    # micro_read_concurrency.
    decode_threads: int = Field(default=4, ge=1)
    # DEPRECATED, ignored.  Bio-Formats images are now read one writer region
    # at a time, so region_size_mb sets the read size and the JVM-heap limit is
    # applied automatically.  Still accepted so existing configs, saved batches
    # and scripts keep working: retired settings warn, never fail (Bugra,
    # 2026-10-05).
    bf_tile_size_mb: Optional[float] = Field(default=None, gt=0.0)
    jvm_memory: Optional[str] = "1g"

    @model_validator(mode="before")
    @classmethod
    def _renamed(cls, data):
        return rename_settings(data) if isinstance(data, dict) else data

    @field_validator('memory_budget', 'decode_memory', 'slurm_worker_memory', mode='before')
    @classmethod
    def _memory_size(cls, v, info: ValidationInfo):
        from eubi_bridge.core.memory import is_auto, parse_mb
        if info.field_name != 'slurm_worker_memory' and is_auto(v):
            return "auto"
        if isinstance(v, (int, float)):
            v = f"{v}MB"
        parse_mb(v)                                   # a clear error for a bad size
        return str(v).strip()

    @field_validator('region_size_mb', mode='before')
    @classmethod
    def _region_size(cls, v):
        from eubi_bridge.core.memory import is_auto
        if is_auto(v):
            return "auto"
        mb = float(v)
        if mb <= 0:
            raise ValueError(f"region_size_mb must be > 0 or 'auto', got {v!r}")
        return mb

    @field_validator('bf_tile_size_mb')
    @classmethod
    def _warn_bf_tile_size_deprecated(cls, v):
        global _bf_tile_size_deprecation_warned
        if v is not None and not _bf_tile_size_deprecation_warned:
            _bf_tile_size_deprecation_warned = True
            _logger.warning(
                "DEPRECATION: 'bf_tile_size_mb' is no longer used and is ignored. "
                "Bio-Formats images are read one region at a time, so "
                "'region_size_mb' now sets the read size. Remove it from your "
                "config, batch or script; it will be rejected in a future release.")
        return v

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
    slurm_sif_path: Optional[str] = None   # Apptainer SIF for workers

    @field_validator("on_local_cluster")
    @classmethod
    def _retired(cls, v, info: ValidationInfo):
        warn_retired(info.field_name, v)
        return None
    max_concurrent_downscale_layers: int = Field(default=3, gt=0)


# ---------------------------------------------------------------------------
# ReaderConfig
# ---------------------------------------------------------------------------

class ReaderConfig(BaseModel):
    """Reader / series-selection parameters."""

    model_config = ConfigDict(extra="ignore")

    as_mosaic: bool = False
    # A file micro-reader reads as a plate (Harmony, CellVoyager, MetaXpress, IN Cell, a
    # CZI with a plate) becomes ONE NGFF plate store; False: one store per image, with a
    # warning that the file is a plate.
    as_plate: bool = True
    # Accept int, 'all', or comma-separated ints (e.g. '0,1,2')
    view_index: Union[int, str] = 0
    phase_index: int = Field(default=0, ge=0)
    # Accept int, 'all', or comma-separated ints (e.g. '0,1')
    illumination_index: Union[int, str] = 0
    # Accept int, 'all', or comma-separated ints (e.g. '0,2,4'). Was scene_index
    # (RENAMED_INDEX_SETTINGS: still accepted, with a FutureWarning).
    series_index: Union[int, str] = 0
    rotation_index: int = Field(default=0, ge=0)
    mosaic_tile_index: Union[int, str, None] = None
    sample_index: int = Field(default=0, ge=0)
    force_bioformats: bool = False
    # Where pixels come from: 'micro' (micro-reader, the default; falls back to
    # the standard reader, with a warning, for a file it does not read) or
    # 'standard' (the format's usual reader).
    pixel_reader: Literal['standard', 'micro'] = 'micro'
    # With pixel_reader='micro': also open the standard reader and compare its
    # pixels with micro-reader's (slower; for checking a new format).  Off:
    # micro-reader is the reader, checked against the metadata only.
    micro_check_pixels: bool = False
    # When True, arrays for each view / illumination are stacked along the
    # channel axis rather than written as separate OME-Zarr outputs.
    concat_views: bool = False
    concat_illuminations: bool = False

    @model_validator(mode="before")
    @classmethod
    def _renamed(cls, data):
        return rename_settings(data) if isinstance(data, dict) else data

    @field_validator('series_index', mode='before')
    @classmethod
    def _validate_series_index(cls, v):
        if isinstance(v, int) or v == 'all':
            return v
        if isinstance(v, (tuple, list)):
            parts = [int(x) for x in v]
            return parts[0] if len(parts) == 1 else ','.join(str(x) for x in parts)
        if isinstance(v, str):
            try:
                parts = [int(x.strip()) for x in v.split(',')]
                return parts[0] if len(parts) == 1 else v
            except ValueError:
                raise ValueError(f"series_index must be an int, 'all', or comma-separated ints; got {v!r}")
        return v

    @field_validator('view_index', 'illumination_index', mode='before')
    @classmethod
    def _validate_multi_index(cls, v):
        if isinstance(v, int) or v == 'all':
            return v
        if isinstance(v, (tuple, list)):
            parts = [int(x) for x in v]
            return parts[0] if len(parts) == 1 else ','.join(str(x) for x in parts)
        if isinstance(v, str):
            try:
                parts = [int(x.strip()) for x in v.split(',')]
                return parts[0] if len(parts) == 1 else v
            except ValueError:
                raise ValueError(
                    f"view_index / illumination_index must be an int, 'all', or "
                    f"comma-separated ints; got {v!r}"
                )
        return v

    @field_validator('mosaic_tile_index', mode='before')
    @classmethod
    def _validate_mosaic_tile_index(cls, v):
        if v is None or isinstance(v, int) or v == 'all':
            return v
        if isinstance(v, (tuple, list)):
            parts = [int(x) for x in v]
            return parts[0] if len(parts) == 1 else ','.join(str(x) for x in parts)
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
    # OME-Zarr (NGFF) version, e.g. '0.4' or '0.5'.  Supersedes zarr_format when
    # set to a recognised version; None falls back to zarr_format (deprecated).
    ome_zarr_version: Optional[str] = None
    zarr_format: Literal[2, 3] = 2
    # DEPRECATED, ignored (2026-10-05).  micro-reader, the default reader,
    # creates no dask arrays, so there is nothing to skip; the option only
    # sent TIFFs past micro-reader, to eubi's own TIFF route.  Still accepted
    # so existing configs, saved batches and scripts keep working.
    skip_dask: Optional[bool] = None
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
    compressor: Optional[str] = "blosc"
    compressor_params: dict = Field(default_factory=dict)
    overwrite: bool = False
    save_omexml: bool = True
    # Write acquisition metadata NGFF has no field for (view/illumination
    # indices, objective NA, per-channel emission) into the namespaced
    # 'eubi_bridge' attrs block.  None = auto: on when the conversion splits an
    # acquisition into several containers, since that is when the relationship
    # between the pieces would otherwise be lost.
    export_acquisition_metadata: Optional[bool] = None
    squeeze: bool = True
    # 'auto' is the on-disk sentinel; _collect_params converts it to None at
    # runtime — we accept both here so JSON round-trips work unchanged.
    dtype: Optional[str] = "auto"
    # RETIRED in 0.2.0 (roadmap step 6): ome_zarr_pyramid writes every output.
    # Still accepted, so saved configs, batches and scripts keep working; 'eubi'
    # (eubi-bridge's own writer, moved to legacy_0_1/) warns once and writes with
    # ome_zarr_pyramid.
    writer: Literal['auto', 'eubi', 'ozp'] = 'auto'
    # The bytes backend of the ozp writer (dyna-zarr): TensorStore, or zarrista (Rust,
    # zarrs; `pip install "dyna-zarr[zarrista]"`, which installs zarrista-bounded). With
    # either, shards larger than the region budget are written chunk by chunk in bounded
    # memory (local outputs). eubi-bridge's own writer only writes with TensorStore.
    io_backend: Literal['tensorstore', 'zarrista'] = 'tensorstore'

    @field_validator("compressor_params", mode="before")
    @classmethod
    def _parse_compressor_params(cls, v):
        """Accept a JSON object string as well as a dict.

        A conversion table carries one value per cell, so ``compressor_params``
        can only travel as text.  Accepting JSON here lets a CSV row override the
        codec *and* its parameters together — they are meaningless apart, since
        e.g. ``GZip(cname='lz4')`` raises TypeError.
        """
        if v is None or v == "":
            return {}
        if isinstance(v, str):
            import json
            try:
                parsed = json.loads(v)
            except ValueError as exc:
                raise ValueError(
                    f"compressor_params must be a JSON object, got {v!r} ({exc})"
                ) from None
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"compressor_params must be a JSON object, got {type(parsed).__name__}"
                )
            return parsed
        return v

    @field_validator("compressor")
    @classmethod
    def _validate_compressor(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
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

    @field_validator("writer", mode="before")
    @classmethod
    def _retired_writer(cls, v):
        """writer='eubi' (retired in 0.2.0): warned about once, written by ozp."""
        if isinstance(v, str) and v.strip().lower() == "eubi":
            if "writer" not in _retired_warned:
                _retired_warned.add("writer")
                _logger.warning(
                    "DEPRECATION: writer='eubi' is no longer available: eubi-bridge 0.2.0 "
                    "writes every output with ome_zarr_pyramid (same pixels). Remove "
                    "'writer' from your config, batch or script; it will be rejected in "
                    "a future release.")
            return "ozp"
        return v

    @field_validator("export_acquisition_metadata", mode="before")
    @classmethod
    def _tristate_words(cls, v):
        """'auto' / 'on' / 'off', as the GUI and a batch cell spell the three states."""
        if isinstance(v, str):
            return {"auto": None, "on": True, "off": False}.get(v.strip().lower(), v)
        return v

    @field_validator("skip_dask")
    @classmethod
    def _retired(cls, v):
        warn_retired("skip_dask", v)
        return None

    @field_validator("ome_zarr_version", mode="before")
    @classmethod
    def _coerce_ome_zarr_version(cls, v):
        # Fire/JSON may deliver a numeric like 0.5; normalise to a string.
        if v is None:
            return None
        return str(v)

    @model_validator(mode="after")
    def _resolve_ome_zarr_version(self) -> "ConversionConfig":
        """Resolve ome_zarr_version / zarr_format into a consistent pair.

        ome_zarr_version supersedes zarr_format; an unset version falls back to
        zarr_format, an unknown one raises.  Resolution here is SILENT — the deprecation
        warning is emitted once at the conversion entry point (to_zarr), so it
        reflects the user's actual choice rather than every model build.  Runs
        before the compressor check so it validates against the effective format.
        """
        version, fmt = resolve_ome_zarr_target(self.ome_zarr_version, self.zarr_format,
                                               warn=False)
        self.ome_zarr_version = version
        self.zarr_format = fmt
        return self

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
            _check_blosc_cname(cname)
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
    keep_existing_resolutions: bool = False
    # Smart (isotropic) downscaling: when enabled, the first pyramid level uses
    # automatically-computed factors to make voxels as isotropic as possible.
    # Per-axis overrides may be given (None = auto).  Matches the GUI controls.
    apply_smart_downscaling: bool = False
    time_smart_scale_factor: Optional[int] = None
    z_smart_scale_factor: Optional[int] = None
    y_smart_scale_factor: Optional[int] = None
    x_smart_scale_factor: Optional[int] = None

    @field_validator("n_layers")
    @classmethod
    def _validate_n_layers(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 1:
            raise ValueError(f"n_layers must be ≥ 1, got {v}.")
        return v


# ---------------------------------------------------------------------------
# ConcatenationConfig
# ---------------------------------------------------------------------------

class MetadataConfig(BaseModel):
    """How the output's metadata is read and described.

    These were kept under ``conversion`` while the GUI has always presented them
    on a Metadata tab, and ``channel_colors`` / ``channel_labels`` had no model
    at all -- they survived in a config file only because unknown keys were
    persisted verbatim.  Giving them a section makes the file say what the
    interface says, and makes the two colour settings first-class rather than
    accidental.

    ``save_omexml`` and ``export_acquisition_metadata`` deliberately stay in
    ``conversion``: they decide what the writer emits rather than describing the
    image, and moving them would break more existing configs for less gain.
    """

    model_config = ConfigDict(extra="ignore")

    # 'micro' (the default): micro-reader reads metadata and pixels, and a file
    # it does not read falls back to Bio-Formats -- the JVM starts only then.
    metadata_reader: str = "micro"
    override_channel_names: bool = False
    channel_intensity_limits: Literal["from_dtype", "from_array", "auto"] = "from_dtype"
    # "idx,RRGGBB;..." and "idx,label;..." as the CLI accepts them.  Empty means
    # every channel keeps its source colour/name, or is given one automatically.
    channel_colors: str = ""
    channel_labels: str = ""

    # Physical pixel size and unit per axis.  None means "keep whatever the
    # source file says", which is both the old behaviour and what a blank cell
    # in a conversion table resolves to -- so no 'auto' sentinel is needed here,
    # unlike dtype/n_layers where None already means "compute it for me".
    # There is deliberately no channel_scale/channel_unit: a channel has no
    # physical extent.
    time_scale: Optional[float] = Field(default=None, gt=0)
    z_scale:    Optional[float] = Field(default=None, gt=0)
    y_scale:    Optional[float] = Field(default=None, gt=0)
    x_scale:    Optional[float] = Field(default=None, gt=0)
    time_unit:  Optional[str] = None
    z_unit:     Optional[str] = None
    y_unit:     Optional[str] = None
    x_unit:     Optional[str] = None

    @field_validator('time_scale', 'z_scale', 'y_scale', 'x_scale', mode='before')
    @classmethod
    def _one_number_notation(cls, v, info: ValidationInfo):
        return parse_scale(v, info.field_name)


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
    # Names the concatenated output and, in a conversion table, decides which
    # rows are concatenated together.  A single run has one group, so setting it
    # here only prefixes the output name; the grouping meaning appears when a
    # table gives different rows different values.
    aggregative_group: Optional[str] = None


# ---------------------------------------------------------------------------
# CompressorConfig  (replaces the writers.py dataclass)
# ---------------------------------------------------------------------------

class CompressorConfig(BaseModel):
    """Compressor name + parameter bundle.

    Validates the name against the union of v2 and v3 supported codecs.
    When zarr_format is known, use ConversionConfig for the stricter per-format check.
    """

    model_config = ConfigDict(extra="ignore")

    name: Optional[str] = "blosc"
    params: dict = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
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
        if self.name and self.name.lower() == "blosc" and self.params:
            cname = self.params.get("cname", "lz4")
            _check_blosc_cname(cname)
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
    | set(MetadataConfig.model_fields)
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
    # Computed, never user-supplied: the basename for the output store without
    # its '.zarr' suffix.  Set only when two inputs would otherwise produce the
    # same store name; None keeps the default of naming it after the input file.
    resolved_basename: Optional[str] = None
    cluster: ClusterConfig    = Field(default_factory=ClusterConfig)
    readers: ReaderConfig     = Field(default_factory=ReaderConfig)
    conversion: ConversionConfig = Field(default_factory=ConversionConfig)
    downscale: DownscaleConfig   = Field(default_factory=DownscaleConfig)
    metadata: MetadataConfig     = Field(default_factory=MetadataConfig)
    extra: dict               = Field(default_factory=dict)

    @classmethod
    def from_kwargs(
        cls,
        input_path: str,
        output_path: str,
        kwargs: dict,
        resolved_basename: Optional[str] = None,
    ) -> "ConversionJob":
        """Build a validated ``ConversionJob`` from a flat merged-kwargs dict.

        Each config section is validated independently (``extra='ignore'`` on
        the sub-models silently drops unknown keys).  Keys not belonging to any
        model go into ``extra`` for passthrough to the worker. Renamed settings are
        renamed first, so an old name never lands in ``extra``.
        """
        kwargs = rename_settings(kwargs)
        return cls(
            input_path=input_path,
            output_path=output_path,
            resolved_basename=resolved_basename,
            cluster=ClusterConfig(**kwargs),
            readers=ReaderConfig(**kwargs),
            conversion=ConversionConfig(**kwargs),
            downscale=DownscaleConfig(**kwargs),
            metadata=MetadataConfig(**kwargs),
            extra={k: v for k, v in kwargs.items() if k not in _CONFIG_MODEL_KEYS},
        )

    def to_worker_kwargs(self) -> dict:
        """Flatten back to the dict that ``unary_worker_sync`` expects."""
        result: dict = {}
        result.update(self.cluster.model_dump())
        result.update(self.readers.model_dump())
        result.update(self.conversion.model_dump())
        result.update(self.downscale.model_dump())
        result.update(self.metadata.model_dump())
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
    # Names this group of files in a mixed conversion table, and is prefixed to
    # the output name so several groups from one run stay distinguishable.
    # None means no prefix, which is what every conversion did before the
    # column existed.
    aggregative_group: Optional[str] = None
    cluster: ClusterConfig       = Field(default_factory=ClusterConfig)
    readers: ReaderConfig        = Field(default_factory=ReaderConfig)
    conversion: ConversionConfig = Field(default_factory=ConversionConfig)
    downscale: DownscaleConfig   = Field(default_factory=DownscaleConfig)
    metadata: MetadataConfig     = Field(default_factory=MetadataConfig)

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
        kwargs = rename_settings(kwargs)
        agg_keys = _AGGREGATIVE_KEYS
        return cls(
            input_path=input_path,
            output_path=output_path,
            cluster=ClusterConfig(**kwargs),
            readers=ReaderConfig(**kwargs),
            conversion=ConversionConfig(**kwargs),
            downscale=DownscaleConfig(**kwargs),
            metadata=MetadataConfig(**kwargs),
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
        result.update(self.metadata.model_dump())
        result.update(self.extra)
        # Carried through so the writer can prefix the derived output name,
        # keeping several groups from one run distinguishable.
        if self.aggregative_group:
            result["aggregative_group"] = self.aggregative_group
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

class MetadataUpdateConfig(BaseModel):
    """Parameters for pixel-metadata and channel-metadata update operations.

    All scale / unit fields default to ``None`` meaning "keep the value already
    stored in the file".  An update edits metadata only: squeeze and crop
    change the arrays, so they belong to conversion, and callers that pass
    them (whole config sections) have them ignored like any other extra key.
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

    @field_validator('time_scale', 'channel_scale', 'z_scale', 'y_scale', 'x_scale',
                     mode='before')
    @classmethod
    def _one_number_notation(cls, v, info: ValidationInfo):
        return parse_scale(v, info.field_name)

    # Misc update options
    save_omexml: bool = True
    series:      Union[int, str] = "all"

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
