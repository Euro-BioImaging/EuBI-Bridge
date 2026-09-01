"""
Batch model: a queue of one-to-one conversions persisted as a CSV table.

A batch is a folder holding two files:

    batch.csv           one row per input file; sparse per-row overrides
    batch_config.json   snapshot of the global GUI config the rows diff against

``eubi to_zarr <batch.csv>`` already understands this format: ``take_filepaths``
reads the table, resolves ``input_path`` / ``output_path`` relative to the CSV's
own directory, and ``ebridge.to_zarr`` merges each row over the global config
(blank cells fall through to the global value).  This module only has to produce
a table that obeys those rules. It adds no new format.

Two constraints come from that CLI path and are enforced here:

* Table input is unary-only.  ``take_filepaths`` raises outright when
  ``concatenation_axes`` is set, so aggregative jobs cannot be batched
  (see :func:`can_batch`).
* Only scalar values survive a CSV round-trip.  Dicts (``compressor_params``)
  and range tuples cannot be per-row overrides; they stay in the snapshot and a
  row that would need to change one is reported via :meth:`BatchModel.add`.
"""
from __future__ import annotations

import csv
import json
import os
from copy import deepcopy
from typing import Any, Iterable, NamedTuple

from eubi_bridge.qt_gui.workers.conversion_worker import _build_kwargs

# Columns that are addressed positionally rather than as parameter overrides.
_PATH_COLUMNS = ("input_path", "output_path")

# Parameters that cannot be expressed in a single CSV cell.  They are carried by
# the config snapshot instead; a row is never allowed to override them.
#
# ``compressor`` / ``compressor_params`` are deliberately absent: the params dict
# is written as JSON and parsed back by ConversionConfig, so the codec and its
# parameters can vary per row.  They must always be emitted *together* (see
# _COUPLED_KEYS) because e.g. ``GZip(cname='lz4')`` raises TypeError.
_NON_ROW_OVERRIDABLE = frozenset({
    "time_range", "channel_range", "z_range", "y_range", "x_range",
})

# Groups of keys that are only meaningful as a set.  If any member differs from
# the baseline, every member is written to the row so they can never desynchronise.
_COUPLED_KEYS: tuple[frozenset[str], ...] = (
    frozenset({"compressor", "compressor_params"}),
)

# Members of a coupled group are stored even when they have no editor of their
# own. compressor_params rides along with compressor and must reach the CSV.
_COUPLED_ALL: frozenset[str] = frozenset().union(*_COUPLED_KEYS)

# Coupled members with no editor of their own.  They belong in the CSV but not
# in the queue view, where they would be an uneditable column.
_COUPLED_RIDERS: frozenset[str] = frozenset({"compressor_params"})

# Keys that _build_kwargs() emits only conditionally but whose absence is
# harmless, because another key the row *can* override already governs them.
# ``target_chunk_mb`` is only read when ``auto_chunk`` is true, and ``auto_chunk``
# is itself a per-row override, so inheriting a stale value changes nothing.
_HARMLESS_WHEN_ABSENT = frozenset({"target_chunk_mb"})

# Parameters that describe the run itself rather than any one conversion.  One
# batch runs in a single process against a single cluster, so these can only ever
# take the baseline's value; offering them per row would be meaningless.
_CLUSTER_KEYS = frozenset({
    "max_workers", "queue_size", "region_size_mb", "max_concurrency",
    "max_concurrent_downscale_layers", "max_concurrent_scenes",
    "memory_per_worker", "bf_tile_size_mb", "jvm_memory", "bf_read_concurrency",
    "on_local_cluster", "on_slurm", "slurm_partition", "slurm_account",
    "slurm_time", "slurm_sif_path", "slurm_worker_timeout",
})


class ParamSpec(NamedTuple):
    """How to render an editor for one row-overridable parameter.

    Deliberately UI-toolkit agnostic: the batch model owns the description of
    what a parameter *is*, and the Qt layer decides which widget expresses it.

    *tab* and *group* locate the parameter on the conversion form: the tab it
    lives on, and the group box within that tab (empty when it sits loose on the
    tab).  Both the Edit Cells dialog and the queue table order themselves by the
    declaration order of :data:`_PARAM_SPECS`, so parameters appear where the
    user already expects them rather than alphabetically.
    """

    key: str
    label: str
    kind: str                       # bool | int | float | choice | text
    choices: tuple[str, ...] = ()
    minimum: float = 0
    maximum: float = 1_000_000
    tooltip: str = ""
    tab: str = ""
    group: str = ""
    # Parent parameter this one is subordinate to, and the parent value(s) that
    # make it active.  ``auto_chunk=True`` means the writer computes chunks, so
    # the manual per-axis sizes are inert; the editor greys them out and drops
    # any override, rather than leaving a value that looks applied but is not.
    # ``active_when`` is compared against the parent's value after canonicalising
    # both, so it works for booleans and for choices alike (sharding needs
    # ome_zarr_version == '0.5').
    depends_on: str = ""
    active_when: Any = True


_INDEX_TIP = ("'all', a single index, or a comma-separated list such as '0,2,3'.")

# The parameters an Edit Cells dialog may offer.  Anything absent is either
# non-overridable (_NON_ROW_OVERRIDABLE), cluster-wide (_CLUSTER_KEYS), or has no
# sensible single-cell editor; those keep the baseline value.
_PARAM_SPECS: tuple[ParamSpec, ...] = (
    # ---- Paths ----
    # Editable like any other cell, but with a browse button: these are among
    # the most frequently corrected fields, and retyping a long path by hand is
    # both slow and error prone.  'file' picks an existing input (a directory
    # too, since an OME-Zarr store is one); 'directory' picks the output folder.
    ParamSpec("input_path", "Input path", "file", tab="Paths"),
    ParamSpec("output_path", "Output path", "directory", tab="Paths"),

    # ---- Reader tab ----
    ParamSpec("scene_index", "Scene index", "text", tooltip=_INDEX_TIP,
              tab="Reader", group="Scenes"),
    ParamSpec("mosaic_tile_index", "Mosaic tile index", "text",
              tooltip=_INDEX_TIP, tab="Reader", group="Tiles"),
    ParamSpec("as_mosaic", "Read as mosaic", "bool",
              tab="Reader", group="Tiles"),
    ParamSpec("view_index", "View index", "text", tooltip=_INDEX_TIP,
              tab="Reader", group="Views"),
    ParamSpec("concat_views", "Concatenate views", "bool",
              tab="Reader", group="Views"),
    ParamSpec("illumination_index", "Illumination index", "text",
              tooltip=_INDEX_TIP, tab="Reader", group="Illuminations"),
    ParamSpec("concat_illuminations", "Concatenate illuminations", "bool",
              tab="Reader", group="Illuminations"),
    ParamSpec("phase_index", "Phase index", "text", tooltip=_INDEX_TIP,
              tab="Reader", group="Other Indices (Experimental)"),
    ParamSpec("rotation_index", "Rotation index", "text", tooltip=_INDEX_TIP,
              tab="Reader", group="Other Indices (Experimental)"),
    ParamSpec("sample_index", "Sample index", "text", tooltip=_INDEX_TIP,
              tab="Reader", group="Other Indices (Experimental)"),
    ParamSpec("force_bioformats", "Force Bio-Formats", "bool", tab="Reader"),

    # ---- Conversion tab (group order follows the form) ----
    ParamSpec("compressor", "Compressor", "choice",
              ("blosc", "gzip", "zstd", "bz2", "none"),
              tab="Conversion", group="Compression"),
    ParamSpec("auto_chunk", "Auto chunk", "bool",
              tab="Conversion", group="Chunking"),
    ParamSpec("target_chunk_mb", "Target chunk (MB)", "float", minimum=0.001,
              maximum=100_000, tooltip="Only used when Auto chunk is on.",
              tab="Conversion", group="Chunking",
              depends_on="auto_chunk", active_when=True),
    *(ParamSpec(f"{ax}_chunk", f"Chunk {ax}", "int", minimum=1,
                maximum=1_000_000, tooltip="Only used when Auto chunk is off.",
                tab="Conversion", group="Chunking",
                depends_on="auto_chunk", active_when=False)
      for ax in ("time", "channel", "z", "y", "x")),
    ParamSpec("ome_zarr_version", "OME-Zarr version", "choice", ("0.4", "0.5"),
              tab="Conversion", group="OME-Zarr Version and Sharding"),
    # Sharding exists only in zarr v3, which OME-Zarr 0.5 selects; under 0.4 the
    # writer has no shard concept at all, so the coefficients do nothing.
    *(ParamSpec(f"{ax}_shard_coef", f"Shard coef {ax}", "int", minimum=1,
                maximum=1_000,
                tab="Conversion", group="OME-Zarr Version and Sharding",
                depends_on="ome_zarr_version", active_when="0.5")
      for ax in ("time", "channel", "z", "y", "x")),
    # Loose on the Conversion tab, with no group box on the form either.
    ParamSpec("dtype", "Data type", "choice",
              ("auto", "uint8", "uint16", "uint32", "int8", "int16", "int32",
               "float32", "float64"), tab="Conversion"),
    ParamSpec("overwrite", "Overwrite existing", "bool", tab="Conversion"),
    ParamSpec("squeeze", "Squeeze dimensions", "bool", tab="Conversion"),
    ParamSpec("save_omexml", "Save OME-XML", "bool", tab="Conversion"),
    ParamSpec("override_channel_names", "Override channel names", "bool",
              tab="Conversion"),
    ParamSpec("skip_dask", "Skip dask", "bool", tab="Conversion"),
    ParamSpec("verbose", "Verbose", "bool", tab="Conversion"),

    # ---- Downscaling tab ----
    # 'auto' is a value the row states, not an absence: it means auto-detect the
    # layer count regardless of the config, whereas a blank cell falls back to
    # the config's setting.  Mirrors the existing dtype='auto' sentinel, which
    # _normalise_row_overrides() converts to None for the CLI.
    ParamSpec("n_layers", "Resolution layers", "auto_int", minimum=1,
              maximum=20,
              tooltip="'auto' detects the layer count; blank uses the config.",
              tab="Downscaling",
              depends_on="keep_existing_resolutions", active_when=False),
    # Only consulted when the layer count is auto-detected: update_downscaler()
    # calls calculate_n_layers(shape, scale_factor, min_dimension_size) solely
    # when n_layers is None/'auto'.
    ParamSpec("min_dimension_size", "Min dimension size", "int", minimum=1,
              maximum=1_000_000, tab="Downscaling",
              depends_on="n_layers", active_when="auto"),
    ParamSpec("downscale_method", "Downscale method", "choice",
              ("simple", "mean", "median", "gaussian"), tab="Downscaling",
              depends_on="keep_existing_resolutions", active_when=False),
    ParamSpec("keep_existing_resolutions", "Keep existing resolutions", "bool",
              tab="Downscaling"),
    ParamSpec("apply_smart_downscaling", "Apply smart downscaling", "bool",
              tab="Downscaling"),
    *(ParamSpec(f"{ax}_scale_factor", f"Scale factor {ax}", "int", minimum=1,
                maximum=1_000,
                tab="Downscaling", group="Scale Factors per Dimension",
                depends_on="keep_existing_resolutions", active_when=False)
      for ax in ("time", "channel", "z", "y", "x")),

    *(ParamSpec(f"{ax}_smart_scale_factor", f"Smart scale {ax}", "int",
                minimum=1, maximum=1_000,
                tooltip="Only used when Apply smart downscaling is on.",
                tab="Downscaling", group="Scale Factors per Dimension",
                depends_on="apply_smart_downscaling", active_when=True)
      for ax in ("time", "z", "y", "x")),

    # ---- Metadata tab ----
    ParamSpec("channel_intensity_limits", "Channel intensity limits", "choice",
              ("from_dtype", "from_array"), tab="Metadata"),
    ParamSpec("metadata_reader", "Metadata reader", "choice",
              ("bioio", "bfio", "bioformats"), tab="Metadata"),
    # Edited as raw text here rather than with a colour picker: a batch row is
    # one cell, and the CLI's own format is the clearest thing to show.
    ParamSpec("channel_colors", "Channel colours", "text", tab="Metadata",
              tooltip="'idx,RRGGBB' pairs separated by ';', e.g. "
                      "'0,FF0000;1,00FF00'.  Channels not listed are "
                      "coloured automatically."),
    # Physical scale overrides. Emitted only when the form enables them, but a
    # row may still carry one, so each needs an editor.
    *(ParamSpec(f"{ax}_scale", f"{ax.capitalize()} scale", "float",
                minimum=0.0, maximum=1_000_000, tab="Metadata",
                group="Physical Scale")
      for ax in ("time", "z", "y", "x")),
    *(ParamSpec(f"{ax}_unit", f"{ax.capitalize()} unit", "text",
                tab="Metadata", group="Physical Scale")
      for ax in ("time", "z", "y", "x")),
)

# Declaration order is the canonical presentation order. It mirrors the
# conversion form's tabs (Reader, Conversion, Downscaling, Metadata) and the
# group boxes within them.
_PARAM_ORDER: dict[str, int] = {
    spec.key: i for i, spec in enumerate(_PARAM_SPECS)}

_SPEC_BY_KEY: dict[str, ParamSpec] = {spec.key: spec for spec in _PARAM_SPECS}


# Placeholder column inserted between two categories.  A narrow blank gutter
# reads as a boundary far more clearly than a change of header text alone, which
# matters once the table is dozens of columns wide.
# Not a valid parameter name, so it can never collide with a real column.
SEPARATOR = "__separator__"


def with_separators(columns: list[str]) -> list[str]:
    """Insert a :data:`SEPARATOR` wherever the category changes.

    Never leading or trailing, and never doubled, so the table gains one thin
    gutter per boundary and nothing else.
    """
    out: list[str] = []
    previous = None
    for key in columns:
        tab = column_header(key)[0]
        if previous is not None and tab != previous:
            out.append(SEPARATOR)
        out.append(key)
        previous = tab
    return out


def parameter_tabs() -> list[str]:
    """The parameter categories, in the order the conversion form presents them.

    'Paths' is excluded: those columns are always shown and are not a category
    the user can toggle away.
    """
    tabs: list[str] = []
    for spec in _PARAM_SPECS:
        if spec.tab and spec.tab != "Paths" and spec.tab not in tabs:
            tabs.append(spec.tab)
    return tabs


def spec_for(key: str) -> ParamSpec | None:
    """The editor description for *key*, or None when it has no cell editor."""
    return _SPEC_BY_KEY.get(key)


def uneditable_reason(key: str) -> str | None:
    """Why *key* cannot be edited per row, or None when it can be."""
    if key in _PATH_COLUMNS:
        return None
    if key in _NON_ROW_OVERRIDABLE:
        return ("applies to the whole batch: set it before adding rows, "
                "or start a separate batch")
    if key in _CLUSTER_KEYS:
        return "a cluster setting: one batch runs as a single job"
    if key == "compressor_params":
        return "edited together with 'compressor'"
    if key == "zarr_format":
        return "derived from 'OME-Zarr version': set that instead"
    if key not in _SPEC_BY_KEY:
        return "not editable per row"
    return None


def sort_keys(keys: Iterable[str]) -> list[str]:
    """Order *keys* as the conversion form presents them.

    Path columns lead, then parameters in :data:`_PARAM_SPECS` order.  Anything
    with no spec keeps a stable alphabetical position at the end, so an unknown
    column is still rendered rather than dropped.
    """
    def rank(key: str) -> tuple[int, float, str]:
        if key in _PATH_COLUMNS:
            return (0, _PATH_COLUMNS.index(key), "")
        if key in _PARAM_ORDER:
            return (1, _PARAM_ORDER[key], "")
        return (2, 0, key)

    return sorted(keys, key=rank)


def grouped_specs(keys: Iterable[str]) -> list[tuple[str, list[tuple[str, list[ParamSpec]]]]]:
    """*keys* nested as ``[(tab, [(group, [spec, ...]), ...]), ...]``.

    Mirrors the conversion form's own hierarchy: a tab holds group boxes, and a
    parameter that sits loose on a tab (no group box on the form) comes back
    under the empty group ``""`` so the caller can place it directly on the tab.

    Only the tabs and groups the selection actually touches are returned, but
    each keeps its position from :data:`_PARAM_SPECS`, so selecting x/y chunks
    and a downscaling flag yields Conversion(Chunking(...)) then Downscaling(...),
    never an alphabetical flattening.
    """
    tabs: dict[str, dict[str, list[ParamSpec]]] = {}
    for key in sort_keys(keys):
        spec = _SPEC_BY_KEY.get(key)
        if spec is None:
            continue
        groups = tabs.setdefault(spec.tab or "Other", {})
        groups.setdefault(spec.group, []).append(spec)
    return [(tab, list(groups.items())) for tab, groups in tabs.items()]


def column_header(key: str) -> tuple[str, str, str]:
    """Header text for column *key* as ``(tab, group, label)``.

    A flat table header cannot span columns, so the queue puts the group on its
    own line above the parameter label and leaves the tab to be conveyed by the
    header tint.  Path columns carry no hierarchy and return empty strings.
    """
    spec = _SPEC_BY_KEY.get(key)
    if spec is None:
        return "", "", key
    return spec.tab, spec.group, spec.label


# GUI-config locations backing the parameters in _NON_ROW_OVERRIDABLE.
#
# Because no row can carry these, whatever the baseline holds is what the whole
# batch runs with.  Taking the baseline wholesale from the config file would
# therefore discard the user's current compression / range settings outright, so
# these specific spots are pinned from the live UI when the baseline is captured.
_UI_PINNED_PATHS: tuple[tuple[str, str], ...] = (
    ("conversion", "dimRangeTime"),
    ("conversion", "dimRangeChannel"),
    ("conversion", "dimRangeZ"),
    ("conversion", "dimRangeY"),
    ("conversion", "dimRangeX"),
)


def make_baseline(persisted: dict, ui_config: dict) -> dict:
    """Build the batch baseline: the saved config, plus the UI's pinned settings.

    Using the saved config as the anchor keeps the first row's deliberate
    changes visible as overrides.  Overlaying the pinned paths keeps settings
    that cannot vary per row (compression above all) actually applied.
    """
    base = deepcopy(persisted)
    for section, key in _UI_PINNED_PATHS:
        ui_section = ui_config.get(section)
        if isinstance(ui_section, dict) and key in ui_section:
            base.setdefault(section, {})[key] = deepcopy(ui_section[key])
    return base

DEFAULT_BATCH_NAME = "batch.csv"
CONFIG_SNAPSHOT_NAME = "batch_config.json"


# ── value serialisation ───────────────────────────────────────────────────────

def _is_csv_scalar(value: Any) -> bool:
    """True when *value* survives a write/read round-trip through a CSV cell."""
    return value is None or isinstance(value, (str, int, float, bool))


def to_cell(value: Any) -> str:
    """Render *value* into a CSV cell.

    Also used by the GUI table so the queue displays exactly what will be
    written to disk.

    Integer sequences become comma-joined strings (``[1, 2]`` -> ``"1,2"``),
    which the index validators in ``config_models`` accept directly.  ``None``
    becomes an empty cell, which ``ebridge.to_zarr`` skips so the global value
    applies.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, dict):
        # JSON, so ConversionConfig can parse it back out of the cell.
        # sort_keys keeps the rendering stable, so two equal dicts always
        # compare equal regardless of insertion order.
        return json.dumps(value, sort_keys=True)
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def _canonical(value: Any) -> Any:
    """Normalise a value so equal settings compare equal across representations.

    The same setting can reach us as a typed Python value (straight from the UI)
    or as text/float parsed back out of a CSV: ``128`` vs ``128.0`` vs
    ``"128"``, or ``[1, 2]`` vs ``"1,2"``.  Comparing raw values would mark
    unchanged cells as modified after a batch is reloaded.
    """
    if value is None or value == "":
        return ""
    if isinstance(value, bool):          # before int: bool is an int subclass
        return "True" if value else "False"
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.capitalize()
    try:
        return float(value)              # 128, 128.0 and "128" all collapse here
    except (TypeError, ValueError):
        pass
    return to_cell(value)                # dicts/sequences via their CSV form


def values_equal(a: Any, b: Any) -> bool:
    """True when two values mean the same setting."""
    return _canonical(a) == _canonical(b)


# Parameters whose ``None`` means "compute it automatically" rather than "unset".
# A CSV cannot write None (blank already means "use the config"), so these carry
# the literal string 'auto', exactly as ``dtype`` already does.
_AUTO_SENTINEL_KEYS = frozenset({"n_layers"})


def _to_sentinel(key: str, value: Any) -> Any:
    """Render *value* for a cell, turning an auto-meaning None into 'auto'."""
    if key in _AUTO_SENTINEL_KEYS and value is None:
        return "auto"
    return value


def from_sentinel(key: str, value: Any) -> Any:
    """Invert :func:`_to_sentinel`, so 'auto' becomes None for the CLI."""
    if key in _AUTO_SENTINEL_KEYS and isinstance(value, str)             and value.strip().lower() == "auto":
        return None
    return value


def default_compressor_params(compressor: Any) -> dict:
    """Parameters a freshly chosen *compressor* should start from.

    Codec parameters are not interchangeable: handing blosc's ``cname`` to
    ``GZip`` raises TypeError at write time.  Switching codec therefore has to
    replace the parameter dict, not inherit the previous codec's.
    """
    name = str(compressor or "").strip().lower()
    if name == "blosc":
        return {"cname": "lz4", "clevel": 5, "shuffle": 1}
    if name in ("gzip", "zstd", "bz2"):
        return {"level": 5}
    return {}


def _overridable(value: Any) -> bool:
    """True when *value* can be carried as a per-row override."""
    if _is_csv_scalar(value):
        return True
    # Dicts travel as JSON; ConversionConfig parses compressor_params back.
    if isinstance(value, dict):
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            return False
        return True
    # Sequences of ints round-trip as "1,2" and are understood by the
    # scene/view/illumination/tile index validators.
    return (isinstance(value, (list, tuple))
            and all(isinstance(v, int) for v in value))


# ── batch model ───────────────────────────────────────────────────────────────

class BatchModel:
    """An ordered queue of conversion rows plus the config they diff against.

    The first :meth:`add` captures the active GUI config as the batch baseline.
    Every later row stores only the parameters that differ from that baseline,
    so the CSV stays narrow and readable and rows inherit the rest at run time.
    """

    def __init__(self, full: bool = False) -> None:
        # Categories the user asked to see in full, over and above the columns
        # that appear because a row deviates from the config.
        self.shown_tabs: set[str] = set()
        self._rows: list[dict[str, Any]] = []
        self._base_config: dict | None = None
        # Config that highlighting compares against, and its cached flat form.
        self._compare_config: dict | None = None
        self._compare_kwargs: dict[str, Any] | None = None
        # Full table: write every parameter on every row instead of only the
        # deviations.  Rows become self-describing at the cost of width.
        self.full = full

    # -- state -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def rows(self) -> list[dict[str, Any]]:
        return self._rows

    @property
    def base_config(self) -> dict | None:
        return self._base_config

    def set_baseline(self, config: dict) -> None:
        """Fix the config that rows are diffed against.

        Must be set before the first :meth:`add`, otherwise that row is compared
        against itself and records no overrides at all.
        """
        self._base_config = deepcopy(config)
        self._compare_kwargs = None          # baseline may now be the comparison

    def baseline_summary(self) -> str:
        """One-line description of the settings that apply to the whole batch.

        These are the parameters no row can override, so they are invisible in
        the table and would otherwise be impossible to check before running.
        """
        if self._base_config is None:
            return ""
        conv = self._base_config.get("conversion", {}) or {}
        parts: list[str] = []

        ranges = [f"{ax}={conv.get('dimRange' + key)}"
                  for ax, key in (("T", "Time"), ("C", "Channel"),
                                  ("Z", "Z"), ("Y", "Y"), ("X", "X"))
                  if conv.get("dimRange" + key)]
        if ranges:
            parts.append("ranges: " + ", ".join(ranges))

        return " | ".join(parts)

    def clear(self) -> None:
        self._rows.clear()
        self._base_config = None

    def remove(self, index: int) -> None:
        del self._rows[index]
        if not self._rows:
            self._base_config = None

    def duplicate(self, index: int) -> None:
        self._rows.insert(index + 1, deepcopy(self._rows[index]))

    def move(self, index: int, delta: int) -> int:
        """Move a row by *delta* positions; returns its new index."""
        new_index = max(0, min(len(self._rows) - 1, index + delta))
        if new_index != index:
            self._rows.insert(new_index, self._rows.pop(index))
        return new_index

    # -- building --------------------------------------------------------

    def add(
        self,
        ui_config: dict,
        input_paths: Iterable[str],
        output_path: str,
    ) -> list[str]:
        """Append one row per input path.

        Returns a list of human-readable warnings, currently the parameters
        that differ from the baseline but cannot be stored per row.  Those keep
        the baseline value, so the caller should surface the warning rather than
        silently produce a batch that behaves unexpectedly.

        If no baseline was set, *ui_config* becomes it, which means this first
        row is diffed against itself and records nothing.  Callers should call
        :meth:`set_baseline` with the persisted config first so that settings
        the user changed before the first add are still visible as overrides.
        """
        if self._base_config is None:
            self._base_config = deepcopy(ui_config)

        base_kwargs = _build_kwargs(self._base_config)
        row_kwargs = _build_kwargs(ui_config)

        overrides: dict[str, Any] = {}
        blocked: list[str] = []

        # Walk the union of both key sets.  _build_kwargs() emits some keys only
        # conditionally (target_chunk_mb, the physical-scale overrides), so a key
        # the baseline has and this row drops is a real difference too, and one
        # a CSV cannot express, since a blank cell means "inherit", not "unset".
        for key in set(row_kwargs) | set(base_kwargs):
            if key not in row_kwargs:
                if key not in _HARMLESS_WHEN_ABSENT:
                    blocked.append(key)
                continue
            value = _to_sentinel(key, row_kwargs[key])
            if _to_sentinel(key, base_kwargs.get(key)) == value:
                continue
            # Anything without a per-row editor keeps the baseline value, so the
            # user must be told rather than have their change silently dropped.
            # Coupled riders are exempt: they travel with the key that owns them.
            if (key in _NON_ROW_OVERRIDABLE or not _overridable(value)
                    or (key not in _COUPLED_RIDERS
                        and uneditable_reason(key) is not None)):
                blocked.append(key)
                continue
            overrides[key] = value

        # Emit coupled keys as a unit: a row that changed the codec but inherited
        # the previous codec's parameters would raise TypeError at write time.
        for group in _COUPLED_KEYS:
            if group & set(overrides):
                for key in group:
                    if key not in overrides and key in row_kwargs:
                        overrides[key] = row_kwargs[key]

        # Rows keep every writable parameter, not just the deviations.  Sparse
        # vs full is then purely a rendering choice (see :meth:`columns` and
        # :meth:`cell`), so the view can be switched after rows were added.
        # Only keys a row can actually override are stored.  Cluster settings and
        # derived values would otherwise surface as full-table columns that
        # reject every edit: visible, unexplained, and unusable.
        storable = {k: _to_sentinel(k, v) for k, v in row_kwargs.items()
                    if k not in _NON_ROW_OVERRIDABLE and _overridable(v)
                    and (k in _COUPLED_ALL or uneditable_reason(k) is None)}

        for path in input_paths:
            self._rows.append({
                "input_path": path,
                "output_path": output_path,
                **storable,
            })

        return blocked

    def common_value(self, row_indices: Iterable[int], key: str) -> tuple[Any, bool]:
        """The value *key* holds across *row_indices*.

        Returns ``(value, agreed)``.  When the selected rows disagree, *agreed*
        is False and the caller should show an empty editor so that leaving it
        alone does not flatten the differences.
        """
        values = [self._rows[i].get(key) for i in row_indices]
        if not values:
            return None, False
        first = values[0]
        if all(values_equal(first, v) for v in values[1:]):
            return first, True
        return None, False

    def update_cells(self, row_indices: Iterable[int], key: str, value: Any) -> None:
        """Set *key* to *value* on every row in *row_indices*.

        Coupled keys are written as a set so a row can never carry a codec
        without its matching parameters (see :data:`_COUPLED_KEYS`).  The value
        is stored even when it equals the baseline: rows hold every writable
        parameter and it is :meth:`cell` that decides what a sparse view emits,
        so storing the baseline value here is exactly how an override is undone.
        """
        if key not in _PATH_COLUMNS and uneditable_reason(key) is not None:
            raise KeyError(f"{key!r} cannot be overridden per row")
        if not _overridable(value):
            raise ValueError(f"{value!r} cannot be stored in a CSV cell")
        if key in _PATH_COLUMNS and not str(value).strip():
            raise ValueError(f"{key!r} cannot be blank")

        base = self._effective_kwargs()
        for index in row_indices:
            row = self._rows[index]
            row[key] = value
            if key == "compressor":
                # The previous codec's parameters are meaningless for the new
                # one: blosc's cname/shuffle passed to GZip raise TypeError.
                # Keep them only while the codec is unchanged.
                row["compressor_params"] = (
                    dict(base.get("compressor_params") or {})
                    if values_equal(base.get("compressor"), value)
                    else default_compressor_params(value))

    def reset_cells(self, row_indices: Iterable[int], key: str) -> None:
        """Drop *key*'s override on every row in *row_indices*.

        The row goes back to the baseline value, which is what a blank cell
        means when ``ebridge.to_zarr`` merges the table.
        """
        if key in _PATH_COLUMNS:
            # A path is the row's identity, not an override: there is no
            # baseline to fall back to, and a blank cell would be unrunnable.
            raise KeyError(f"{key!r} has no baseline to fall back to")
        if uneditable_reason(key) is not None:
            raise KeyError(f"{key!r} cannot be overridden per row")
        base = self._effective_kwargs()
        if key not in base:
            for index in row_indices:
                self._rows[index].pop(key, None)
            return
        for index in row_indices:
            self._rows[index][key] = base[key]

    # -- views -----------------------------------------------------------

    def set_compare_config(self, config: dict | None) -> None:
        """Set the global config that highlighting compares rows against.

        Kept separate from the baseline: the baseline is frozen when the batch
        starts and defines run-time inheritance, whereas this tracks whichever
        config file is currently selected, so highlighting stays meaningful if
        the user loads a different one.
        """
        self._compare_config = deepcopy(config) if config else None
        self._compare_kwargs = None          # invalidate cache

    def _comparison_kwargs(self) -> dict[str, Any]:
        """Flat kwargs of the config rows are highlighted against (cached)."""
        if self._compare_kwargs is None:
            source = self._compare_config or self._base_config
            self._compare_kwargs = _build_kwargs(source) if source else {}
        return self._compare_kwargs

    def differs(self, row: dict[str, Any], key: str) -> bool:
        """True when *row* sets *key* to something other than the global config."""
        if key in _PATH_COLUMNS or key not in row:
            return False
        return not values_equal(
            _to_sentinel(key, self._comparison_kwargs().get(key)), row[key])

    def _effective_kwargs(self) -> dict[str, Any]:
        """Flat kwargs a row inherits when it does not state a value itself."""
        if self._base_config is None:
            return {}
        return _build_kwargs(self._base_config)

    def cell(self, row: dict[str, Any], key: str) -> Any:
        """Value to render/write for *key*.

        In sparse mode only deviations are emitted. A blank cell means "inherit
        the baseline", which is exactly how ``ebridge.to_zarr`` reads it.  In full
        mode every row states its effective value, including for keys the row
        never captured: ``_build_kwargs`` emits some conditionally (e.g.
        ``target_chunk_mb`` only when auto-chunking is on), so those are filled
        from the baseline the row would otherwise have inherited.
        """
        if key in _PATH_COLUMNS:
            return row.get(key)
        if self.full:
            return (row[key] if key in row
                    else _to_sentinel(key, self._effective_kwargs().get(key)))
        if key not in row:
            return None
        return row[key] if self.differs(row, key) else None

    def is_inert(self, row: dict[str, Any], key: str,
                 _seen: frozenset[str] = frozenset()) -> bool:
        """True when *key*'s parent switch makes it have no effect for *row*.

        Manual chunk sizes do nothing while ``auto_chunk`` is on, so showing them
        as ordinary values invites the reading that a manual edit will override
        the automatic choice.  The parent's effective value is what the row will
        actually run with: its own override when it has one, else the baseline's.
        """
        spec = _SPEC_BY_KEY.get(key)
        if spec is None or not spec.depends_on or key in _seen:
            return False
        parent = row.get(spec.depends_on)
        if parent is None:
            parent = self._effective_kwargs().get(spec.depends_on)
        if parent is None:
            return False
        # A parent that is itself inert cannot make anything live: with
        # keep_existing_resolutions on, n_layers is ignored, so min_dimension_size
        # (which depends on n_layers) is ignored too.
        if self.is_inert(row, spec.depends_on, _seen | {key}):
            return True
        if isinstance(spec.active_when, bool):
            return bool(parent) != spec.active_when
        return not values_equal(parent, spec.active_when)

    def config_value(self, key: str) -> Any:
        """What *key* falls back to when a row states nothing, or None.

        This is the value a blank cell resolves to at run time, so it is what
        the editor's reset button puts back.  Paths have no such fallback.
        """
        if key in _PATH_COLUMNS:
            return None
        return _to_sentinel(key, self._effective_kwargs().get(key))

    def parent_switch(self, key: str) -> str:
        """The parameter *key* is subordinate to, or '' when it stands alone."""
        spec = _SPEC_BY_KEY.get(key)
        return spec.depends_on if spec else ""

    def inert_reason(self, row: dict[str, Any], key: str) -> str:
        """Why *key* does nothing for *row*, or '' when it is live.

        States the parent's actual disabling value rather than naming the
        parent alone: "apply_smart_downscaling" by itself reads as though
        enabling it is what disables the dependant, which is the opposite of
        what it does.
        """
        if not self.is_inert(row, key):
            return ""
        spec = _SPEC_BY_KEY.get(key)
        if spec is None or not spec.depends_on:
            return ""
        parent = spec.depends_on
        # Report the parent's own condition when the chain is what disables
        # this one, so the message names the setting the user has to change.
        if self.is_inert(row, parent):
            return self.inert_reason(row, parent)
        if isinstance(spec.active_when, bool):
            return f"{parent}={not spec.active_when}"
        return f"{parent}!={spec.active_when}"

    def columns(self, for_csv: bool = False) -> list[str]:
        """Path columns first, then the parameter columns this view needs.

        Ordered the way the conversion form presents its parameters rather than
        alphabetically, so a wide table stays scannable: related settings
        (chunking, downscaling) sit together instead of being scattered by name.

        *for_csv* keeps columns that must reach the file but have no editor of
        their own. ``compressor_params`` rides along with ``compressor`` and
        would break the written batch if it were dropped.
        """
        extra: set[str] = set()
        for row in self._rows:
            for key in row:
                if key in _PATH_COLUMNS:
                    continue
                if not for_csv and key in _COUPLED_RIDERS:
                    continue
                # A column always appears once some row deviates from the
                # config: shown_tabs only *adds* categories, it never hides a
                # deviation the user needs to see.
                if (self.full
                        or self.differs(row, key)
                        or column_header(key)[0] in self.shown_tabs):
                    extra.add(key)
        return [*_PATH_COLUMNS, *sort_keys(extra)]

    # -- validation ------------------------------------------------------

    def validate(self) -> list[str]:
        """Check the batch for problems that would only surface mid-run."""
        problems: list[str] = []

        if not self._rows:
            problems.append("Batch is empty.")
            return problems

        for i, row in enumerate(self._rows, 1):
            src = row.get("input_path", "")
            if not src:
                problems.append(f"Row {i}: no input path.")
            elif not os.path.exists(src):
                problems.append(f"Row {i}: input does not exist: {src}")
            if not row.get("output_path"):
                problems.append(f"Row {i}: no output path.")

        # A duplicated output path means one conversion silently overwrites
        # another; easy to create by adding the same file twice.
        seen: dict[tuple[str, str], int] = {}
        for i, row in enumerate(self._rows, 1):
            key = (str(row.get("output_path", "")),
                   os.path.basename(str(row.get("input_path", ""))))
            if key in seen:
                problems.append(
                    f"Row {i}: same output target as row {seen[key]} "
                    f"({key[1]} -> {key[0]}) : one would overwrite the other."
                )
            else:
                seen[key] = i

        return problems

    # -- persistence -----------------------------------------------------

    def save(self, csv_path: str) -> str:
        """Write ``batch.csv`` plus its config snapshot; returns the CSV path.

        Paths are written relative to the CSV's own directory whenever possible,
        matching how ``take_filepaths`` resolves them, so the folder stays
        portable across machines.
        """
        csv_path = os.path.abspath(csv_path)
        base_dir = os.path.dirname(csv_path)
        os.makedirs(base_dir, exist_ok=True)

        columns = self.columns(for_csv=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(columns)
            for row in self._rows:
                writer.writerow([
                    to_cell(_relativise(row.get(col), base_dir)
                            if col in _PATH_COLUMNS else self.cell(row, col))
                    for col in columns
                ])

        if self._base_config is not None:
            with open(os.path.join(base_dir, CONFIG_SNAPSHOT_NAME),
                      "w", encoding="utf-8") as fh:
                json.dump(self._base_config, fh, indent=2)

        return csv_path

    @classmethod
    def load(cls, csv_path: str) -> "BatchModel":
        """Read a batch back, resolving paths and its config snapshot."""
        csv_path = os.path.abspath(csv_path)
        base_dir = os.path.dirname(csv_path)

        model = cls()
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for raw in csv.DictReader(fh):
                row = {k: v for k, v in raw.items() if k and v not in (None, "")}
                for col in _PATH_COLUMNS:
                    if row.get(col):
                        row[col] = _absolutise(row[col], base_dir)
                model._rows.append(row)

        snapshot = os.path.join(base_dir, CONFIG_SNAPSHOT_NAME)
        if os.path.exists(snapshot):
            with open(snapshot, encoding="utf-8") as fh:
                model._base_config = json.load(fh)

        return model


# ── path helpers ──────────────────────────────────────────────────────────────

def _relativise(path: Any, base_dir: str) -> Any:
    """Make *path* relative to *base_dir*, but only when it sits underneath it.

    A self-contained batch folder (CSV next to its data) stays portable.  Data
    living elsewhere keeps an absolute path rather than growing a ``../../..``
    chain that is both unreadable and fragile if the batch folder moves.
    """
    if not path or not isinstance(path, str):
        return path
    try:
        rel = os.path.relpath(path, base_dir)
    except ValueError:
        # Different drive on Windows, where an absolute path is still valid.
        return path
    if rel.startswith(os.pardir):
        return path
    return rel.replace(os.sep, "/")


def _absolutise(path: str, base_dir: str) -> str:
    return path if os.path.isabs(path) else os.path.normpath(
        os.path.join(base_dir, path))


# ── aggregative guard ─────────────────────────────────────────────────────────

def can_batch(ui_config: dict) -> tuple[bool, str]:
    """Whether the current GUI settings describe a batchable conversion.

    ``take_filepaths`` rejects a table whenever ``concatenation_axes`` is set,
    and ``ebridge.to_zarr`` picks the unary/aggregative branch globally before
    the table is ever read, so an aggregative job cannot be represented as a
    CSV row.  Better to refuse up front than to write a batch that dies on the
    first run.
    """
    axes = (ui_config.get("concatenation", {}) or {}).get("concatenationAxes", "")
    if str(axes).strip():
        return False, (
            "Aggregative conversions cannot be batched.\n\n"
            "Batches are CSV tables, and table input supports one-to-one "
            "conversions only. Clear 'Concat axes' on the Conversion tab to "
            "add this conversion to a batch, or run it directly with Start."
        )
    return True, ""
