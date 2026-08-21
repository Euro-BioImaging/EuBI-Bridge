"""
Batch model — a queue of one-to-one conversions persisted as a CSV table.

A batch is a folder holding two files:

    batch.csv           one row per input file; sparse per-row overrides
    batch_config.json   snapshot of the global GUI config the rows diff against

``eubi to_zarr <batch.csv>`` already understands this format: ``take_filepaths``
reads the table, resolves ``input_path`` / ``output_path`` relative to the CSV's
own directory, and ``ebridge.to_zarr`` merges each row over the global config
(blank cells fall through to the global value).  This module only has to produce
a table that obeys those rules — it adds no new format.

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
from typing import Any, Iterable

from eubi_bridge.qt_gui.workers.conversion_worker import _build_kwargs

# Columns that are addressed positionally rather than as parameter overrides.
_PATH_COLUMNS = ("input_path", "output_path")

# Parameters that cannot be expressed in a single CSV cell.  They are carried by
# the config snapshot instead; a row is never allowed to override them.
#
# ``compressor`` / ``compressor_params`` are deliberately absent: the params dict
# is written as JSON and parsed back by ConversionConfig, so the codec and its
# parameters can vary per row.  They must always be emitted *together* — see
# _COUPLED_KEYS — because e.g. ``GZip(cname='lz4')`` raises TypeError.
_NON_ROW_OVERRIDABLE = frozenset({
    "time_range", "channel_range", "z_range", "y_range", "x_range",
})

# Groups of keys that are only meaningful as a set.  If any member differs from
# the baseline, every member is written to the row so they can never desynchronise.
_COUPLED_KEYS: tuple[frozenset[str], ...] = (
    frozenset({"compressor", "compressor_params"}),
)

# Keys that _build_kwargs() emits only conditionally but whose absence is
# harmless, because another key the row *can* override already governs them.
# ``target_chunk_mb`` is only read when ``auto_chunk`` is true, and ``auto_chunk``
# is itself a per-row override — so inheriting a stale value changes nothing.
_HARMLESS_WHEN_ABSENT = frozenset({"target_chunk_mb"})

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
    that cannot vary per row — compression above all — actually applied.
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
    or as text/float parsed back out of a CSV — ``128`` vs ``128.0`` vs
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

        Returns a list of human-readable warnings — currently the parameters
        that differ from the baseline but cannot be stored per row.  Those keep
        the baseline value, so the caller should surface the warning rather than
        silently produce a batch that behaves unexpectedly.

        If no baseline was set, *ui_config* becomes it — which means this first
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
        # the baseline has and this row drops is a real difference too — and one
        # a CSV cannot express, since a blank cell means "inherit", not "unset".
        for key in set(row_kwargs) | set(base_kwargs):
            if key not in row_kwargs:
                if key not in _HARMLESS_WHEN_ABSENT:
                    blocked.append(key)
                continue
            value = row_kwargs[key]
            if base_kwargs.get(key) == value:
                continue
            if key in _NON_ROW_OVERRIDABLE or not _overridable(value):
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
        storable = {k: v for k, v in row_kwargs.items()
                    if k not in _NON_ROW_OVERRIDABLE and _overridable(v)}

        for path in input_paths:
            self._rows.append({
                "input_path": path,
                "output_path": output_path,
                **storable,
            })

        return blocked

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
        return not values_equal(self._comparison_kwargs().get(key), row[key])

    def _effective_kwargs(self) -> dict[str, Any]:
        """Flat kwargs a row inherits when it does not state a value itself."""
        if self._base_config is None:
            return {}
        return _build_kwargs(self._base_config)

    def cell(self, row: dict[str, Any], key: str) -> Any:
        """Value to render/write for *key*.

        In sparse mode only deviations are emitted — a blank cell means "inherit
        the baseline", which is exactly how ``ebridge.to_zarr`` reads it.  In full
        mode every row states its effective value, including for keys the row
        never captured: ``_build_kwargs`` emits some conditionally (e.g.
        ``target_chunk_mb`` only when auto-chunking is on), so those are filled
        from the baseline the row would otherwise have inherited.
        """
        if key in _PATH_COLUMNS:
            return row.get(key)
        if self.full:
            return row[key] if key in row else self._effective_kwargs().get(key)
        if key not in row:
            return None
        return row[key] if self.differs(row, key) else None

    def columns(self) -> list[str]:
        """Path columns first, then the parameter columns this view needs."""
        extra: set[str] = set()
        for row in self._rows:
            for key in row:
                if key in _PATH_COLUMNS:
                    continue
                if self.full or self.differs(row, key):
                    extra.add(key)
        return [*_PATH_COLUMNS, *sorted(extra)]

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
                problems.append(f"Row {i}: input does not exist — {src}")
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
                    f"({key[1]} -> {key[0]}) — one would overwrite the other."
                )
            else:
                seen[key] = i

        return problems

    # -- persistence -----------------------------------------------------

    def save(self, csv_path: str) -> str:
        """Write ``batch.csv`` plus its config snapshot; returns the CSV path.

        Paths are written relative to the CSV's own directory whenever possible,
        matching how ``take_filepaths`` resolves them — so the folder stays
        portable across machines.
        """
        csv_path = os.path.abspath(csv_path)
        base_dir = os.path.dirname(csv_path)
        os.makedirs(base_dir, exist_ok=True)

        columns = self.columns()
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
        # Different drive on Windows — an absolute path is still valid.
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
    the table is ever read — so an aggregative job cannot be represented as a
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
