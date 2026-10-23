"""Golden outputs of eubi-bridge 0.1.x for the comparison tests (roadmap step 6).

Until 0.2.0 the tests converted every case twice, on eubi-bridge 0.1.x's conversion path
and on the new one, and compared the outputs. The 0.1.x path is retired (``legacy_0_1/``),
so its outputs were recorded once, while it still ran, into
``tests/golden_0_1/<name>.json``: per OME-Zarr store, its group attributes (the eubi-bridge
version stamp left out, temporary paths as ``<tmp>``) and per level the shape, dtype,
chunks and a SHA-256 of the pixels. The tests compare today's outputs with them.

The goldens cannot be recorded again: the recording versions of these tests are in
``legacy_0_1/tests/golden_recording/`` (they ran on eubi-bridge 0.2.0's development tree
at commit 90ab07f, plus the step 6 test changes, with ``EUBI_RECORD_GOLDEN=1``).

Changes made to them since, by hand or by script, never by recording:

- 2026-10-20 (0.3.0): ``series_index`` added beside ``scene_index`` by
  ``tools/add_series_index_to_goldens.py``, no re-recording (the eubi_bridge block holds
  both names from 0.3.0, plan 8.32).
- 2026-10-21 (0.3.1): the multiscale ``type`` added to the 25 goldens whose outputs carry
  it since ozp 0.3.1 (the downscale method, ``"simple"``, or ``"unknown"`` for one level),
  by a script that checked each new file equals the old one plus that key; no
  re-recording.
- 2026-10-22 (spatial plan 9.18, the release after 0.3.1): ``eubi:coordinate_system``
  and ``eubi:coordinate_systems`` added to the 16 placed stores of 7 goldens
  (pyramid_path-ome_tiff-*, -czi_mosaic-*, -czi_views-*) by
  ``tools/add_coordinate_system_to_goldens.py``, which converts each case again and
  checks that every store equals the golden plus exactly these keys, levels unchanged; no
  re-recording. They need the micro-reader that declares coordinate systems.
- 2026-10-22 (0.3.2): omero channel colours, where eubi now colours channels the file gives
  no colour (or one shared colour, as the older readers' white): ``fill_channel_colors``.
  Updated by the tests themselves with ``EUBI_GOLDEN_CHANNEL_COLORS=1`` (``sync_colors``),
  which rewrites a golden only when the new output differs from it in omero channel
  colours alone; no re-recording. metadata_path-auto_windows-v2/-v3 (compared with the
  converted store, not the updated one) had their two white channels set by hand to
  DAPI's and GFP's colours (``wavelength_color(461)`` / ``(509)``).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

GOLDEN = Path(__file__).parent / "golden_0_1"
#: the file name of each golden loaded (by id of the loaded value), for ``sync_colors``
_NAMES: dict = {}


def _strip_version(attrs: dict) -> dict:
    attrs = json.loads(json.dumps(attrs))
    for holder in (attrs, attrs.get("ome") or {}):
        block = holder.get("eubi_bridge")
        if isinstance(block, dict):
            block.pop("version", None)
    return attrs


def _portable(value):
    """`value` with every path under <tmp> written with '/' separators: the golden files
    were recorded on Windows, and the same output names its source file with '/' on
    Linux and macOS."""
    if isinstance(value, str):
        return value.replace("\\", "/") if value.startswith("<tmp>") else value
    if isinstance(value, dict):
        return {k: _portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_portable(v) for v in value]
    return value


def _normalise(value, tmp: Path):
    """`value` (JSON-able) with every spelling of the test's temporary folder as <tmp>."""
    text = json.dumps(value)
    tmp = str(tmp)
    for spelling in (tmp.replace("\\", "\\\\"), tmp.replace("\\", "/"), tmp):
        text = text.replace(spelling, "<tmp>")
    return _portable(json.loads(text))


def describe(root: Path, tmp: Path) -> dict:
    """{store: {"attrs": ..., "levels": [[shape, dtype, chunks, sha256], ...]}}."""
    import zarr
    out = {}
    for store in sorted(Path(root).glob("*.zarr")):
        group = zarr.open_group(str(store), mode="r")
        levels = []
        for _, arr in sorted(group.arrays(), key=lambda kv: int(kv[0])):
            pixels = np.ascontiguousarray(arr[...])
            levels.append([list(arr.shape), str(arr.dtype), list(arr.chunks),
                           hashlib.sha256(pixels.tobytes()).hexdigest()])
        out[store.name] = {"attrs": _normalise(_strip_version(group.attrs.asdict()), tmp),
                           "levels": levels}
    return out


def load(name: str) -> dict:
    path = GOLDEN / f"{name}.json"
    if not path.exists():
        raise AssertionError(f"no golden output {path.name} (see tests/_golden.py)")
    loaded = _portable(json.loads(path.read_text(encoding="utf-8")))
    _NAMES[id(loaded)] = name
    return loaded


class _NotColorsOnly(Exception):
    pass


def _color_diffs(old, new, path=()) -> list:
    """The paths where `old` and `new` differ, each an omero channel's colour
    (``.../omero/channels/<i>/color``); _NotColorsOnly when anything else differs."""
    if isinstance(old, dict) and isinstance(new, dict):
        if sorted(old) != sorted(new):
            raise _NotColorsOnly(path)
        return [d for k in old for d in _color_diffs(old[k], new[k], path + (k,))]
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            raise _NotColorsOnly(path)
        return [d for i, (o, n) in enumerate(zip(old, new))
                for d in _color_diffs(o, n, path + (i,))]
    if old == new:
        return []
    if len(path) >= 4 and path[-4:-2] == ("omero", "channels") and path[-1] == "color":
        return [path]
    raise _NotColorsOnly(path)


def _dump(value, newline_at_end: bool) -> bytes:
    """A golden file's format: sorted keys, indent 1, ASCII escapes, CRLF."""
    text = json.dumps(value, indent=1, sort_keys=True) + ("\n" if newline_at_end else "")
    return text.replace("\n", "\r\n").encode("utf-8")


def sync_colors(expected, actual, key=None) -> None:
    """With EUBI_GOLDEN_CHANNEL_COLORS=1: when `actual` differs from the loaded golden
    `expected` (its part `key`) only in omero channel colours, write the new colours into
    the golden file and into `expected` (module docstring, 0.3.2). Else nothing."""
    if os.environ.get("EUBI_GOLDEN_CHANNEL_COLORS") != "1" or id(expected) not in _NAMES:
        return
    old = expected if key is None else expected[key]
    try:
        diffs = _color_diffs(old, actual)
    except _NotColorsOnly:
        return                                  # the comparison reports it
    if not diffs:
        return
    path = GOLDEN / f"{_NAMES[id(expected)]}.json"
    raw = path.read_bytes()
    stored = json.loads(raw)
    newline_at_end = raw.endswith(b"\n")
    if _dump(stored, newline_at_end) != raw:
        raise AssertionError(f"{path.name}: not in the format sync_colors writes")
    for where in diffs:
        color = actual
        for step in where:
            color = color[step]
        for target in (stored if key is None else stored[key], old):
            for step in where[:-1]:
                target = target[step]
            target[where[-1]] = color
    path.write_bytes(_dump(stored, newline_at_end))


def golden(name: str) -> dict:
    """The golden description `name` (``describe`` of 0.1.x's output)."""
    return load(name)


def value(name: str):
    """The golden value `name` (anything JSON-able, computed on 0.1.x's path)."""
    return load(name)


def comparable(attrs: dict, tmp: Path) -> dict:
    """Group attributes as golden files hold them: no version stamp, paths as <tmp>."""
    return _normalise(_strip_version(attrs), tmp)


def fingerprint(array) -> list:
    """[shape, dtype, sha256] of an array, as golden files hold pixels."""
    array = np.ascontiguousarray(array)
    return [list(array.shape), str(array.dtype), hashlib.sha256(array.tobytes()).hexdigest()]


def assert_same(expected: dict, actual: dict, what: str = "") -> None:
    """Every store, attribute and level of `actual` as in `expected`."""
    assert sorted(actual) == sorted(expected), f"{what}: stores {sorted(actual)} != {sorted(expected)}"
    for store in expected:
        e, a = expected[store], actual[store]
        assert a["attrs"] == e["attrs"], f"{what}: {store} attributes differ"
        assert a["levels"] == e["levels"], f"{what}: {store} levels differ"


def check(name: str, new_root: Path, tmp: Path) -> dict:
    """The output in `new_root` against the golden 0.1.x output `name`."""
    expected = golden(name)
    actual = describe(new_root, tmp)
    sync_colors(expected, actual)
    assert_same(expected, actual, name)
    return actual
