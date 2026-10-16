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
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

GOLDEN = Path(__file__).parent / "golden_0_1"


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
    return _portable(json.loads(path.read_text(encoding="utf-8")))


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
    assert_same(expected, actual, name)
    return actual
