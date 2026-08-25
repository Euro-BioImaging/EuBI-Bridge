"""Recursive browsing must treat an OME-Zarr store as one image.

Walking into a ``.zarr`` directory offers every chunk file ('0/0/0/4') as a
separate input; conversion then fails with UnknownFormatException from
Bio-Formats.  These tests pin the store-level behaviour.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eubi_bridge.qt_gui.core.file_service import list_local_recursive


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A folder holding a plain TIFF plus a zarr v2 store with real chunks."""
    (tmp_path / "plain").mkdir()
    (tmp_path / "plain" / "image.tif").write_bytes(b"II*\0")

    store = tmp_path / "out" / "image.ome.zarr"
    (store / "0").mkdir(parents=True)
    (store / ".zattrs").write_text(json.dumps({"multiscales": [{}]}))
    (store / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (store / "0" / ".zarray").write_text(json.dumps({"zarr_format": 2}))
    for i in range(5):
        (store / "0" / f"0.0.0.{i}").write_bytes(b"\0" * 8)
    return tmp_path


def _paths(entries):
    return [e["path"] for e in entries]


def test_chunks_are_not_offered_as_inputs(tree):
    entries = list_local_recursive(str(tree), include_patterns=["*"])
    leaked = [p for p in _paths(entries)
              if ".ome.zarr" in p and not p.endswith(".ome.zarr")]
    assert leaked == [], f"chunk files offered as inputs: {leaked[:3]}"


def test_store_is_offered_once(tree):
    entries = list_local_recursive(str(tree), include_patterns=["*"])
    stores = [e for e in entries if e["isOmeZarr"]]
    assert len(stores) == 1
    assert stores[0]["path"].endswith("image.ome.zarr")
    assert stores[0]["isDirectory"] is True


def test_sibling_files_still_found(tree):
    """Pruning the store must not hide anything outside it."""
    entries = list_local_recursive(str(tree), include_patterns=["*.tif"])
    assert [Path(p).name for p in _paths(entries)] == ["image.tif"]


def test_zarr_pattern_matches_the_store(tree):
    """`*.zarr` should select the store — impossible while it was descended."""
    entries = list_local_recursive(str(tree), include_patterns=["*.zarr"])
    assert len(entries) == 1
    assert entries[0]["path"].endswith("image.ome.zarr")


def test_excludes_apply_to_the_store(tree):
    entries = list_local_recursive(
        str(tree), include_patterns=["*"], exclude_patterns=["*.zarr"])
    assert not [e for e in entries if e["isOmeZarr"]]
