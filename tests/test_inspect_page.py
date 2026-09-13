"""The Inspect page: reading an OME-Zarr's metadata, and editing it in place.

No suite touched this module before 0.1.3, yet it is half the GUI -- and the
half that *writes back* to a store the user already converted, so a fault here
damages finished work rather than merely failing a job.

The viewer and render paths are left out: they need a display and real pixel
throughput, which is a different kind of test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.conftest import qt_available


@pytest.fixture
def app():
    if not qt_available():
        pytest.skip("PyQt6 unavailable or no usable Qt platform plugin")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """A real converted OME-Zarr, since these helpers parse actual metadata."""
    if not qt_available():
        pytest.skip("PyQt6 unavailable")
    tmp = tmp_path_factory.mktemp("inspect")
    source = tmp / "img.tif"
    tifffile.imwrite(
        source, np.random.randint(0, 255, (4, 2, 32, 32)).astype(np.uint8),
        imagej=True, metadata={"axes": "ZCYX"})
    from eubi_bridge.ebridge import EuBIBridge
    out = tmp / "out"
    EuBIBridge().to_zarr(str(source), str(out), verbose=False)
    stores = list(out.glob("*.zarr"))
    assert stores, "conversion produced no store"
    return stores[0]


class TestZattrsHelpers:
    """Read/modify/write of .zattrs, which every edit on this page goes through."""

    def test_it_reads_a_real_store(self, store):
        from eubi_bridge.qt_gui.pages.inspect_page import _load_zattrs
        data, _ = _load_zattrs(str(store))
        assert "multiscales" in data

    def test_a_write_round_trips(self, tmp_path):
        from eubi_bridge.qt_gui.pages.inspect_page import _load_zattrs, _save_zattrs
        target = tmp_path / ".zattrs"
        _save_zattrs(str(target), {"multiscales": [{"axes": []}], "marker": 1})
        assert json.loads(target.read_text())["marker"] == 1

    def test_saving_leaves_valid_json(self, tmp_path):
        """A truncated or invalid write would make the store unreadable."""
        from eubi_bridge.qt_gui.pages.inspect_page import _save_zattrs
        target = tmp_path / ".zattrs"
        _save_zattrs(str(target), {"a": [1, 2, 3], "b": {"c": "d"}})
        assert json.loads(target.read_text()) == {"a": [1, 2, 3], "b": {"c": "d"}}


class TestPixelSizeEditing:
    """Pixel sizes are editable here and written back into the store."""

    def test_they_are_read_from_the_store(self, app, store):
        """Each entry is {axis, value, unit} -- the shape the editor rows and
        _update_scales_on_disk both expect."""
        from eubi_bridge.qt_gui.pages.inspect_page import InspectPage
        page = InspectPage()
        sizes = page._read_pixel_sizes(str(store))
        assert sizes, "no pixel sizes read from a real store"
        assert all({"axis", "value", "unit"} <= set(entry) for entry in sizes)
        assert {e["axis"] for e in sizes} >= {"z", "y", "x"}
        page.deleteLater()

    def test_writing_changes_the_stored_scale(self, store):
        from eubi_bridge.qt_gui.pages.inspect_page import (
            _load_zattrs, _update_scales_on_disk)
        before, _ = _load_zattrs(str(store))
        original = before["multiscales"][0]["datasets"][0][
            "coordinateTransformations"][0]["scale"]

        axes = [a["name"] for a in before["multiscales"][0]["axes"]]
        scales = [{"axis": name, "value": 9.0 if name == "z" else 1.0,
                   "unit": "micrometer"} for name in axes]
        _update_scales_on_disk(str(store), scales)

        after, _ = _load_zattrs(str(store))
        updated = after["multiscales"][0]["datasets"][0][
            "coordinateTransformations"][0]["scale"]
        assert updated != original
        assert 9.0 in updated

        # Put it back: the fixture is module-scoped and shared.
        restore = [{"axis": n, "value": v, "unit": "micrometer"}
                   for n, v in zip(axes, original)]
        _update_scales_on_disk(str(store), restore)


class TestChannelEditing:
    def test_channel_metadata_is_written_back(self, store):
        from eubi_bridge.qt_gui.pages.inspect_page import (
            _load_zattrs, _update_channels_on_disk)
        before, _ = _load_zattrs(str(store))
        channels = before.get("omero", {}).get("channels")
        if not channels:
            pytest.skip("store has no omero channel metadata")

        edited = [dict(c) for c in channels]
        edited[0]["label"] = "RENAMED"
        _update_channels_on_disk(str(store), edited)

        after, _ = _load_zattrs(str(store))
        assert after["omero"]["channels"][0]["label"] == "RENAMED"

        _update_channels_on_disk(str(store), channels)   # restore


class TestCompressorFormatting:
    """The metadata tree shows the codec per pyramid level.

    The input is the page's own ``{"name": ..., "params": {...}}`` summary, not
    a raw zarr codec dict.
    """

    def test_a_blosc_codec_names_its_inner_codec_and_level(self):
        from eubi_bridge.qt_gui.pages.inspect_page import _fmt_compressor
        text = _fmt_compressor(
            {"name": "blosc", "params": {"inner_codec": "lz4", "level": 5}})
        assert "blosc/lz4" in text
        assert "L5" in text

    def test_no_compression_is_stated_not_blank(self):
        """An empty cell would read as "unknown" rather than "uncompressed"."""
        from eubi_bridge.qt_gui.pages.inspect_page import _fmt_compressor
        assert _fmt_compressor({"name": "none"}) == "none"
        assert _fmt_compressor({}).strip() == "none"

    def test_a_codec_without_params_is_still_named(self):
        from eubi_bridge.qt_gui.pages.inspect_page import _fmt_compressor
        assert _fmt_compressor({"name": "zstd"}) == "zstd"

    def test_noshuffle_is_not_advertised(self):
        """Only a shuffle that is actually on is worth the space."""
        from eubi_bridge.qt_gui.pages.inspect_page import _fmt_compressor
        text = _fmt_compressor(
            {"name": "blosc", "params": {"shuffle": "noshuffle"}})
        assert "noshuffle" not in text

    def test_an_active_shuffle_is_shown(self):
        from eubi_bridge.qt_gui.pages.inspect_page import _fmt_compressor
        text = _fmt_compressor(
            {"name": "blosc", "params": {"shuffle": "bitshuffle"}})
        assert "bitshuffle" in text


class TestInspectPageBuilds:
    def test_it_constructs(self, app):
        from eubi_bridge.qt_gui.pages.inspect_page import InspectPage
        page = InspectPage()
        assert page is not None
        page.deleteLater()

    def test_loading_a_store_does_not_raise(self, app, store):
        """The page is handed a path by the sidebar; a bad load must not crash."""
        from eubi_bridge.qt_gui.pages.inspect_page import InspectPage
        page = InspectPage()
        page._load_zarr(str(store))
        app.processEvents()
        page.deleteLater()

    def test_a_missing_path_does_not_raise(self, app, tmp_path):
        from eubi_bridge.qt_gui.pages.inspect_page import InspectPage
        page = InspectPage()
        page._load_zarr(str(tmp_path / "nope.zarr"))
        app.processEvents()
        page.deleteLater()
