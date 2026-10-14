"""update_pixel_meta / update_channel_meta edit metadata, never pixels.

Both commands collect the whole 'conversion' config section, so conversion
settings reached the update: the default ``squeeze=True`` made every update of
a store with a singleton axis fail ("No zarr group connected"), a saved crop
range made every update fail (``ArrayManager.crop()`` was called with
positional ranges), and a conversion ``dtype`` would set the channel windows
of a store that kept its own dtype.  Separately, the OME-XML companion was
rewritten whenever the store had one, whatever ``save_omexml`` said -- and a
conversion did the same to an OME-Zarr INPUT.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    """An EuBIBridge whose config lives in a temp dir, never the user's own."""
    from eubi_bridge.ebridge import ConfigManager, EuBIBridge
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    monkeypatch.setattr(ConfigManager, "_get_config_dir",
                        lambda self: config_dir)
    monkeypatch.setattr(ConfigManager, "_get_json_path",
                        lambda self: config_dir / ".eubi_config.json")
    return EuBIBridge()


def _convert(bridge, tmp_path, name, **kwargs) -> Path:
    """A uint16 z-stack converted to OME-Zarr (zarr v2, with OME-XML)."""
    source = tmp_path / f"{name}.tif"
    tifffile.imwrite(
        source, np.random.randint(10, 50, (4, 16, 16)).astype(np.uint16),
        imagej=True, metadata={"axes": "ZYX"})
    out = tmp_path / f"out_{name}"
    bridge.to_zarr(str(source), str(out), zarr_format=2, verbose=False,
                   **kwargs)
    return next(out.glob("*.zarr"))


def _attrs(store: Path) -> dict:
    return json.loads((store / ".zattrs").read_text())


def _axes(store: Path) -> str:
    return "".join(a["name"] for a in _attrs(store)["multiscales"][0]["axes"])


def _shape(store: Path) -> list:
    return json.loads((store / "0" / ".zarray").read_text())["shape"]


def _scale(store: Path) -> list:
    datasets = _attrs(store)["multiscales"][0]["datasets"]
    return datasets[0]["coordinateTransformations"][0]["scale"]


def _xml(store: Path) -> Path:
    return store / "OME" / "METADATA.ome.xml"


@pytest.fixture
def unsqueezed(bridge, tmp_path):
    """A store that keeps its singleton t and c axes, as many tools write."""
    store = _convert(bridge, tmp_path, "unsqueezed", squeeze=False)
    assert 1 in _shape(store), "the fixture needs a singleton axis"
    return store


class TestSingletonAxes:
    """The default squeeze=True of conversions must not reach an update."""

    def test_pixel_meta_on_an_unsqueezed_store(self, bridge, unsqueezed):
        axes, shape = _axes(unsqueezed), _shape(unsqueezed)
        bridge.update_pixel_meta(str(unsqueezed), z_scale=0.75)
        assert _scale(unsqueezed)[axes.index("z")] == 0.75
        assert _axes(unsqueezed) == axes
        assert _shape(unsqueezed) == shape

    def test_channel_meta_on_an_unsqueezed_store(self, bridge, unsqueezed):
        axes = _axes(unsqueezed)
        bridge.update_channel_meta(str(unsqueezed), channel_labels="0,DAPI")
        assert _attrs(unsqueezed)["omero"]["channels"][0]["label"] == "DAPI"
        assert _axes(unsqueezed) == axes

    def test_explicit_squeeze_in_the_config(self, bridge, unsqueezed):
        bridge.configure.conversion(squeeze=True)
        axes = _axes(unsqueezed)
        bridge.update_pixel_meta(str(unsqueezed), x_scale=0.25)
        assert _scale(unsqueezed)[-1] == 0.25
        assert _axes(unsqueezed) == axes


class TestConversionSettingsIgnored:
    """Crop ranges and dtype are conversion settings; an update ignores them."""

    def test_a_saved_crop_range(self, bridge, tmp_path):
        store = _convert(bridge, tmp_path, "crop")
        shape = _shape(store)
        bridge.configure.conversion(z_range="0,2", y_range="0,8")
        bridge.update_pixel_meta(str(store), x_scale=0.25)
        assert _scale(store)[-1] == 0.25
        assert _shape(store) == shape

    def test_the_worker_ignores_squeeze_and_crop(self, unsqueezed):
        """Every caller goes through update_worker, so the rule lives there."""
        from eubi_bridge.conversion.metadata_update_worker import update_worker
        axes, shape = _axes(unsqueezed), _shape(unsqueezed)
        asyncio.run(update_worker(
            str(unsqueezed), x_scale=0.25, squeeze=True,
            z_range=(0, 2), channel_range=(0, 1), save_omexml=False))
        assert _scale(unsqueezed)[-1] == 0.25
        assert (_axes(unsqueezed), _shape(unsqueezed)) == (axes, shape)

    def test_a_conversion_dtype_does_not_set_the_windows(self, bridge, tmp_path):
        """The store is uint16; a uint8 conversion setting must not make its
        windows describe uint8 data."""
        store = _convert(bridge, tmp_path, "dtype")
        bridge.configure.conversion(dtype="uint8")
        bridge.update_channel_meta(str(store), channel_labels="0,GFP")
        window = _attrs(store)["omero"]["channels"][0]["window"]
        assert window["end"] == np.iinfo(np.uint16).max


class TestSaveOmeXml:
    def test_false_leaves_the_xml_alone(self, bridge, tmp_path):
        store = _convert(bridge, tmp_path, "keep")
        before = _xml(store).read_bytes()
        bridge.configure.conversion(save_omexml=False)
        bridge.update_pixel_meta(str(store), x_scale=0.25)
        assert _scale(store)[-1] == 0.25
        assert _xml(store).read_bytes() == before

    def test_true_rewrites_it(self, bridge, tmp_path):
        store = _convert(bridge, tmp_path, "rewrite")
        bridge.update_pixel_meta(str(store), x_scale=0.25)
        assert 'PhysicalSizeX="0.25"' in _xml(store).read_text(encoding="utf-8")


class TestConversionInput:
    def test_converting_an_ome_zarr_does_not_write_into_it(self, bridge, tmp_path):
        source = _convert(bridge, tmp_path, "input")
        old = 1_000_000_000   # 2001: anything the conversion writes is newer
        os.utime(_xml(source), (old, old))
        bridge.to_zarr(str(source), str(tmp_path / "second"), verbose=False)
        assert os.stat(_xml(source)).st_mtime == old
