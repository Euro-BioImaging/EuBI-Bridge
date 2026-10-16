"""The metadata commands on the 0.2.0 path (eubi_bridge/conversion/metadata_path.py,
roadmap step 4).

- update_pixel_meta / update_channel_meta: one ozp ``IO().update_metadata`` call, on
  threads. The same store is updated on both paths; the results must be the same, except
  the decided differences: ``auto`` windows are the coarsest level's percentiles (as for
  conversions; the old path left the windows), and the OME-XML carries the new channel
  names (the old path rewrote it with the old ones).
- show_pixel_meta: a file's metadata as a conversion writes it (same reader, same
  scales, labels and colours).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("micro_reader")
pytest.importorskip("ome_zarr_pyramid")
tifffile = pytest.importorskip("tifffile")
zarr = pytest.importorskip("zarr")

RNG = np.random.default_rng(5)


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    """An EuBIBridge whose config lives in a temp dir, never the user's own."""
    from eubi_bridge.ebridge import ConfigManager, EuBIBridge
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    monkeypatch.setattr(ConfigManager, "_get_config_dir", lambda self: config_dir)
    monkeypatch.setattr(ConfigManager, "_get_json_path",
                        lambda self: config_dir / ".eubi_config.json")
    return EuBIBridge()


def _source(tmp_path, names=("DAPI", "GFP")) -> Path:
    src = tmp_path / "two_channels.ome.tif"
    data = RNG.integers(100, 3000, (2, 8, 300, 320)).astype("u2")
    meta = {"axes": "CZYX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5, "PhysicalSizeZ": 2.0}
    if names:
        meta["Channel"] = {"Name": list(names)}
    tifffile.imwrite(str(src), data, ome=True, photometric="minisblack", metadata=meta)
    return src


@pytest.fixture(params=[2, 3], ids=["v2", "v3"])
def store(request, bridge, tmp_path, monkeypatch):
    monkeypatch.setenv("EUBI_NEW_PATH", "1")
    out = tmp_path / "converted"
    bridge.to_zarr(str(_source(tmp_path)), str(out), squeeze=False, verbose=False,
                   zarr_format=request.param)
    (found,) = out.glob("*.zarr")
    return found


def _attrs(store: Path) -> dict:
    if (store / ".zattrs").exists():
        return json.loads((store / ".zattrs").read_text())
    return json.loads((store / "zarr.json").read_text())["attributes"]


def _ngff(store: Path) -> dict:
    attrs = _attrs(store)
    return attrs.get("ome", attrs)


def _both(bridge, store, tmp_path, monkeypatch, command, **kwargs):
    """`command` on a copy of `store` on each path: (old copy, new copy)."""
    copies = []
    for flag in ("0", "1"):
        copy = tmp_path / f"{command}_{flag}_{store.name}"
        shutil.copytree(store, copy)
        monkeypatch.setenv("EUBI_NEW_PATH", flag)
        getattr(bridge, command)(str(copy), **kwargs)
        copies.append(copy)
    return copies


def _xml(store: Path) -> str:
    return (store / "OME" / "METADATA.ome.xml").read_text(encoding="utf-8")


def test_pixel_meta_on_both_paths(bridge, store, tmp_path, monkeypatch):
    old, new = _both(bridge, store, tmp_path, monkeypatch, "update_pixel_meta",
                     x_scale=0.3, z_scale=1.5, z_unit="nanometer", time_unit="millisecond")
    assert _attrs(old) == _attrs(new)
    datasets = _ngff(new)["multiscales"][0]["datasets"]
    assert len(datasets) > 1
    assert datasets[1]["coordinateTransformations"][0]["scale"][-1] == pytest.approx(0.6)
    for xml in (_xml(old), _xml(new)):
        assert 'PhysicalSizeX="0.3"' in xml and 'PhysicalSizeZ="1.5"' in xml


def test_channel_meta_on_both_paths(bridge, store, tmp_path, monkeypatch):
    old, new = _both(bridge, store, tmp_path, monkeypatch, "update_channel_meta",
                     channel_labels="0,A;1,B", channel_colors="1,magenta")
    assert _attrs(old) == _attrs(new)
    assert [(c["label"], c["color"]) for c in _ngff(new)["omero"]["channels"]] == \
        [("A", "FFFFFF"), ("B", "FF00FF")]
    assert 'Name="A"' in _xml(new) and 'Name="B"' in _xml(new)     # the new names
    assert 'Name="DAPI"' in _xml(old)                              # 0.1.x kept the old ones


def test_from_array_windows_on_both_paths(bridge, store, tmp_path, monkeypatch):
    old, new = _both(bridge, store, tmp_path, monkeypatch, "update_channel_meta",
                     channel_intensity_limits="from_array")
    assert _attrs(old) == _attrs(new)
    window = _ngff(new)["omero"]["channels"][0]["window"]
    assert (window["min"], window["max"]) == (0, 65535) and 100 <= window["start"] < 200


def test_auto_windows_are_the_coarsest_levels_percentiles(bridge, store, tmp_path,
                                                          monkeypatch):
    old, new = _both(bridge, store, tmp_path, monkeypatch, "update_channel_meta",
                     channel_intensity_limits="auto")
    assert _ngff(old)["omero"] == _ngff(store)["omero"]            # 0.1.x left them
    datasets = _ngff(new)["multiscales"][0]["datasets"]
    coarsest = np.asarray(zarr.open_group(str(new), mode="r")[datasets[-1]["path"]])
    for k, ch in enumerate(_ngff(new)["omero"]["channels"]):
        lo, hi = np.percentile(coarsest[:, k], (0.1, 99.9))
        assert (ch["window"]["start"], ch["window"]["end"]) == (int(np.floor(lo)),
                                                                int(np.ceil(hi)))


def test_pixel_meta_keeps_the_windows(bridge, store, monkeypatch):
    from ome_zarr_pyramid import IO
    monkeypatch.setenv("EUBI_NEW_PATH", "1")
    IO().update_metadata(str(store), channels={0: {"window": {"start": 7, "end": 9}}})
    bridge.update_pixel_meta(str(store), x_scale=0.25)
    assert _ngff(store)["omero"]["channels"][0]["window"]["start"] == 7


def test_updates_run_on_threads(bridge, store, monkeypatch):
    """No Java VM and no ArrayManager: a process pool is not needed (nor started)."""
    from eubi_bridge.conversion import updater

    def refuse(*args, **kwargs):
        raise AssertionError("a process pool was started")

    monkeypatch.setenv("EUBI_NEW_PATH", "1")
    monkeypatch.setattr(updater, "ProcessPoolExecutor", refuse)
    bridge.update_pixel_meta(str(store), y_scale=0.4)
    assert _ngff(store)["multiscales"][0]["datasets"][0]["coordinateTransformations"][0][
        "scale"][-2] == 0.4


def test_show_pixel_meta_shows_what_a_conversion_writes(bridge, tmp_path, monkeypatch):
    from eubi_bridge.conversion.metadata_path import read_metadata
    monkeypatch.setenv("EUBI_NEW_PATH", "1")
    src = _source(tmp_path, names=None)                    # unnamed channels
    bridge.to_zarr(str(src), str(tmp_path / "out"), squeeze=False, verbose=False)
    (written,) = (tmp_path / "out").glob("*.zarr")
    shown = read_metadata(str(src), {"series": 0})
    ngff = _ngff(written)
    axes = "".join(a["name"] for a in ngff["multiscales"][0]["axes"])
    assert shown["axes"] == axes
    scale = ngff["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
    assert [shown["scale"][ax] for ax in axes] == scale
    assert shown["channels"] == [{"label": c["label"], "color": c["color"]}
                                 for c in ngff["omero"]["channels"]]
    assert [c["label"] for c in shown["channels"]] == ["Channel 0", "Channel 1"]
    assert read_metadata(str(written), {}) | {"input_path": None, "series": None} == \
        shown | {"input_path": None, "series": None}       # the OME-Zarr reads the same


def test_show_pixel_meta_lists_every_scene_asked_for(tmp_path, monkeypatch):
    from eubi_bridge.conversion.metadata_path import read_metadata
    monkeypatch.setenv("EUBI_NEW_PATH", "1")
    src = tmp_path / "two_scenes.ome.tif"
    with tifffile.TiffWriter(str(src), ome=True) as tw:
        tw.write(np.zeros((2, 16, 20), "u2"), photometric="minisblack",
                 metadata={"axes": "CYX"})
        tw.write(np.zeros((3, 24, 20), "u1"), photometric="minisblack",
                 metadata={"axes": "ZYX"})
    shown = read_metadata(str(src), {"series": "all"})
    assert [s["dtype"] for s in shown] == ["uint16", "uint8"]
    assert [s["input_path"].endswith(f"[scene {k}]") for k, s in enumerate(shown)] == [True] * 2
    assert read_metadata(str(src), {"series": 1})["shape"]["z"] == 3


def test_renaming_a_channel_keeps_its_window(bridge, store, tmp_path, monkeypatch):
    """update_channel_meta changes the windows only when channel_intensity_limits is
    given (Bugra, 2026-10-16; 0.1.x reset them to the dtype's range on every update)."""
    from ome_zarr_pyramid import IO
    IO().update_metadata(str(store), channels={0: {"window": {"start": 7, "end": 9}}})
    old, new = _both(bridge, store, tmp_path, monkeypatch, "update_channel_meta",
                     channel_labels="0,Hoechst")
    for copy in (old, new):
        channel = _ngff(copy)["omero"]["channels"][0]
        assert channel["label"] == "Hoechst"
        assert (channel["window"]["start"], channel["window"]["end"]) == (7, 9)
    monkeypatch.setenv("EUBI_NEW_PATH", "1")
    bridge.update_channel_meta(str(new), channel_intensity_limits="from_dtype")
    assert _ngff(new)["omero"]["channels"][0]["window"]["end"] == 65535   # when asked
