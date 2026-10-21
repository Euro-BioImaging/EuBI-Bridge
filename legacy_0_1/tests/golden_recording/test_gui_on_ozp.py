"""eubi-bridge's GUI on ome_zarr_pyramid's Pyramid (roadmap step 5).

The Inspect tab's plane server reads stores with ozp's Pyramid, and its metadata edits
(pixel sizes, channels) are ozp ``IO().update_metadata`` calls. Every server answer must
be what it was on eubi-bridge 0.1.x's own Pyramid (golden file, tests/_golden.py, recorded
before step 6 retired it).
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ome_zarr_pyramid")
tifffile = pytest.importorskip("tifffile")
zarr = pytest.importorskip("zarr")
Image = pytest.importorskip("PIL.Image")

from tests import _golden  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "eubi_bridge" / "qt_gui" / "server" / "zarr_plane_server.py"


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """Two channels, z, three levels, OME-Zarr 0.5."""
    from ome_zarr_pyramid import IO, Pyramid
    data = np.random.default_rng(3).integers(100, 3000, (1, 2, 6, 300, 320)).astype("u2")
    pyr = Pyramid().from_arrays([data], axis_order="tczyx",
                                scales=[[1.0, 1.0, 2.0, 0.5, 0.5]], version="0.5")
    pyr = pyr.set_channels({0: {"label": "DAPI", "color": "0000FF"},
                            1: {"label": "GFP", "color": "00FF00"}})
    path = tmp_path_factory.mktemp("gui") / "img.zarr"
    IO().write_pyramid(pyr.downscale(n_layers=3, defer=True), str(path), windows="dtype")
    return path


def _start(flag: str, port: int) -> subprocess.Popen:
    env = dict(os.environ, EUBI_NEW_PATH=flag)
    proc = subprocess.Popen([sys.executable, str(SERVER), str(port)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    for line in proc.stdout:
        if '"ready"' in line:
            return proc
    proc.kill()
    pytest.skip("the plane server did not start")


def _get(port, endpoint, **params):
    url = f"http://127.0.0.1:{port}{endpoint}?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=120) as r:
        body, kind = r.read(), r.headers.get("Content-Type", "")
    if kind.startswith("image"):
        return np.asarray(Image.open(io.BytesIO(body)))
    return json.loads(body)


def _post(port, endpoint, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{endpoint}",
                                 data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def _answers(port, path) -> dict:
    channels = json.dumps([{"index": i, "color": c, "visible": True, "intensityMin": 0,
                            "intensityMax": 2000} for i, c in enumerate(("#FF0000", "#00FF00"))])
    out = {"info": _get(port, "/info", path=path), "metadata": _get(port, "/metadata", path=path)}
    for level in range(out["metadata"]["resolutionLevels"]):
        grid = _get(port, "/tile_grid", path=path, level=level, orientation="XY")
        out[f"grid {level}"] = grid
        tile = grid["tiles"][-1]
        out[f"tile {level}"] = _get(port, "/tile", path=path, level=level, orientation="XY",
                                    t=0, z=0, rowStart=tile["dataRowStart"],
                                    rowEnd=tile["dataRowEnd"], colStart=tile["dataColStart"],
                                    colEnd=tile["dataColEnd"], channels=channels)
        out[f"plane {level}"] = _get(port, "/plane", path=path, level=level,
                                     orientation="XY", t=0, z=0, fovSize=256, channels=channels)
    out["plane XZ"] = _get(port, "/plane", path=path, level=0, orientation="XZ", t=0,
                           sliceIdx=10, fovSize=256, channels=channels)
    out["minmax"] = [_get(port, "/channel_minmax", path=path, channel=c) for c in range(2)]
    return out


SCALES = [{"axis": "x", "value": 0.33, "unit": "micrometer"},
          {"axis": "z", "value": 1.7, "unit": "nanometer"}]


def _comparable(answers: dict, store: Path, tmp_path: Path) -> dict:
    """Answers as golden files hold them: images as fingerprints, paths as <tmp>."""
    out = {k: _golden.fingerprint(v) if isinstance(v, np.ndarray) else v
           for k, v in answers.items()}
    return _golden._normalise(_golden._normalise(out, store.parent), tmp_path)


def _answers_and_update(flag: str, port: int, store: Path, tmp_path: Path) -> dict:
    server = _start(flag, port)
    try:
        out = _answers(port, str(store))
        copy = tmp_path / str(port) / "copy.zarr"
        shutil.copytree(store, copy)
        assert _post(port, "/update_scales", {"path": str(copy), "scales": SCALES}) == {"ok": True}
        out["metadata after update"] = _get(port, "/metadata", path=str(copy))
        return _comparable(out, store, tmp_path / str(port))
    finally:
        server.terminate()


def test_the_plane_server_answers_as_on_0_1(store, tmp_path):
    old = _golden.value("gui-plane_server",
                        lambda: _answers_and_update("0", 5791, store, tmp_path))
    new = _answers_and_update("1", 5792, store, tmp_path)
    assert new.keys() == old.keys()
    assert new["metadata"]["resolutionLevels"] == 3
    for key in old:
        assert new[key] == old[key], key
    x = next(s for s in new["metadata after update"]["scales"] if s.get("axis") == "x")
    assert x["value"] == pytest.approx(0.33)


def _ngff(path: Path) -> dict:
    attrs = json.loads((path / "zarr.json").read_text())["attributes"]
    return attrs.get("ome", attrs)


def test_the_inspect_pages_edits_are_ozp_updates(store, tmp_path, monkeypatch):
    from eubi_bridge.qt_gui.pages import inspect_page as page
    copy = tmp_path / "edited.zarr"
    shutil.copytree(store, copy)
    chunks = sorted(p.stat().st_mtime_ns for p in copy.rglob("c/*"))
    monkeypatch.setenv("EUBI_NEW_PATH", "1")
    page._update_scales_on_disk(str(copy), [{"axis": "x", "value": 0.25, "unit": "micrometer"}])
    page._update_channels_on_disk(str(copy), [
        {"label": "Hoechst", "color": "#0000FF", "visible": True, "intensityMin": 10,
         "intensityMax": 900},
        {"label": "GFP", "color": "#00FF00", "visible": False, "intensityMin": 5,
         "intensityMax": 1200},
        {"label": "extra", "color": "#FFFFFF"},              # beyond the image's channels
    ])
    ngff = _ngff(copy)
    scales = [d["coordinateTransformations"][0]["scale"][-1]
              for d in ngff["multiscales"][0]["datasets"]]
    assert scales == pytest.approx([0.25, 0.5, 1.0])         # every level keeps its factor
    channels = ngff["omero"]["channels"]
    assert len(channels) == 2                                # the image's channels only
    assert (channels[0]["label"], channels[0]["window"]) == (
        "Hoechst", {"min": 10, "max": 900, "start": 10, "end": 900})
    assert channels[1]["active"] is False
    assert sorted(p.stat().st_mtime_ns for p in copy.rglob("c/*")) == chunks   # no pixels


@pytest.mark.parametrize("outcome", ["fail_with_a_long_traceback", "succeed"])
def test_the_conversion_worker_ends_on_a_long_failure(outcome, monkeypatch):
    """The Convert page's worker waited for its conversion process to exit before
    reading the result; a process cannot exit before what it put on a queue is read,
    so a batch that failed with a long traceback hung the GUI forever (found 2026-10-16
    on a batch whose rows all failed; 0.1.x too)."""
    import threading

    from tests import _gui_worker_helpers as helpers
    from tests.conftest import qt_available
    if not qt_available():
        pytest.skip("PyQt6 unavailable or no usable Qt platform plugin")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from eubi_bridge.qt_gui.workers import conversion_worker as cw
    monkeypatch.setattr(cw, "_conversion_subprocess", getattr(helpers, outcome))
    monkeypatch.setattr(cw, "_build_kwargs", lambda config: {})
    worker = cw.ConversionWorker({"inputPath": "x", "outputPath": "y"})
    seen = []
    worker.finished.connect(lambda: seen.append("finished"))
    worker.failed.connect(lambda message: seen.append(len(message)))
    runner = threading.Thread(target=worker.run, daemon=True)
    runner.start()
    runner.join(timeout=120)
    assert not runner.is_alive(), "the worker never ended"
    app.processEvents()                  # signals from another thread arrive queued
    assert seen == (["finished"] if outcome == "succeed" else [len("Traceback (most recent call last):\n") + 1_000_000])


def _run_batch(page, app, timeout=600):
    """Press Run Batch and wait for the worker's finished / failed signal."""
    import time
    done = []
    page._on_finished = lambda *a: done.append("finished")
    page._on_failed = lambda message: done.append(f"failed: {message[-2000:]}")
    page._on_batch_run()
    page._worker.finished.disconnect()
    page._worker.failed.disconnect()
    page._worker.finished.connect(lambda: done.append("finished"))
    page._worker.failed.connect(lambda message: done.append(f"failed: {message[-2000:]}"))
    start = time.time()
    while not done and time.time() - start < timeout:
        app.processEvents()
        time.sleep(0.05)
    page._worker.wait(10_000)
    return done[0] if done else "TIMEOUT"


def test_a_gui_batch_runs_every_row_with_its_own_settings(tmp_path, monkeypatch):
    """Batch mode as a user runs it: files queued on the Convert page, cells edited per
    row (colours + windows, dtype + levels, a CZI's scenes, an aggregative group), Run
    Batch pressed; the page's worker spawns its process, which converts the queue as a
    table. Then the same batch onto its own outputs: every row fails, and the run must
    end with that failure (it hung before)."""
    from copy import deepcopy

    from tests.conftest import qt_available
    if not qt_available():
        pytest.skip("PyQt6 unavailable or no usable Qt platform plugin")
    czi = Path(os.environ.get("MICROSCOPY_DATA") or r"C:\Users\oezdemir\Desktop\data\microscopy",
               "formats", "czi", "zenodo-7015307", "S=2_2x2_T=3_Z=4_CH=1.czi")
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))       # never the user's config
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("EUBI_CONFIG_DIR", str(tmp_path / "home" / ".eubi_bridge"))
    monkeypatch.setenv("EUBI_NEW_PATH", "1")
    rng = np.random.default_rng(9)
    src = tmp_path / "in"
    src.mkdir()
    ome = src / "two_channels.ome.tif"
    tifffile.imwrite(str(ome), rng.integers(100, 3000, (2, 6, 200, 240)).astype("u2"),
                     ome=True, photometric="minisblack", metadata={
                         "axes": "CZYX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5,
                         "PhysicalSizeZ": 2.0, "Channel": {"Name": ["DAPI", "GFP"]}})
    stack = src / "stack.tif"
    tifffile.imwrite(str(stack), rng.integers(0, 250, (8, 128, 160)).astype("u1"),
                     imagej=True, metadata={"axes": "ZYX"})
    series = []
    for t in range(3):
        series.append(src / f"series_T{t}.tif")
        tifffile.imwrite(str(series[-1]), np.full((4, 64, 80), 10 * (t + 1), "u2"),
                         imagej=True, metadata={"axes": "ZYX"})

    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from eubi_bridge.qt_gui.pages.convert_page import ConvertPage
    page = ConvertPage()
    out = tmp_path / "out"
    config = page._batch_ui_config()
    page._batch.set_baseline(deepcopy(config))
    rows = [str(ome), str(stack)] + ([str(czi)] if czi.exists() else []) + [str(p) for p in series]
    page._batch.add(config, rows, str(out))
    group = [len(rows) - 3, len(rows) - 2, len(rows) - 1]
    model = page._batch
    model.update_cells([0], "channel_colors", "0,magenta;1,yellow")
    model.update_cells([0], "channel_intensity_limits", "from_array")
    model.update_cells([1], "dtype", "uint16")
    model.update_cells([1], "n_layers", 2)
    if czi.exists():
        model.update_cells([2], "scene_index", "all")
        model.update_cells([2], "as_mosaic", True)
    model.update_cells(group, "aggregative_group", "G")
    model.update_cells(group, "concatenation_axes", "t")
    model.update_cells(group, "time_tag", "T")
    page._refresh_batch_table()

    assert _run_batch(page, app) == "finished"
    stores = sorted(p.name for p in out.glob("*.zarr"))
    expected = ["G_series_T_tset.zarr", "stack.zarr", "two_channels.zarr"]
    if czi.exists():
        expected += ["S=2_2x2_T=3_Z=4_CH=1_scene0.zarr", "S=2_2x2_T=3_Z=4_CH=1_scene1.zarr"]
    assert stores == sorted(expected)

    def ngff(name):
        attrs = dict(zarr.open_group(str(out / name), mode="r").attrs)
        return attrs.get("ome", attrs)

    channels = ngff("two_channels.zarr")["omero"]["channels"]
    assert [(c["label"], c["color"]) for c in channels] == [("DAPI", "FF00FF"), ("GFP", "FFFF00")]
    assert 100 <= channels[0]["window"]["start"] < 200                  # from_array
    stack_group = zarr.open_group(str(out / "stack.zarr"), mode="r")
    assert stack_group["0"].dtype == np.uint16
    assert len(ngff("stack.zarr")["multiscales"][0]["datasets"]) == 2     # n_layers
    joined = np.asarray(zarr.open_group(str(out / "G_series_T_tset.zarr"), mode="r")["0"])
    assert joined.shape[0] == 3 and [int(joined[t].max()) for t in range(3)] == [10, 20, 30]
    assert ngff("G_series_T_tset.zarr")["omero"]["channels"][0]["label"] == "Channel 0"

    result = _run_batch(page, app)                    # again: the outputs exist now
    assert result.startswith("failed") and "Output already exists" in result
