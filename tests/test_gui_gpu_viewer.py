"""The Inspect page on the GPU viewer (micro-glancer, Workplan WP14): 3-D views, linked
views, the histogram panel, saving a transform, plates, and the choice between the GPU
and the classic viewer.

micro-glancer is not on PyPI yet: these tests skip when it cannot be imported (put its
``python`` folder on PYTHONPATH to run them).
"""
import json
import os
import time

import numpy as np
import pytest

pytest.importorskip("PyQt6.QtWidgets")
pytest.importorskip("micro_glancer.qt")
ozp = pytest.importorskip("ome_zarr_pyramid")


@pytest.fixture(scope="module")
def app():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def gpu(monkeypatch):
    monkeypatch.setenv("EUBI_VIEWER", "gpu")
    monkeypatch.setenv("MICRO_GLANCER_MODE", "offscreen")


def _write(path, data, axes, scales):
    pyr = ozp.Pyramid().from_arrays([data], axis_order=axes, scales=scales, version="0.5")
    ozp.IO().write_pyramid(pyr, str(path), overwrite=True)
    return str(path)


@pytest.fixture
def volume(tmp_path):
    zz, yy, xx = np.mgrid[0:12, 0:60, 0:80]
    a = ((zz * 300 + yy * 20 + xx * 10) % 4000).astype("uint16")
    data = np.stack([a, 4000 - a])
    return _write(tmp_path / "vol.ome.zarr", data, "czyx", [[1.0, 2.0, 1.0, 1.0]])


def _group(path, attrs):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "zarr.json"), "w") as f:
        json.dump({"zarr_format": 3, "node_type": "group", "attributes": attrs}, f)


@pytest.fixture
def plate(tmp_path):
    root = str(tmp_path / "plate.ome.zarr")
    wells = []
    for k, (path, r, c) in enumerate([("A/1", 0, 0), ("B/2", 1, 1)]):
        for f in range(2):
            data = np.full((30, 40), 1000 * (k + 1) + 100 * f, "uint16")
            _write(os.path.join(root, path, str(f)), data, "yx", [[1.0, 1.0]])
        _group(os.path.join(root, path), {"ome": {"version": "0.5", "well": {
            "images": [{"path": "0"}, {"path": "1"}]}}})
        wells.append({"path": path, "rowIndex": r, "columnIndex": c})
    _group(os.path.join(root, "A"), {})
    _group(os.path.join(root, "B"), {})
    _group(root, {"ome": {"version": "0.5", "plate": {
        "rows": [{"name": "A"}, {"name": "B"}], "columns": [{"name": "1"}, {"name": "2"}],
        "wells": wells}}})
    return root


def _pump(app, seconds=0.5):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        app.processEvents()
        time.sleep(0.01)


def _page(app, path):
    from eubi_bridge.qt_gui.pages.inspect_page import InspectPage
    page = InspectPage()
    page.resize(1400, 900)
    page.show()
    page._tabs.setCurrentIndex(1)
    _pump(app, 0.3)
    page._load_zarr(path)
    _pump(app)
    return page


def test_the_page_uses_the_gpu_viewer(app, volume):
    from eubi_bridge.qt_gui.widgets.gpu_viewer import GpuZarrViewer
    page = _page(app, volume)
    viewer = page._viewer
    assert isinstance(viewer, GpuZarrViewer)
    v = viewer.view.viewer
    assert page._pyr is not None and page._path == volume
    assert viewer.axes == "czyx" and viewer.shape == [2, 12, 60, 80]
    assert len(viewer.channel_panel.channel_data()) == 2
    assert "Level" in page._status_bar.text()
    # planes: z as a slider
    slider, _ = viewer._rows["z"]
    slider.setValue(7)
    assert viewer.z == 7
    # orientation, linked views, the 3-D view and what it shows
    viewer.orient["XZ"].trigger()
    assert viewer.orientation == "XZ"
    viewer.orient["XY"].trigger()
    viewer.linked.trigger()
    assert v.linked
    viewer.linked.trigger()
    viewer.three_d.trigger()
    assert v.mip
    viewer.mode.setCurrentIndex(1)
    assert v.projection == "volume"
    viewer.density.setValue(50)
    assert v.density == pytest.approx(10 ** -1.5, rel=1e-6)
    viewer.shading.setChecked(False)
    assert not v.shading
    _pump(app, 0.2)
    assert "3D: volume" in viewer.status_text()
    # keys change the view: the toolbar follows
    v.mip = False
    viewer.view._changed()
    _pump(app, 0.1)
    assert not viewer.three_d.isChecked()
    page.close()


def test_histogram_and_channels_stay_in_step(app, volume):
    page = _page(app, volume)
    viewer = page._viewer
    v = viewer.view.viewer
    _pump(app, 0.5)
    viewer.histogram_panel.refresh()
    assert viewer.histogram_panel.hist._hist is not None
    # contrast set on micro-glancer's side flows into eubi's panel ("Save Channels")
    v.set_channel(0, lo=100.0, hi=900.0, layer=0)
    viewer.view._changed()
    _pump(app, 0.2)
    row = viewer.channel_panel.channel_data()[0]
    assert (row["intensityMin"], row["intensityMax"]) == (100.0, 900.0)
    # and eubi's panel drives the viewer
    viewer.channel_panel.set_channel_range(1, 50.0, 500.0)
    _pump(app, 0.5)
    c = v.layers[0]["channels"][1]
    assert (c["lo"], c["hi"]) == (50.0, 500.0)
    page.close()


def test_saving_a_transform(app, volume):
    page = _page(app, volume)
    viewer = page._viewer
    v = viewer.view.viewer
    v.move_layer(10, 0, layer=0)
    assert "moved" in viewer.status_text()
    shift = v.transform_matrix(0)
    viewer._save_transform()
    _pump(app, 0.3)
    with open(os.path.join(volume, "zarr.json")) as f:
        doc = json.load(f)
    t = doc["attributes"]["ome"]["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    translation = [x for x in t if x["type"] == "translation"][0]["translation"]
    assert translation[-1] == pytest.approx(shift[3][4]) and translation[-1] != 0
    assert "moved" not in viewer.status_text()             # reopened where it was saved
    page.close()


def test_a_plate(app, plate):
    page = _page(app, plate)
    viewer = page._viewer
    v = viewer.view.viewer
    assert viewer.is_plate and v.layers[0]["plate"]
    assert page._pyr is not None                            # the metadata of a field
    assert viewer._contents_box.isVisibleTo(viewer)
    top = viewer.contents.view.topLevelItem(0)
    assert top.childCount() == 2
    spin, _ = viewer._rows["field"]
    # every field of each well (at its position) first, one field on request
    assert v.plate(layer=0)["field"] is None and spin.text() == "all"
    assert "plate: every field" in viewer.status_text()
    spin.setValue(1)
    assert v.plate(layer=0)["field"] == 1
    assert "plate: field 1" in viewer.status_text()
    # saving metadata on a plate is refused (it lives in each field)
    page._ch_confirm_cb.setChecked(True)
    page._on_save_channels()
    assert "field" in page._ch_status_lbl.text()
    # a field opens from the tree; the tree stays, the metadata follows
    field = top.child(1).child(0)
    viewer.contents.view.itemActivated.emit(field, 0)
    _pump(app, 0.3)
    assert not viewer.is_plate and page._path.replace("\\", "/").endswith("B/2/0")
    assert viewer.contents.view.topLevelItem(0).childCount() == 2
    page.close()


def test_the_classic_viewer_when_chosen(app, volume, monkeypatch):
    monkeypatch.setenv("EUBI_VIEWER", "classic")
    from eubi_bridge.qt_gui.widgets.zarr_viewer import ZarrViewer
    from eubi_bridge.qt_gui.pages.inspect_page import InspectPage
    page = InspectPage()
    assert isinstance(page._viewer, ZarrViewer)
    page.close()


def test_the_viewer_setting_round_trips(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    from eubi_bridge.qt_gui import settings_dialog as sd
    sd.set_viewer("classic")
    with open(os.path.join(tmp_path, "EuBI-Bridge", "settings.json")) as f:
        assert json.load(f)["viewer"] == "classic"
    sd._current_viewer = "auto"
    sd._load_settings()
    assert sd.current_viewer() == "classic"
    monkeypatch.delenv("EUBI_VIEWER")
    from eubi_bridge.qt_gui.widgets.viewers import viewer_choice
    assert viewer_choice() == "classic"
    sd.set_viewer("auto")
    assert viewer_choice() == "gpu"


def test_a_collection_opens_with_every_image_at_its_position(app, tmp_path):
    # a bioformats2raw collection of two tiles 50 apart: opened whole, each where it sits
    root = str(tmp_path / "tiles.ome.zarr")
    for s in range(2):
        data = np.full((30, 40), 1000 + 100 * s, "uint16")
        path = _write(os.path.join(root, str(s)), data, "yx", [[1.0, 1.0]])
        doc_path = os.path.join(path, "zarr.json")
        with open(doc_path) as f:
            doc = json.load(f)
        for ds in doc["attributes"]["ome"]["multiscales"][0]["datasets"]:
            ts = [t for t in ds["coordinateTransformations"] if t["type"] != "translation"]
            ds["coordinateTransformations"] = ts + [
                {"type": "translation", "translation": [0.0, 50.0 * s]}]
        with open(doc_path, "w") as f:
            json.dump(doc, f)
    _group(root, {"ome": {"version": "0.5", "bioformats2raw.layout": 3}})
    page = _page(app, root)
    viewer = page._viewer
    v = viewer.view.viewer
    m = v.members(layer=0)
    assert m["names"] == ["series 0", "series 1"] and not m["gridded"]
    assert "2 images at their positions" in viewer.status_text()
    assert page._pyr is not None                            # the metadata of series 0
    # the second tile 50 to the right: its pixel at world (60, 10) has its value
    v.look_at(45.0, 15.0, 1.0)
    v.settle(30.0)
    w, h = v.size
    q = v.pick(w / 2 + 15.0, h / 2 - 5.0)
    assert q["member"] == ("series 1", "1") and q["values"][0]["value"] == 1100
    # its transform is not saved: no one image holds the collection
    said = []
    viewer.status_changed.connect(said.append)
    viewer._save_transform()
    assert any("collection" in text for text in said), said
    page.close()
