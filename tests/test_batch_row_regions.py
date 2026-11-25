"""A region per batch row, and what a row is (Workplan E33; decisions E-10, E-11).

A row is a file plus a selection of its images (series_index, view and illumination
indices). Rows expand into one row per image on demand (Expand) and when a region is
set on a row that writes several images. A region belongs to the output image: an
aggregative group's joined image (its rows must agree), each image of a file, the
mosaic or each tile as the mosaic setting says; a plate is refused for now.
"""
from __future__ import annotations

import os
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("micro_reader")
pytest.importorskip("ome_zarr_pyramid")
tifffile = pytest.importorskip("tifffile")
zarr = pytest.importorskip("zarr")

from eubi_bridge.conversion import pyramid_path  # noqa: E402
from eubi_bridge.core.config_models import ConversionJob  # noqa: E402
from eubi_bridge.qt_gui.core.batch import BatchModel  # noqa: E402
from eubi_bridge.utils.errors import InputError  # noqa: E402
from eubi_bridge.utils.path_utils import range_value, resolve_group_ranges  # noqa: E402


def _config(**conversion) -> dict:
    """A minimal GUI config (as test_batch_cell_editing builds it)."""
    conv = {"omeZarrVersion": "0.4", "dataType": "auto", "autoChunk": True,
            "targetChunkSizeMb": 32, "squeezeDimensions": False, "saveOmeXml": False,
            "overwrite": False}
    conv.update(conversion)
    return {"cluster": {}, "reader": {}, "downscaling": {}, "metadata": {},
            "concatenation": {}, "conversion": conv}


def _batch(paths, output="/out", **conversion) -> BatchModel:
    model = BatchModel()
    cfg = _config(**conversion)
    model.set_baseline(deepcopy(cfg))
    model.add(cfg, [str(p) for p in paths], str(output))
    return model


# -- the images a row writes (pyramid_path.list_images) --------------------------------

class _Opened:
    """A file as micro-reader opens it: its index, nothing else."""

    def __init__(self, indices, is_plate=False):
        self.indices = indices
        if is_plate:
            self.is_plate = True

    def close(self):
        pass


def _listed(monkeypatch, tmp_path, indices, is_plate=False, **readers):
    monkeypatch.setattr(pyramid_path, "_open_micro",
                        lambda job: _Opened(indices, is_plate))
    path = tmp_path / "a.czi"
    path.write_bytes(b"")
    job = ConversionJob.from_kwargs(str(path), str(tmp_path / "out"), readers)
    return pyramid_path.list_images(job)


def _grid(**counts):
    """Every combination of the given index counts, as micro-reader's index lists them."""
    import itertools
    names = list(counts)
    return [dict(zip(names, values))
            for values in itertools.product(*(range(counts[n]) for n in names))]


def test_one_image_per_series(monkeypatch, tmp_path):
    assert _listed(monkeypatch, tmp_path, _grid(series=3)) == [
        {"series_index": 0}, {"series_index": 1}, {"series_index": 2}]


def test_a_row_naming_one_series_writes_one_image(monkeypatch, tmp_path):
    assert _listed(monkeypatch, tmp_path, _grid(series=3), series_index="1,2") == [
        {"series_index": 1}, {"series_index": 2}]
    assert _listed(monkeypatch, tmp_path, _grid(series=3), series_index=2) == [
        {"series_index": 2}]


def test_views_written_apart_are_an_image_each(monkeypatch, tmp_path):
    assert _listed(monkeypatch, tmp_path, _grid(series=1, view=2, illumination=2)) == [
        {"series_index": 0, "view_index": v, "illumination_index": i}
        for v in range(2) for i in range(2)]


def test_joined_views_are_one_image(monkeypatch, tmp_path):
    indices = _grid(series=1, view=2, illumination=2)
    assert _listed(monkeypatch, tmp_path, indices, concat_views=True) == [
        {"series_index": 0, "illumination_index": 0},
        {"series_index": 0, "illumination_index": 1}]
    assert _listed(monkeypatch, tmp_path, indices, concat_views=True,
                   concat_illuminations=True) == [{"series_index": 0}]


def test_tiles_never_split_a_row(monkeypatch, tmp_path):
    indices = _grid(series=2, tile=4)
    for as_mosaic in (True, False):
        assert _listed(monkeypatch, tmp_path, indices, as_mosaic=as_mosaic) == [
            {"series_index": 0}, {"series_index": 1}]


def test_a_plate_is_not_listed(monkeypatch, tmp_path):
    assert _listed(monkeypatch, tmp_path, _grid(series=2), is_plate=True) is None


def test_a_file_micro_reader_does_not_read_is_refused(monkeypatch, tmp_path):
    def refuse(job):
        raise pyramid_path.NotYet("a format micro-reader does not read")
    monkeypatch.setattr(pyramid_path, "_open_micro", refuse)
    path = tmp_path / "a.xyz"
    path.write_bytes(b"")
    job = ConversionJob.from_kwargs(str(path), str(tmp_path / "out"), {})
    with pytest.raises(InputError, match="micro-reader does not read it.*by hand"):
        pyramid_path.list_images(job)


def test_view_0_is_named_like_the_others(tmp_path):
    # 0.1.5 wrote view_index 0 as the plain image; an expanded row picks view 0 by it,
    # and must write what 'all' writes for that view
    path = tmp_path / "a.czi"
    indices = _grid(series=1, view=2)
    names = {}
    for option in ("all", 0, 1):
        job = ConversionJob.from_kwargs(str(path), str(tmp_path / "out"),
                                        {"view_index": option})
        plans, _ = pyramid_path._plan_outputs(_Opened(indices), job)
        names[option] = [entry.basename for entry, _, _ in plans]
    assert names["all"] == ["a_view0", "a_view1"]
    assert names[0] == ["a_view0"] and names[1] == ["a_view1"]


# -- the batch: expanding rows ----------------------------------------------------------

def _lister(counts):
    """A stand-in for list_images: {file name: number of series, or None (a plate)}."""
    def lister(job):
        count = counts[os.path.basename(str(job.input_path))]
        return None if count is None else [{"series_index": s} for s in range(count)]
    return lister


def test_expanding_a_row_gives_one_row_per_image(tmp_path):
    model = _batch(["a.czi", "b.czi"])
    model.update_cells([0], "dtype", "uint16")
    rows, notes = model.expand(0, _lister({"a.czi": 3}))
    assert rows == [0, 1, 2] and notes == []
    assert [r["series_index"] for r in model.rows[:3]] == [0, 1, 2]
    assert all(r["dtype"] == "uint16" for r in model.rows[:3])       # the rest is copied
    assert model.rows[3]["input_path"] == "b.czi"
    # the selection is the row's own, so it is shown and saved
    assert all(model.cell(r, "series_index") is not None for r in model.rows[:3])


def test_a_row_of_one_image_is_kept(tmp_path):
    model = _batch(["a.czi"])
    assert model.expand(0, _lister({"a.czi": 1})) == ([0], [])
    assert len(model) == 1


def test_expanded_rows_survive_the_csv(tmp_path):
    model = _batch(["a.czi"], output=tmp_path / "out")
    model.expand(0, _lister({"a.czi": 2}))
    model.update_cells([1], "x_range", "2,12")
    csv_path = model.save(str(tmp_path / "batch.csv"))
    back = BatchModel.load(csv_path)
    assert [str(r["series_index"]) for r in back.rows] == ["0", "1"]
    assert back.rows[1]["x_range"] == "2,12" and "x_range" not in back.rows[0]


def test_a_group_row_and_a_plate_are_not_expanded(tmp_path):
    model = _batch(["a.czi", "plate.xml"])
    model.update_cells([0], "aggregative_group", "G")
    with pytest.raises(ValueError, match="joined image"):
        model.expand(0, _lister({"a.czi": 3}))
    with pytest.raises(ValueError, match="plate is written whole"):
        model.expand(1, _lister({"plate.xml": None}))
    assert len(model) == 2


def test_a_folder_row_becomes_its_files_mirrored(tmp_path):
    folder = tmp_path / "data"
    (folder / "sub").mkdir(parents=True)
    tifffile.imwrite(str(folder / "a.tif"), np.zeros((8, 8), "u2"))
    tifffile.imwrite(str(folder / "sub" / "b.tif"), np.zeros((8, 8), "u2"))
    out = tmp_path / "out"
    model = _batch([folder], output=out)
    rows, notes = model.expand(0, _lister({"a.tif": 1, "b.tif": 2}))
    assert rows == [0, 1, 2] and notes == []
    got = [(os.path.basename(r["input_path"]), r["output_path"], r.get("series_index"))
           for r in model.rows]
    assert got == [("a.tif", str(out / "data"), "all"),
                   ("b.tif", str(out / "data" / "sub"), 0),
                   ("b.tif", str(out / "data" / "sub"), 1)]


def test_a_region_expands_the_rows_it_needs_and_refuses_plates():
    model = _batch(["a.czi", "g1.czi", "plate.xml", "one.czi"])
    model.update_cells([1], "aggregative_group", "G")
    targets, notes, refused = model.prepare_region(
        [0, 1, 2, 3], _lister({"a.czi": 3, "plate.xml": None, "one.czi": 1}))
    # a.czi -> rows 0..2; the group row and the one-image row keep theirs
    assert [os.path.basename(model.rows[i]["input_path"]) for i in targets] == [
        "a.czi", "a.czi", "a.czi", "g1.czi", "one.czi"]
    assert [model.rows[i]["input_path"] for i in refused] == ["plate.xml"]
    assert any("expanded into 3 rows" in n for n in notes)
    assert any("plate.xml" in n and "plate is written whole" in n for n in notes)


def test_a_row_whose_images_cannot_be_listed_keeps_the_region_with_a_note():
    def lister(job):
        raise InputError("not readable here")
    model = _batch(["a.czi"])
    targets, notes, refused = model.prepare_region([0], lister)
    assert targets == [0] and refused == []
    assert "could not be listed" in notes[0]


# -- the batch: checking regions ---------------------------------------------------------

def test_range_cells_are_checked():
    assert range_value("0,10") == (0, 10) and range_value((2, 5)) == (2, 5)
    assert range_value("") is None and range_value(float("nan")) is None
    for bad in ("10,0", "5,5", "-1,4", "a,b", "1", "0,5,2", "0.5,3"):
        with pytest.raises(ValueError):
            range_value(bad)


def test_validate_names_a_bad_range_from_a_csv(tmp_path):
    src = tmp_path / "a.tif"
    tifffile.imwrite(str(src), np.zeros((8, 8), "u2"))
    model = _batch([src], output=tmp_path / "out")
    model.rows[0]["z_range"] = "5,1"                  # as a hand-written CSV gives it
    assert any("Row 1: Z range '5,1'" in p for p in model.validate())


def test_a_group_must_give_one_region(tmp_path):
    srcs = []
    for t in range(2):
        srcs.append(tmp_path / f"img_T{t}.tif")
        tifffile.imwrite(str(srcs[-1]), np.zeros((8, 8), "u2"))
    model = _batch(srcs, output=tmp_path / "out")
    model.update_cells([0, 1], "aggregative_group", "G")
    model.update_cells([0, 1], "concatenation_axes", "t")
    model.update_cells([0, 1], "time_tag", "T")
    model.update_cells([0, 1], "y_range", "0,4")
    assert not [p for p in model.validate() if "conflicting" in p]
    model.update_cells([1], "y_range", "0,6")
    assert any("Rows 1, 2" in p and "conflicting y_range" in p for p in model.validate())


def test_rows_of_one_file_collide_only_where_their_images_do(tmp_path):
    src = tmp_path / "a.tif"
    tifffile.imwrite(str(src), np.zeros((8, 8), "u2"))
    model = _batch([src, src], output=tmp_path / "out")
    model.update_cells([0], "series_index", "0")
    model.update_cells([1], "series_index", "1")
    assert not [p for p in model.validate() if "same output target" in p]
    model.update_cells([1], "series_index", "all")
    assert any("Row 2: same output target as row 1" in p for p in model.validate())


def test_group_ranges_blank_means_the_runs_own():
    import pandas as pd
    table = pd.DataFrame({"x_range": [None, "0,5"]})
    resolve_group_ranges("G", table, {"x_range": (0, 5)})          # agree
    with pytest.raises(ValueError, match="conflicting x_range: none, '0,5'"):
        resolve_group_ranges("G", table, {})


# -- the run ---------------------------------------------------------------------------

def test_a_region_on_a_plate_is_refused():
    job = SimpleNamespace(readers=SimpleNamespace(**pyramid_path.INDEX_OPTIONS),
                          conversion=SimpleNamespace(time_range=None, channel_range=None,
                                                     z_range=(0, 5), y_range=None,
                                                     x_range=None))
    with pytest.raises(InputError, match=r"a region \(--z_range 0,5\) is not applied to a "
                                         r"plate.*--as_plate False"):
        pyramid_path._refuse_indices_on_a_plate(_Opened([]), job, "plate.xml")


def test_a_tables_range_cells_reach_each_job(tmp_path, monkeypatch):
    # the command line's table input: a "start,stop" cell per row (E33)
    import eubi_bridge.conversion.dispatcher as dispatcher
    from eubi_bridge.ebridge import EuBIBridge
    captured = {}
    monkeypatch.setattr(dispatcher, "dispatch_unary_jobs",
                        lambda jobs: captured.setdefault("jobs", jobs))
    for name in ("a", "b"):
        tifffile.imwrite(str(tmp_path / f"{name}.tif"), np.zeros((8, 8), "u2"))
    table = tmp_path / "table.csv"
    table.write_text("input_path,x_range,y_range\n"
                     f"{tmp_path / 'a.tif'},\"0,4\",\n"
                     f"{tmp_path / 'b.tif'},,\"2,6\"\n")
    EuBIBridge().to_zarr(str(table), str(tmp_path / "out"), verbose=False)
    got = {os.path.basename(j.input_path): (j.conversion.x_range, j.conversion.y_range)
           for j in captured["jobs"]}
    assert got == {"a.tif": ("0,4", None), "b.tif": (None, "2,6")}


def test_a_group_whose_rows_give_different_regions_is_set_aside(tmp_path):
    from eubi_bridge.ebridge import EuBIBridge
    from eubi_bridge.utils.errors import TasksFailedError
    src = tmp_path / "in"
    src.mkdir()
    for t in range(2):
        tifffile.imwrite(str(src / f"g_T{t}.tif"), np.full((16, 20), t, "u2"))
        tifffile.imwrite(str(src / f"h_T{t}.tif"), np.full((16, 20), t, "u2"))
    out = tmp_path / "out"
    table = tmp_path / "table.csv"
    table.write_text(
        "input_path,output_path,aggregative_group,concatenation_axes,time_tag,y_range\n"
        f"{src / 'g_T0.tif'},{out},G,t,T,\"0,8\"\n"
        f"{src / 'g_T1.tif'},{out},G,t,T,\"0,8\"\n"
        f"{src / 'h_T0.tif'},{out},H,t,T,\"0,8\"\n"
        f"{src / 'h_T1.tif'},{out},H,t,T,\"0,9\"\n")
    with pytest.raises(TasksFailedError) as caught:
        EuBIBridge().to_zarr(str(table), use_threading=True, max_processes=2,
                             squeeze=False, verbose=False)
    message = str(caught.value)
    assert "Group H" in message and "conflicting y_range" in message
    assert "Group G" not in message
    (store,) = list(out.glob("*.zarr"))
    joined = np.asarray(zarr.open_group(str(store), mode="r")["0"])
    assert joined.shape[-2:] == (8, 20)                     # the region of the joined image


def test_two_series_of_one_file_with_their_own_regions(tmp_path):
    """Add to batch -> Expand -> a region per row -> the table -> a real conversion."""
    from eubi_bridge.ebridge import EuBIBridge
    rng = np.random.default_rng(0)
    a = rng.integers(0, 4000, (3, 2, 20, 24)).astype("u2")
    b = rng.integers(0, 4000, (2, 16, 18)).astype("u2")
    src = tmp_path / "two.ome.tif"
    with tifffile.TiffWriter(str(src), ome=True) as tw:
        tw.write(a, metadata={"axes": "ZCYX"}, photometric="minisblack")
        tw.write(b, metadata={"axes": "CYX"}, photometric="minisblack")
    out = tmp_path / "out"
    model = _batch([src], output=out)
    rows, _ = model.expand(0)                       # the real list_images
    assert rows == [0, 1]
    assert [r["series_index"] for r in model.rows] == [0, 1]
    model.update_cells([0], "y_range", "0,10")
    model.update_cells([1], "x_range", "2,12")
    assert model.validate() == []
    EuBIBridge().to_zarr(model.to_table(), use_threading=True, verbose=False)
    stores = sorted(out.glob("*.zarr"))
    assert [s.name for s in stores] == ["two_scene0.ome.zarr", "two_scene1.ome.zarr"]
    first = np.asarray(zarr.open_group(str(stores[0]), mode="r")["0"])
    second = np.asarray(zarr.open_group(str(stores[1]), mode="r")["0"])
    # size-1 axes squeezed or not, the pixels are each series' own region
    np.testing.assert_array_equal(first.squeeze(), a.transpose(1, 0, 2, 3)[:, :, 0:10, :])
    np.testing.assert_array_equal(second.squeeze(), b[:, :, 2:12])
