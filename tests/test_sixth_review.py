"""What the sixth review (2026-11-02) found: tags read in folder names, joined runs that ignored
the image asked for, a failed overwrite that left an empty store, rows dropped."""
from __future__ import annotations

import os
import logging
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from eubi_bridge.conversion import aggregative_path, dispatcher, pyramid_path
from eubi_bridge.conversion.fileset_io import FileSet, tag_folder
from eubi_bridge.core import micro_source
from eubi_bridge.core.data_manager import create_ome_xml
from eubi_bridge.utils import path_utils
from eubi_bridge.utils.errors import InputError

pytest.importorskip("micro_reader")
pytest.importorskip("ome_zarr_pyramid")


def _groups(paths, axes, **tags):
    fs = FileSet(paths, shapes=[(1,) * 5] * len(paths), **tags)
    for axis in axes:
        fs.concatenate_along(axis)
    return fs


# -- tags are read below the folder the files share ------------------------------------------

def test_a_folder_named_like_tags_does_not_group_the_files():
    paths = [f"/d/C1_T5run/img_T{t}_C{c}.tif" for t in range(2) for c in range(3)]
    fs = _groups(paths, [0, 1], axis_tag0="T", axis_tag1="C")
    assert fs.shape_dict[paths[0]] == [2, 3, 1, 1, 1]
    assert len(set(fs.path_dict.values())) == 1


def test_joining_time_alone_keeps_the_channels_apart_in_such_a_folder():
    paths = [f"/d/C1_T5run/img_T{t}_C{c}.tif" for t in range(2) for c in range(3)]
    fs = _groups(paths, [0], axis_tag0="T", axis_tag1="C")
    assert len(set(fs.path_dict.values())) == 3
    assert all(shape[0] == 2 for shape in fs.shape_dict.values())


def test_named_channels_in_a_folder_name_do_not_repeat_a_file():
    paths = ["/d/gfp_mcherry/img_gfp.tif", "/d/gfp_mcherry/img_mcherry.tif"]
    fs = _groups(paths, [1], axis_tag1=("gfp", "mcherry"))
    assert len(set(fs.path_dict.values())) == 1
    assert fs.slice_dict[paths[0]][1] == slice(0, 1) and fs.slice_dict[paths[1]][1] == slice(1, 2)


def test_the_output_name_keeps_the_folder_and_reduces_the_file():
    paths = [f"/d/T5run/img_T{t}.tif" for t in range(2)]
    fs = _groups(paths, [0], axis_tag0="T")
    assert set(fs.path_dict.values()) == {"/d/T5run/img_T_tset.tif"}


def test_a_tag_that_is_a_folder_still_counts():
    paths = ["/d/frames/000/a.tif", "/d/frames/001/a.tif"]
    assert tag_folder(paths, "frames") == "/d/"
    fs = _groups(paths, [0], axis_tag0="frames")
    assert fs.shape_dict[paths[0]][0] == 2


def test_tags_in_subfolders_of_the_input_count():
    paths = ["/d/ch1/img.tif", "/d/ch2/img.tif"]
    fs = _groups(paths, [1], axis_tag1="ch")
    assert fs.shape_dict[paths[0]][1] == 2


def test_a_group_of_a_worker_finds_a_tag_its_files_share():
    # the planner saw ch1/ and ch2/; a worker joins time within ch1/, reading each tag below
    # the folder the planner read it below (its tag_folders: a worker does not climb, and a
    # climb stops at the shared folder's own name, review cycle 1 N-02)
    every = [f"/d/ch{c}/sub/a_T{t}.tif" for c in (1, 2) for t in (0, 1)]
    plan = FileSet(every, shapes=[(1,) * 5] * 4, axis_tag0="T", axis_tag1="ch")
    paths = every[:2]
    fs = _groups(paths, [0], axis_tag0="T", axis_tag1="ch", tag_folders=plan.tag_folders)
    assert fs.shape_dict[paths[0]][0] == 2


def test_a_split_by_number_survives_a_split_by_name():
    paths = [f"/d/a_T{t}_{c}_z{z}.tif" for t in range(2) for c in ("gfp", "rfp") for z in range(3)]
    fs = _groups(paths, [2], axis_tag0="T", axis_tag1=("gfp", "rfp"), axis_tag2="z")
    assert all(shape[2] == 3 for shape in fs.shape_dict.values())
    assert len(set(fs.path_dict.values())) == 4


def test_a_file_that_lacks_a_tag_does_not_shift_the_others():
    paths = ["/d/x.tif", "/d/a_T0_C0.tif", "/d/a_T0_C1.tif"]
    groups = FileSet(paths, shapes=[(1,) * 5] * 3, axis_tag0="T", axis_tag1="C")._split_by("C")
    assert groups == {"C0": [paths[1]], "C1": [paths[2]]}


def test_one_file_for_two_tags_is_refused():
    paths = ["/d/a_gfp_mcherry.tif", "/d/b.tif"]
    with pytest.raises(InputError, match="matches two of the tags"):
        _groups(paths, [1], axis_tag1=("gfp", "mcherry"))


def test_acceptance_reads_tags_as_grouping_does():
    paths = ["/d/Tdata/a_T0.tif", "/d/Tdata/a_T1.tif", "/d/Tdata/stray.tif"]
    assert dispatcher._parse_filepaths_with_tags(paths, ["T"]) == paths[:2]
    frames = ["/d/frames/000/a.tif", "/d/frames/001/a.tif"]
    assert dispatcher._parse_filepaths_with_tags(frames, ["frames"]) == frames


# -- the image a joined run takes ------------------------------------------------------------

def _inputs(**readers):
    options = {"as_mosaic": True, "mosaic_tile_index": "all", "view_index": "all",
               "illumination_index": "all", "rotation_index": 0, "phase_index": 0, **readers}
    return aggregative_path._Inputs(["a.czi"], SimpleNamespace(readers=SimpleNamespace(**options)), 0)


def _file_with(monkeypatch, indices):
    @contextmanager
    def using(path, tiles, multifile=True):
        yield SimpleNamespace(file=SimpleNamespace(indices=indices))
    monkeypatch.setattr(micro_source, "_using", using)


LIGHT_SHEET = [{"series": 0, "view": v, "illumination": i} for v in (0, 1) for i in (0, 1)]


def test_several_views_need_a_choice(monkeypatch):
    _file_with(monkeypatch, LIGHT_SHEET)
    with pytest.raises(InputError, match="2 views.*--view_index"):
        _inputs()._picks("a.czi", "mosaic")


def test_the_chosen_view_and_illumination_are_read(monkeypatch):
    _file_with(monkeypatch, LIGHT_SHEET)
    assert _inputs(view_index=1, illumination_index=1)._picks("a.czi", "mosaic") == {
        "view": 1, "illumination": 1, "rotation": 0, "phase": 0}


def test_a_view_the_file_does_not_have_is_refused(monkeypatch):
    _file_with(monkeypatch, LIGHT_SHEET)
    with pytest.raises(InputError, match="no view 5"):
        _inputs(view_index=5, illumination_index=0)._picks("a.czi", "mosaic")


def test_a_file_of_one_view_needs_no_choice(monkeypatch):
    _file_with(monkeypatch, [{"series": 0}])
    assert _inputs()._picks("a.czi", "mosaic") == {"view": 0, "illumination": 0, "rotation": 0,
                                                   "phase": 0}


def test_several_views_named_are_refused(monkeypatch):
    _file_with(monkeypatch, LIGHT_SHEET)
    with pytest.raises(InputError, match="names several"):
        _inputs(view_index="0,1")._picks("a.czi", "mosaic")


def test_the_chosen_tile_is_joined():
    assert _inputs(mosaic_tile_index=2)._chosen_tile() == 2
    assert _inputs(mosaic_tile_index="all")._chosen_tile() is None
    with pytest.raises(InputError, match="names several"):
        _inputs(mosaic_tile_index="1,2")._chosen_tile()


# -- a failed overwrite ----------------------------------------------------------------------

def test_a_failed_overwrite_leaves_the_store_as_it_was(tmp_path):
    store = tmp_path / "x.zarr"
    store.mkdir()
    (store / "good").write_text("good")
    with pytest.raises(RuntimeError):
        with pyramid_path._replaced_on_success(str(store), True) as target:
            os.mkdir(target)
            assert ".writing-" in target
            raise RuntimeError("a scene failed")
    assert (store / "good").read_text() == "good"
    assert not os.path.exists(target)


def test_an_overwrite_replaces_the_store_when_it_succeeds(tmp_path):
    store = tmp_path / "x.zarr"
    store.mkdir()
    (store / "old").write_text("old")
    with pyramid_path._replaced_on_success(str(store), True) as target:
        os.mkdir(target)
        with open(os.path.join(target, "new"), "w") as f:
            f.write("new")
    assert [p.name for p in store.iterdir()] == ["new"]
    assert [p.name for p in tmp_path.iterdir()] == ["x.zarr"]


def test_a_new_store_is_written_where_it_goes(tmp_path):
    store = str(tmp_path / "x.zarr")
    with pyramid_path._replaced_on_success(store, True) as target:
        assert target == store


# -- rows ------------------------------------------------------------------------------------

def test_one_input_to_two_outputs_stays_two_rows(monkeypatch):
    monkeypatch.setattr(path_utils, "input_mains", lambda paths, plates=True: [
        "/plate/index.xml" if p.endswith("plane.tif") else p for p in paths])
    df = pd.DataFrame({"input_path": ["a.lif", "a.lif", "plane.tif"],
                       "output_path": ["o1", "o2", "o3"]})
    assert len(path_utils.group_input_rows(df)) == 3


def test_the_same_row_twice_warns(monkeypatch, caplog):
    monkeypatch.setattr(path_utils, "input_mains", lambda paths, plates=True: [
        "/plate/index.xml" for _ in paths])
    # two files of one plate: one conversion, no warning (each says it is a file of the
    # plate; tenth review N-04); the same input named twice: one conversion and a warning
    for given, warns in ((["a.tif", "b.tif"], False), (["a.tif", "a.tif"], True)):
        caplog.clear()
        df = pd.DataFrame({"input_path": given, "output_path": ["o", "o"]})
        with caplog.at_level(logging.WARNING):
            assert len(path_utils.group_input_rows(df)) == 1
        assert ("more than once" in caplog.text) is warns


# -- units and the OME-XML -------------------------------------------------------------------

def test_a_spelling_of_a_unit_becomes_its_name():
    assert pyramid_path.unit_spelled("um") == "micrometer"
    assert pyramid_path.unit_spelled("microns") == "micrometer"
    assert pyramid_path.unit_spelled("furlong") == "furlong"


def _pyramid(units):
    meta = SimpleNamespace(axis_order="zyx", resolution_paths=["0"], tag="x", unit_list=units,
                           metadata={}, get_scale=lambda path: [2.0, 0.5, 0.5])
    return SimpleNamespace(meta=meta, base_array=np.zeros((2, 4, 4), "u2"))


def test_a_unit_ome_has_no_name_for_leaves_the_size_out(caplog):
    with caplog.at_level(logging.WARNING):
        xml = pyramid_path.ome_xml(_pyramid(["furlong", "micrometer", "micrometer"]), "uint16")
    assert 'PhysicalSizeX="0.5"' in xml and "PhysicalSizeZ=" not in xml
    assert "furlong" in caplog.text


def test_the_ome_xml_passes_the_schema():
    from ome_types import from_xml
    xml = pyramid_path.ome_xml(_pyramid(["micrometer"] * 3), "uint16")
    assert "MetadataOnly" in xml
    assert from_xml(xml, validate=True).images[0].pixels.size_z == 2


def test_refusals_of_workers_reach_the_user_in_one_line():
    with pytest.raises(InputError, match="^a.czi has 2 views"):
        dispatcher._report_failures([{"status": "error",
                                      "error": "InputError: a.czi has 2 views"}])
    with pytest.raises(RuntimeError, match="tasks failed"):
        dispatcher._report_failures([{"status": "error", "error": "ValueError: broken"}])
