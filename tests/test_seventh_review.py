"""The seventh review's findings outside the tags (those: test_tag_grouping.py)."""
import json
import os

import numpy as np
import pandas as pd
import pytest
import tifffile

from eubi_bridge.conversion import pyramid_path
from eubi_bridge.core.data_manager import abbreviate_units
from eubi_bridge.utils.errors import InputError
from eubi_bridge.utils.path_utils import _without_own_outputs, group_input_rows


def _df(**columns):
    return pd.DataFrame(columns)


# -- the same store chosen twice -------------------------------------------------------------

def test_two_identical_rows_are_one_conversion(tmp_path):
    a = tmp_path / "a.tif"
    tifffile.imwrite(a, np.zeros((8, 8), "u2"))
    out = group_input_rows(_df(input_path=[str(a), str(a)], output_path=["o", "o"]))
    assert len(out) == 1


def test_rows_that_write_something_else_stay(tmp_path):
    a = tmp_path / "a.tif"
    tifffile.imwrite(a, np.zeros((8, 8), "u2"))
    assert len(group_input_rows(_df(input_path=[str(a)] * 2, output_path=["o", "p"]))) == 2
    assert len(group_input_rows(_df(input_path=[str(a)] * 2, output_path=["o", "o"],
                                    series_index=[0, 1]))) == 2


def test_a_blank_cell_and_its_default_are_one_choice(tmp_path):
    a = tmp_path / "a.tif"
    tifffile.imwrite(a, np.zeros((8, 8), "u2"))
    assert len(group_input_rows(_df(input_path=[str(a)] * 2, output_path=["o", "o"],
                                    series_index=[None, "all"]))) == 1


def test_rows_differing_in_as_plate_stay(tmp_path):
    a = tmp_path / "a.tif"
    tifffile.imwrite(a, np.zeros((8, 8), "u2"))
    assert len(group_input_rows(_df(input_path=[str(a)] * 2, output_path=["o", "p"],
                                    as_plate=[True, False]))) == 2


def test_two_identical_rows_run_and_write(tmp_path):
    from eubi_bridge.ebridge import EuBIBridge
    a = tmp_path / "a.tif"
    tifffile.imwrite(a, np.zeros((8, 8), "u2"))
    table = tmp_path / "t.csv"
    table.write_text(f"input_path,output_path\n{a},{tmp_path / 'o'}\n{a},{tmp_path / 'o'}\n")
    EuBIBridge().to_zarr(str(table), str(tmp_path / "other"), use_threading=True)
    assert [p.name for p in (tmp_path / "o").iterdir()] == ["a.zarr"]


# -- overwrite -------------------------------------------------------------------------------

def _store(path, content="old"):
    path.mkdir()
    (path / "data").write_text(content)


def test_a_neighbour_with_a_like_name_is_left_alone(tmp_path):
    store = tmp_path / "x.zarr"
    _store(store)
    _store(tmp_path / "x.writing.zarr", "neighbour")
    with pyramid_path._replaced_on_success(str(store), True) as target:
        os.mkdir(target)
    assert (tmp_path / "x.writing.zarr" / "data").read_text() == "neighbour"


def test_a_killed_runs_leftover_is_cleared_on_the_next_overwrite(tmp_path):
    store = tmp_path / "x.zarr"
    _store(store)
    _store(tmp_path / "x.writing-0123abcd.zarr")
    _store(tmp_path / "x.replaced-0123abcd.zarr")
    with pyramid_path._replaced_on_success(str(store), True) as target:
        os.mkdir(target)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.zarr"]


def test_a_store_in_use_stays_whole(tmp_path):
    store = tmp_path / "x.zarr"
    _store(store)
    real = os.replace

    def busy(src, dst):
        if str(src) == str(store):
            raise PermissionError("in use")
        return real(src, dst)

    pyramid_path.os.replace, keep = busy, real
    try:
        with pytest.raises(InputError, match="stays as it was"):
            with pyramid_path._replaced_on_success(str(store), True) as target:
                os.mkdir(target)
    finally:
        pyramid_path.os.replace = keep
    assert (store / "data").read_text() == "old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.zarr"]


def test_a_failed_swap_puts_the_old_store_back(tmp_path):
    store = tmp_path / "x.zarr"
    _store(store)
    real = os.replace

    def fails_second(src, dst):
        if "writing" in str(src):
            raise OSError("disk")
        return real(src, dst)

    pyramid_path.os.replace = fails_second
    try:
        with pytest.raises(OSError):
            with pyramid_path._replaced_on_success(str(store), True) as target:
                os.mkdir(target)
    finally:
        pyramid_path.os.replace = real
    assert (store / "data").read_text() == "old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.zarr"]


def test_too_little_room_is_said(tmp_path, monkeypatch):
    from collections import namedtuple
    store = tmp_path / "x.zarr"
    _store(store)
    said = []
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(pyramid_path.shutil, "disk_usage", lambda _: Usage(9, 9, 0))
    monkeypatch.setattr(pyramid_path.logger, "warning", said.append)
    pyramid_path._warn_room(str(store))
    assert "--overwrite" in said[0]


# -- what a run wrote is no input ------------------------------------------------------------

def test_an_output_folder_inside_the_input_is_left_out(tmp_path):
    paths = [str(tmp_path / "x.tif"), str(tmp_path / "out" / "x.zarr")]
    assert _without_own_outputs(paths, str(tmp_path / "out"), str(tmp_path)) == paths[:1]


def test_an_output_in_the_input_folder_itself_leaves_the_inputs(tmp_path):
    paths = [str(tmp_path / "x.tif")]
    assert _without_own_outputs(paths, str(tmp_path), str(tmp_path)) == paths


def test_an_output_inside_a_pattern_is_left_out(tmp_path):
    paths = [str(tmp_path / "x.tif"), str(tmp_path / "out" / "x.zarr")]
    assert _without_own_outputs(paths, str(tmp_path / "out"), str(tmp_path / "*")) == paths[:1]


def test_a_store_beside_its_source_in_one_folder_is_its_output(tmp_path):
    paths = [str(tmp_path / "a.tif"), str(tmp_path / "a.zarr"), str(tmp_path / "b.zarr")]
    assert _without_own_outputs(paths, str(tmp_path), str(tmp_path)) == [paths[0], paths[2]]


def test_a_killed_runs_store_is_no_input(tmp_path):
    paths = [str(tmp_path / "x.tif"), str(tmp_path / "x.writing-0123abcd.zarr"),
             str(tmp_path / "y.writing.zarr")]
    assert _without_own_outputs(paths, None, str(tmp_path)) == [paths[0], paths[2]]


# -- units -----------------------------------------------------------------------------------

@pytest.mark.parametrize("unit, short", [("inch", "in"), ("foot", "ft"), ("day", "d"),
                                         ("femtometer", "fm"), ("kilosecond", "ks")])
def test_the_names_ngff_has_are_known(unit, short):
    assert abbreviate_units(unit) == short


def _ome_zarr(path, unit):
    import zarr
    root = zarr.open_group(str(path), mode="w", zarr_format=2)
    root.create_array("0", data=np.ones((2, 4, 16, 16), "u2"), chunks=(1, 4, 16, 16))
    root.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": [{"name": "t", "type": "time", "unit": "second"},
                 {"name": "z", "type": "space", "unit": unit},
                 {"name": "y", "type": "space", "unit": unit},
                 {"name": "x", "type": "space", "unit": unit}],
        "datasets": [{"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": [1.0, 2.0, 0.5, 0.5]},
            {"type": "translation", "translation": [5.0, 7.0, 8.0, 9.0]}]}]}]


def _attrs(store):
    return json.load(open(os.path.join(store, ".zattrs"), encoding="utf-8"))["multiscales"][0]


@pytest.mark.parametrize("unit", ["um", "µm", "micron"])
def test_an_input_unit_comes_out_by_its_ngff_name(tmp_path, unit):
    from eubi_bridge.ebridge import EuBIBridge
    src = tmp_path / "in.zarr"
    _ome_zarr(src, unit)
    EuBIBridge().to_zarr(str(src), str(tmp_path / "o"), use_threading=True, zarr_format=2)
    meta = _attrs(tmp_path / "o" / "in.zarr")
    assert {ax["name"]: ax.get("unit") for ax in meta["axes"]}["x"] == "micrometer"
    moved = [t for t in meta["datasets"][0]["coordinateTransformations"] if t["type"] == "translation"]
    assert moved[0]["translation"] == [0.0, 7.0, 8.0, 9.0]    # the time origin: a warning


def test_a_plate_like_input_with_short_units_keeps_its_sizes_in_the_xml(tmp_path):
    from eubi_bridge.ebridge import EuBIBridge
    src = tmp_path / "in.zarr"
    _ome_zarr(src, "um")
    EuBIBridge().to_zarr(str(src), str(tmp_path / "o"), use_threading=True, zarr_format=2,
                         keep_existing_resolutions=True, save_omexml=True)
    meta = _attrs(tmp_path / "o" / "in.zarr")
    assert {ax["name"]: ax.get("unit") for ax in meta["axes"]}["x"] == "micrometer"


# -- joined runs: what cannot be honoured is refused ----------------------------------------

def test_the_older_readers_refuse_a_view_they_would_swap_for_the_first():
    from types import SimpleNamespace
    from eubi_bridge.conversion.aggregative_path import _Inputs
    readers = SimpleNamespace(view_index=1, illumination_index=0, rotation_index=0, phase_index=0,
                              mosaic_tile_index=None)
    inputs = _Inputs(["a.czi"], SimpleNamespace(readers=readers), 0)
    with pytest.raises(InputError, match="--view_index 1"):
        inputs._refuse_picks("a.czi")
    readers.view_index = "all"
    inputs._refuse_picks("a.czi")
    readers.mosaic_tile_index = 2
    with pytest.raises(InputError, match="--mosaic_tile_index 2"):
        inputs._refuse_picks("a.czi")
