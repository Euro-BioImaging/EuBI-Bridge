"""A conversion table whose aggregative groups are partly refused (review cycle 1, #5).

A group the table cannot convert (its rows disagree on a concatenation setting, its tag
is not usable, two of its files land at one place) is set aside: the rest of the table
still converts, and the run ends with exit 1 and a listing of what was not converted,
each table row with its index, its path and the cells the table gives it.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("micro_reader")
pytest.importorskip("ome_zarr_pyramid")
tifffile = pytest.importorskip("tifffile")
zarr = pytest.importorskip("zarr")


def _table(tmp_path, rows):
    src = tmp_path / "in"
    src.mkdir()
    for t in range(3):
        tifffile.imwrite(str(src / f"good_T{t}.tif"), np.full((16, 20), t, "u2"))
    for name in ("x_T0", "y_T0", "c_T0", "c_T1"):
        tifffile.imwrite(str(src / f"{name}.tif"), np.zeros((16, 20), "u2"))
    out = tmp_path / "out"
    lines = ["input_path,output_path,aggregative_group,concatenation_axes,time_tag"]
    lines += [row.format(src=src, out=out) for row in rows]
    table = tmp_path / "table.csv"
    table.write_text("\n".join(lines) + "\n")
    return table, out


GOOD = ["{src}/good_T0.tif,{out},G,t,T", "{src}/good_T1.tif,{out},G,t,T",
        "{src}/good_T2.tif,{out},G,t,T"]


def _run(table):
    from eubi_bridge.ebridge import EuBIBridge
    EuBIBridge().to_zarr(str(table), use_threading=True, max_processes=2, squeeze=False,
                         verbose=False)


def _joined(out):
    (store,) = list(out.glob("*.zarr"))
    return np.asarray(zarr.open_group(str(store), mode="r")["0"])


def test_a_group_whose_rows_disagree_is_set_aside(tmp_path):
    table, out = _table(tmp_path, GOOD + ["{src}/c_T0.tif,{out},C,t,T",
                                          "{src}/c_T1.tif,{out},C,t,Time"])
    from eubi_bridge.utils.errors import TasksFailedError
    with pytest.raises(TasksFailedError) as caught:
        _run(table)
    message = str(caught.value)
    assert "1 of 2 aggregative group(s)" in message
    assert "Group C" in message and "conflicting time_tag" in message
    # both rows, with their table index, path and cells
    assert "row 4 (table index 3):" in message and "c_T0.tif" in message
    assert "row 5 (table index 4):" in message and "time_tag=Time" in message
    assert "Group G" not in message
    np.testing.assert_array_equal(_joined(out)[:, 0, 0, 0, 0], [0, 1, 2])


def test_a_group_whose_plan_is_refused_is_set_aside(tmp_path):
    # x_T0 and y_T0 are both at t=0 of one output: refused while planning
    table, out = _table(tmp_path, GOOD + ["{src}/x_T0.tif,{out},X,t,T",
                                          "{src}/y_T0.tif,{out},X,t,T"])
    from eubi_bridge.utils.errors import TasksFailedError
    with pytest.raises(TasksFailedError) as caught:
        _run(table)
    message = str(caught.value)
    assert "Group X" in message and "x_T0.tif" in message and "y_T0.tif" in message
    assert "row 4 (table index 3):" in message and "row 5 (table index 4):" in message
    np.testing.assert_array_equal(_joined(out)[:, 0, 0, 0, 0], [0, 1, 2])


def test_a_group_with_an_unusable_tag_is_set_aside(tmp_path):
    table, out = _table(tmp_path, GOOD + ["{src}/c_T0.tif,{out},C,t, ",
                                          "{src}/c_T1.tif,{out},C,t, "])
    from eubi_bridge.utils.errors import TasksFailedError
    with pytest.raises(TasksFailedError, match="Group C"):
        _run(table)
    np.testing.assert_array_equal(_joined(out)[:, 0, 0, 0, 0], [0, 1, 2])


def test_the_run_wide_settings_are_not_listed(tmp_path):
    """Only the table's own cells: the run-wide values filled into the other columns
    would bury the rows under dozens of settings the table never gave."""
    table, _ = _table(tmp_path, GOOD + ["{src}/c_T0.tif,{out},C,t,T",
                                        "{src}/c_T1.tif,{out},C,t,Time"])
    from eubi_bridge.utils.errors import TasksFailedError
    with pytest.raises(TasksFailedError) as caught:
        _run(table)
    assert "max_processes" not in str(caught.value)
    assert "squeeze" not in str(caught.value)


def test_a_table_of_good_groups_still_succeeds(tmp_path):
    table, out = _table(tmp_path, GOOD)
    _run(table)
    assert _joined(out).shape == (3, 1, 1, 16, 20)
