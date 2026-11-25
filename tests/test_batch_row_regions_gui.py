"""The batch tab's Expand button and the Edit Cells range editor (Workplan E33)."""
from __future__ import annotations

import os
from copy import deepcopy

import pytest

from tests.conftest import qt_available

if not qt_available():
    pytest.skip("PyQt6 unavailable or no usable Qt platform plugin", allow_module_level=True)

from eubi_bridge.qt_gui.core import batch as B  # noqa: E402


@pytest.fixture(scope="module")
def page():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from eubi_bridge.qt_gui.pages.convert_page import ConvertPage
    widget = ConvertPage()
    yield widget
    widget.deleteLater()
    app.processEvents()


def _fill(page, paths):
    base = page._ui_to_config()
    page._batch = B.BatchModel()
    page._batch.set_baseline(deepcopy(base))
    page._batch.add(base, paths, "/out")
    page._refresh_batch_table()


def _select(page, rows):
    from PyQt6.QtCore import QItemSelectionModel
    page._batch_table.clearSelection()
    selection = page._batch_table.selectionModel()
    for row in rows:
        selection.select(page._batch_table.model().index(row, 0),
                         QItemSelectionModel.SelectionFlag.Select
                         | QItemSelectionModel.SelectionFlag.Rows)


def test_expand_replaces_the_selected_rows_by_their_images(page, monkeypatch):
    import eubi_bridge.conversion.pyramid_path as pyramid_path
    counts = {"a.czi": 3, "b.czi": 1, "c.czi": 2}
    monkeypatch.setattr(pyramid_path, "list_images", lambda job: [
        {"series_index": s} for s in range(counts[os.path.basename(job.input_path)])])
    _fill(page, ["a.czi", "b.czi", "c.czi"])
    _select(page, [0, 1, 2])
    page._on_batch_expand()
    assert [(os.path.basename(r["input_path"]), r.get("series_index"))
            for r in page._batch.rows] == [("a.czi", 0), ("a.czi", 1), ("a.czi", 2),
                                           ("b.czi", "all"), ("c.czi", 0), ("c.czi", 1)]
    assert page._batch_table.rowCount() == 6
    assert "Expanded into 5 row(s) (3 added)" in page._batch_status.text()
    assert sorted({i.row() for i in page._batch_table.selectionModel().selectedIndexes()}) \
        == [0, 1, 2, 4, 5]


def test_expand_says_why_a_row_stays(page, monkeypatch):
    import eubi_bridge.conversion.pyramid_path as pyramid_path
    monkeypatch.setattr(pyramid_path, "list_images", lambda job: None)    # a plate
    _fill(page, ["plate.xml"])
    _select(page, [0])
    page._on_batch_expand()
    assert len(page._batch) == 1
    assert "plate is written whole" in page._batch_status.text()


def test_the_range_editor_checks_before_it_closes(page):
    from eubi_bridge.qt_gui.widgets.batch_cell_editor import BatchCellEditor
    _fill(page, ["a.czi"])
    dlg = BatchCellEditor(page._batch, [0], ["z_range"], parent=page)
    field = dlg._fields["z_range"]
    field.editor.setText("10,2")
    field._touch()
    assert dlg.problems() and dlg.sets_region()
    dlg.accept()
    assert dlg.result() == 0 and "Z range '10,2'" in dlg._problem.text()
    field.editor.setText("2,10")
    assert not dlg.problems()
    assert dlg.apply() == ["z_range"] and page._batch.rows[0]["z_range"] == "2,10"


def test_a_cleared_range_goes_back_to_the_config(page):
    from eubi_bridge.qt_gui.widgets.batch_cell_editor import BatchCellEditor
    _fill(page, ["a.czi"])
    page._batch.update_cells([0], "x_range", "0,4")
    dlg = BatchCellEditor(page._batch, [0], ["x_range"], parent=page)
    field = dlg._fields["x_range"]
    assert field.editor.text() == "0,4"
    field.editor.setText("")
    field._touch()
    assert not dlg.sets_region()
    dlg.apply()
    assert page._batch.cell(page._batch.rows[0], "x_range") is None


def test_ranges_go_only_to_the_region_rows(page):
    from eubi_bridge.qt_gui.widgets.batch_cell_editor import BatchCellEditor
    _fill(page, ["a.czi", "plate.xml"])
    dlg = BatchCellEditor(page._batch, [0, 1], ["y_range", "dtype"], parent=page)
    dlg._fields["y_range"].editor.setText("0,8")
    dlg._fields["y_range"]._touch()
    dlg._fields["dtype"].editor.setCurrentText("uint16")
    dlg.apply(rows=[0, 1], region_rows=[0])
    assert page._batch.rows[0]["y_range"] == "0,8"
    assert page._batch.cell(page._batch.rows[1], "y_range") is None
    assert [r["dtype"] for r in page._batch.rows] == ["uint16", "uint16"]
