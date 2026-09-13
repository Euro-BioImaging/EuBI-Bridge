"""The Convert page's buttons and toggles, driven as a user drives them.

Every other GUI test in this suite exercises :class:`BatchModel` -- the layer
*below* the handlers.  That left the handlers themselves uncovered, and it is
where the bugs actually were: multi-row Duplicate selected only the last copy
because ``selectRow`` replaces rather than adds, and the fix for it raised
``NameError`` on a missing import.  The model tests passed throughout both.

So these tests call ``_on_*`` methods and assert on widget state, not on the
model alone.  Handlers that open a modal ``QFileDialog`` (Save/Load Batch, the
config buttons) are driven through the path-taking helper underneath instead,
since a modal dialog would hang a headless run.
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.conftest import qt_available


@pytest.fixture
def page():
    """A fresh ConvertPage on a QApplication shared by the whole module."""
    if not qt_available():
        pytest.skip("PyQt6 unavailable or no usable Qt platform plugin")
    from PyQt6.QtWidgets import QApplication
    from eubi_bridge.qt_gui.pages.convert_page import ConvertPage
    app = QApplication.instance() or QApplication([])
    widget = ConvertPage()
    yield widget
    widget.deleteLater()
    app.processEvents()


def _queue(page, tmp_path, count=6):
    """Queue *count* real files, as pressing Add to Batch would."""
    sources = []
    for index in range(count):
        source = tmp_path / f"f{index}.tif"
        source.write_bytes(b"")
        sources.append(str(source))
    config = page._batch_ui_config()
    page._batch.set_baseline(deepcopy(config))
    page._batch.add(config, sources, str(tmp_path / "out"))
    page._refresh_batch_table()
    return sources


def _names(page):
    import os
    return [os.path.basename(row["input_path"]) for row in page._batch.rows]


def _select_rows(page, rows):
    """Click one cell in each of *rows*, which is how a user picks them."""
    from PyQt6.QtCore import QItemSelectionModel
    selection = page._batch_table.selectionModel()
    selection.clearSelection()
    for row in rows:
        selection.select(page._batch_table.model().index(row, 0),
                         QItemSelectionModel.SelectionFlag.Select)


class TestRemoveButton:
    def test_removes_every_selected_row(self, page, tmp_path):
        _queue(page, tmp_path)
        _select_rows(page, [0, 2, 5])
        page._on_batch_remove()
        assert _names(page) == ["f1.tif", "f3.tif", "f4.tif"]

    def test_reports_what_it_did(self, page, tmp_path):
        _queue(page, tmp_path)
        _select_rows(page, [0, 1])
        page._on_batch_remove()
        assert "Removed 2" in page._batch_status.text()

    def test_an_empty_selection_is_refused(self, page, tmp_path):
        _queue(page, tmp_path, 2)
        page._batch_table.clearSelection()
        page._on_batch_remove()
        assert len(page._batch) == 2
        assert "Select a row" in page._batch_status.text()


class TestDuplicateButton:
    def test_each_copy_lands_after_its_original(self, page, tmp_path):
        _queue(page, tmp_path)
        _select_rows(page, [0, 2, 5])
        page._on_batch_duplicate()
        assert _names(page) == ["f0.tif", "f0.tif", "f1.tif", "f2.tif",
                                "f2.tif", "f3.tif", "f4.tif", "f5.tif",
                                "f5.tif"]

    def test_every_copy_stays_selected(self, page, tmp_path):
        """selectRow() replaces the selection, so a loop of it leaves one."""
        _queue(page, tmp_path)
        _select_rows(page, [0, 2, 5])
        page._on_batch_duplicate()
        assert page._selected_batch_rows() == [1, 4, 8]

    def test_the_selection_holds_the_copies(self, page, tmp_path):
        _queue(page, tmp_path)
        _select_rows(page, [0, 2, 5])
        page._on_batch_duplicate()
        names = _names(page)
        assert [names[i] for i in page._selected_batch_rows()] == [
            "f0.tif", "f2.tif", "f5.tif"]

    def test_an_empty_selection_is_refused(self, page, tmp_path):
        _queue(page, tmp_path, 2)
        page._batch_table.clearSelection()
        page._on_batch_duplicate()
        assert len(page._batch) == 2


class TestMoveAndClear:
    def test_move_down_reorders(self, page, tmp_path):
        _queue(page, tmp_path, 3)
        _select_rows(page, [0])
        page._on_batch_move(1)
        assert _names(page)[:2] == ["f1.tif", "f0.tif"]

    def test_move_up_at_the_top_is_a_no_op(self, page, tmp_path):
        _queue(page, tmp_path, 3)
        _select_rows(page, [0])
        page._on_batch_move(-1)
        assert _names(page) == ["f0.tif", "f1.tif", "f2.tif"]

    def test_clear_empties_the_queue(self, page, tmp_path):
        _queue(page, tmp_path, 3)
        page._on_batch_clear()
        assert len(page._batch) == 0
        assert page._batch_table.rowCount() == 0


class TestAddToBatchRefusals:
    """Add does nothing without inputs or an output, and has to say why.

    The message used to go only to the batch log, which lives in a sub-tab of
    a tab that is not even attached in Run mode -- so Add read as silently
    broken.
    """

    def test_no_selection_is_reported(self, page):
        page._output_edit.setText("/tmp/out")
        page._on_add_to_batch()
        assert len(page._batch) == 0
        assert "no files are selected" in page._batch_status.text()

    def test_no_output_path_is_reported(self, page, tmp_path):
        source = tmp_path / "a.tif"
        source.write_bytes(b"")
        page._browser.selected_paths = lambda: [str(source)]
        page._output_edit.setText("")
        page._on_add_to_batch()
        assert len(page._batch) == 0
        assert "no output path" in page._batch_status.text()

    def test_the_refusal_is_visible_in_run_mode(self, page):
        """Run mode has no Batch tab, so the Run log is the visible surface."""
        page._batch_mode.setChecked(False)
        page._output_edit.setText("")
        page._on_add_to_batch()
        # LogWidget buffers lines and renders them on a timer, so a headless
        # test has to flush before reading the rendered text.
        page._log._flush()
        assert "ERROR" in page._log._text.toPlainText()


class TestBatchModeToggle:
    def test_it_swaps_the_execution_tab(self, page):
        from eubi_bridge.qt_gui.pages.convert_page import _LAST_TAB
        assert page._tabs.tabText(_LAST_TAB) == "Run"
        page._batch_mode.setChecked(True)
        assert page._tabs.tabText(_LAST_TAB) == "Batch"
        page._batch_mode.setChecked(False)
        assert page._tabs.tabText(_LAST_TAB) == "Run"

    @pytest.mark.parametrize("tab_index", [0, 2, 4])
    def test_it_leaves_the_current_tab_alone(self, page, tab_index):
        """Toggling a mode must not yank the user off the tab they are editing."""
        page._tabs.setCurrentIndex(tab_index)
        page._batch_mode.setChecked(True)
        assert page._tabs.currentIndex() == tab_index
        page._batch_mode.setChecked(False)
        assert page._tabs.currentIndex() == tab_index

    def test_the_execution_tab_follows_the_mode(self, page):
        """Being *on* the execution tab is the one case that should follow."""
        from eubi_bridge.qt_gui.pages.convert_page import _LAST_TAB
        page._tabs.setCurrentIndex(_LAST_TAB)
        page._batch_mode.setChecked(True)
        assert page._tabs.currentIndex() == _LAST_TAB
        assert page._tabs.tabText(_LAST_TAB) == "Batch"

    def test_the_group_hint_changes_with_the_mode(self, page):
        """A blank group is optional in Run mode and not in Batch mode."""
        assert "optional" in page._concat_group_edit.placeholderText()
        page._batch_mode.setChecked(True)
        assert "required" in page._concat_group_edit.placeholderText()
        page._batch_mode.setChecked(False)
        assert "optional" in page._concat_group_edit.placeholderText()


class TestDisclosureToggles:
    """Ranges and concatenation hide independently: cropping a single file has
    nothing to do with joining a series."""

    def test_both_start_hidden(self, page):
        page.show()
        assert page._range_group.isHidden()
        assert page._concat_group.isHidden()

    def test_ranges_toggle_alone(self, page):
        page.show()
        page._show_ranges.setChecked(True)
        assert not page._range_group.isHidden()
        assert page._concat_group.isHidden()

    def test_concatenation_toggles_alone(self, page):
        page.show()
        page._show_concat.setChecked(True)
        assert not page._concat_group.isHidden()
        assert page._range_group.isHidden()

    def test_a_loaded_range_reveals_only_ranges(self, page):
        """A hidden box still applies, so anything set must be on screen."""
        page.show()
        page._range_edits["z"].setText("0,10")
        page._reveal_advanced_conv_if_set()
        assert not page._range_group.isHidden()
        assert page._concat_group.isHidden()

    def test_loaded_concat_settings_reveal_only_concatenation(self, page):
        page.show()
        page._load_config_to_ui({"concatenation": {"concatenationAxes": "z"}})
        assert not page._concat_group.isHidden()
        assert page._range_group.isHidden()

    def test_a_plain_config_leaves_both_shut(self, page):
        page.show()
        page._load_config_to_ui({"conversion": {}})
        assert page._range_group.isHidden()
        assert page._concat_group.isHidden()


class TestQueueViewToggles:
    def test_a_category_adds_its_columns(self, page, tmp_path):
        _queue(page, tmp_path, 2)
        before = page._batch_table.columnCount()
        page._on_batch_tab_toggled("Downscaling", True)
        assert page._batch_table.columnCount() > before

    def test_unticking_a_category_removes_them(self, page, tmp_path):
        _queue(page, tmp_path, 2)
        before = page._batch_table.columnCount()
        page._on_batch_tab_toggled("Downscaling", True)
        page._on_batch_tab_toggled("Downscaling", False)
        assert page._batch_table.columnCount() == before

    def test_full_table_shows_every_parameter(self, page, tmp_path):
        _queue(page, tmp_path, 2)
        before = page._batch_table.columnCount()
        page._on_batch_full_toggled(True)
        assert page._batch_table.columnCount() > before
        assert page._batch.full is True


class TestRunGuards:
    def test_an_empty_batch_will_not_run(self, page):
        page._on_batch_run()
        assert page._worker is None
        assert "empty" in page._batch_status.text()

    def test_a_batch_that_fails_validation_will_not_run(self, page, tmp_path):
        """Concatenation axes with no group convert one-to-one, silently."""
        _queue(page, tmp_path, 2)
        page._batch.update_cells([0, 1], "concatenation_axes", "z")
        page._batch.update_cells([0, 1], "aggregative_group", "")
        page._on_batch_run()
        assert page._worker is None
        assert "Cannot run" in page._batch_status.text()


class TestSaveGuards:
    def test_an_invalid_batch_is_not_written(self, page, tmp_path):
        _queue(page, tmp_path, 2)
        page._batch.update_cells([0, 1], "concatenation_axes", "z")
        page._batch.update_cells([0, 1], "aggregative_group", "")
        target = tmp_path / "batch.csv"
        assert page._batch_save_to(str(target)) is None
        assert not target.exists()

    def test_a_valid_batch_is_written(self, page, tmp_path):
        _queue(page, tmp_path, 2)
        target = tmp_path / "batch.csv"
        assert page._batch_save_to(str(target)) is not None
        assert target.exists()
