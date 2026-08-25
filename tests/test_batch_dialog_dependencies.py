"""Dependent parameters grey out and drop their overrides in the Edit Cells dialog.

Turning auto-chunking on makes the manual per-axis sizes inert.  Leaving an
override in place would show a column that reads as applied while the writer
ignores it, so the dialog clears it — matching the Run tab, which resets the
manual spins on the same toggle.
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("PyQt6.QtWidgets")

from eubi_bridge.qt_gui.core.batch import BatchModel

_KEYS = ["auto_chunk", "target_chunk_mb", "z_chunk", "x_chunk"]


@pytest.fixture(scope="module")
def qapp():
    """Offscreen QApplication shared by the module."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def model() -> BatchModel:
    cfg = {
        "cluster": {}, "reader": {}, "downscaling": {}, "metadata": {},
        "concatenation": {},
        "conversion": {"autoChunk": True, "targetChunkSizeMb": 32, "chunkZ": 96},
    }
    m = BatchModel()
    m.set_baseline(deepcopy(cfg))
    m.add(cfg, ["a.tif", "b.tif"], "/out")
    return m


def _dialog(model, keys=_KEYS):
    from eubi_bridge.qt_gui.widgets.batch_cell_editor import BatchCellEditor
    return BatchCellEditor(model, [0, 1], keys)


def test_manual_chunks_start_inactive_when_auto_is_on(qapp, model):
    dlg = _dialog(model)
    assert dlg._fields["z_chunk"].inactive
    assert not dlg._fields["z_chunk"].editor.isEnabled()
    # The parameter auto-chunking *does* use stays live.
    assert not dlg._fields["target_chunk_mb"].inactive


def test_toggling_parent_flips_dependants_live(qapp, model):
    dlg = _dialog(model)
    dlg._fields["auto_chunk"].editor.setCurrentText("False")
    assert not dlg._fields["z_chunk"].inactive
    assert dlg._fields["z_chunk"].editor.isEnabled()
    # target_chunk_mb is the mirror case — inert once auto is off.
    assert dlg._fields["target_chunk_mb"].inactive


def test_manual_chunk_can_be_set_when_auto_is_off(qapp, model):
    dlg = _dialog(model)
    dlg._fields["auto_chunk"].editor.setCurrentText("False")
    dlg._fields["z_chunk"].editor.setValue(64)
    dlg.apply()
    assert model.cell(model.rows[0], "z_chunk") == 64
    assert "z_chunk" in model.columns()


def test_re_enabling_auto_clears_the_stale_override(qapp, model):
    """The reported requirement: the manual value must not survive."""
    first = _dialog(model)
    first._fields["auto_chunk"].editor.setCurrentText("False")
    first._fields["z_chunk"].editor.setValue(64)
    first.apply()
    assert model.cell(model.rows[0], "z_chunk") == 64

    second = _dialog(model)
    second._fields["auto_chunk"].editor.setCurrentText("True")
    assert second._fields["z_chunk"].inactive
    second.apply()

    assert model.cell(model.rows[0], "z_chunk") is None
    assert "z_chunk" not in model.columns()


def test_inactive_field_reports_why(qapp, model):
    dlg = _dialog(model)
    assert "auto_chunk" in dlg._fields["z_chunk"].editor.toolTip()


def test_mixed_parent_leaves_field_editable(qapp, model):
    """Selected rows disagreeing about the parent must not lock the field.

    The parameter is live for at least one of them, so greying it would make
    that row uneditable through the dialog.
    """
    model.update_cells([1], "auto_chunk", False)
    dlg = _dialog(model)
    assert not dlg._fields["z_chunk"].inactive


def test_clearing_is_per_row(qapp, model):
    """A row still using the parameter keeps its value when another is cleared."""
    model.update_cells([0], "auto_chunk", False)
    model.update_cells([0], "z_chunk", 64)
    model.update_cells([1], "auto_chunk", False)
    model.update_cells([1], "z_chunk", 32)
    # Row 1 goes back to auto-chunking; its override is now inert.
    model.update_cells([1], "auto_chunk", True)

    dlg = _dialog(model)
    dlg.apply()

    assert model.cell(model.rows[0], "z_chunk") == 64      # still manual
    assert model.cell(model.rows[1], "z_chunk") is None    # cleared


def test_choice_parent_toggles_dependants(qapp, model):
    """ome_zarr_version is a choice, not a bool — the wiring must handle both."""
    from eubi_bridge.qt_gui.widgets.batch_cell_editor import BatchCellEditor
    dlg = BatchCellEditor(model, [0], ["ome_zarr_version", "z_shard_coef"])
    dlg._fields["ome_zarr_version"].editor.setCurrentText("0.4")
    assert dlg._fields["z_shard_coef"].inactive
    dlg._fields["ome_zarr_version"].editor.setCurrentText("0.5")
    assert not dlg._fields["z_shard_coef"].inactive


def test_unshown_parameters_can_be_added(qapp, model):
    """A parameter with no column yet is otherwise unreachable."""
    from eubi_bridge.qt_gui.widgets.batch_cell_editor import (
        BatchCellEditor, _ADD_PARAMETER)
    dlg = BatchCellEditor(model, [0], ["dtype"])
    offered = [dlg._add_combo.itemData(i)
               for i in range(dlg._add_combo.count())]
    assert "n_layers" in offered
    assert "dtype" not in offered          # already present

    dlg._add_combo.setCurrentIndex(offered.index("n_layers"))
    assert dlg.added_key == "n_layers"
    assert dlg.result() == _ADD_PARAMETER


def test_reset_button_restores_the_config_value(qapp, model):
    """Replaces the old Inherit checkbox: it fills in the config value.

    Setting a field back to that value already clears the override, so this is
    a shortcut for a number the user would otherwise have to know and retype.
    """
    from eubi_bridge.qt_gui.widgets.batch_cell_editor import BatchCellEditor
    model.update_cells([0], "auto_chunk", False)
    model.update_cells([0], "z_chunk", 64)
    assert model.cell(model.rows[0], "z_chunk") == 64

    dlg = BatchCellEditor(model, [0], ["z_chunk"])
    field = dlg._fields["z_chunk"]
    assert field.config_value == model.config_value("z_chunk")
    field._on_reset_to_config()
    dlg.apply()

    assert model.cell(model.rows[0], "z_chunk") is None
    assert "z_chunk" not in model.columns()


def test_tooltip_states_the_config_value(qapp, model):
    """The value behind a blank cell must be visible, not guessed."""
    from eubi_bridge.qt_gui.widgets.batch_cell_editor import BatchCellEditor
    dlg = BatchCellEditor(model, [0], ["dtype"])
    tooltip = dlg._fields["dtype"].editor.toolTip()
    assert "Config value" in tooltip
    # An inert field reports why instead, which takes precedence.
    inert_dlg = BatchCellEditor(model, [0], ["z_chunk"])
    assert "Not used" in inert_dlg._fields["z_chunk"].editor.toolTip()


def test_paths_have_no_reset_button(qapp, model):
    """A path has no config fallback, so there is nothing to reset it to."""
    from eubi_bridge.qt_gui.widgets.batch_cell_editor import BatchCellEditor
    dlg = BatchCellEditor(model, [0], ["input_path", "output_path"])
    for key in ("input_path", "output_path"):
        assert dlg._fields[key].config_value is None
        assert dlg._fields[key].reset_btn.isHidden()


def test_typing_the_config_value_also_clears_the_override(qapp, model):
    """The reset button is a shortcut, not the only route."""
    from eubi_bridge.qt_gui.widgets.batch_cell_editor import BatchCellEditor
    model.update_cells([0], "dtype", "uint16")
    dlg = BatchCellEditor(model, [0], ["dtype"])
    field = dlg._fields["dtype"]
    field.set_value(model.config_value("dtype"))
    field._touch()
    dlg.apply()
    assert model.cell(model.rows[0], "dtype") is None
