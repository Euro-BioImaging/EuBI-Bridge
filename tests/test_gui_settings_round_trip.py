"""Every setting the Convert page shows survives a save and a load, and reaches the run.

Added 2026-10-09 with the writer / byte-backend / acquisition-metadata / pixel-check
fields. The same tests caught settings the page showed but the config file never kept
(the SLURM fields, max concurrent downscale layers): a save replaces the whole config,
so a field missing from the mapping silently fell back to its default.
"""
from __future__ import annotations

import json

import pytest

from tests.conftest import qt_available


@pytest.fixture
def page():
    if not qt_available():
        pytest.skip("PyQt6 unavailable or no usable Qt platform plugin")
    from PyQt6.QtWidgets import QApplication
    from eubi_bridge.qt_gui.pages.convert_page import ConvertPage
    app = QApplication.instance() or QApplication([])
    widget = ConvertPage()
    yield widget
    widget.deleteLater()
    app.processEvents()


def _fresh(page):
    from eubi_bridge.qt_gui.pages.convert_page import ConvertPage
    return ConvertPage()


def _set(page):
    """Non-default values in every field this round trip covers."""
    from eubi_bridge.utils.capabilities import has_bioformats
    page._writer.setCurrentText("ozp")
    if page._io_backend.count() > 1:
        page._io_backend.setCurrentText("zarrista")
    page._export_acquisition.setCurrentIndex(page._export_acquisition.findData(True))
    page._use_micro_reader.setChecked(True)
    if has_bioformats():
        page._micro_check_pixels.setChecked(True)
    page._max_concurrent_downscale_layers.setValue(5)
    page._slurm_time.setText("02:30:00")
    page._slurm_account.setText("lab42")
    page._slurm_partition.setText("gpu")
    page._slurm_worker_timeout.setValue(123)
    page._slurm_sif_path.setText("/images/eubi.sif")
    page._memory_budget.setValue(12.0)
    page._decode_memory.setValue(0.5)
    page._slurm_worker_memory.setValue(16.0)
    page._region_size_mb.setValue(64.0)


def _saved_and_loaded(page, tmp_path):
    """The page's config through config_manager.action_save, as the config file has it,
    and back as the page receives it."""
    from eubi_bridge.qt_gui.server import config_manager as cm
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    cm.action_save(str(cfg_dir), json.dumps(page._ui_to_config()))
    stored = json.loads((cfg_dir / ".eubi_config.json").read_text())
    return stored, cm._config_to_react(stored)


def test_new_fields_survive_save_and_load(page, tmp_path, capsys):
    from eubi_bridge.utils.capabilities import has_bioformats
    _set(page)
    stored, gui = _saved_and_loaded(page, tmp_path)
    capsys.readouterr()
    conv, cluster, readers = stored["conversion"], stored["cluster"], stored["readers"]
    assert conv["writer"] == "ozp"
    assert conv["io_backend"] == page._io_backend.currentText()
    assert conv["export_acquisition_metadata"] is True
    assert readers["micro_check_pixels"] is has_bioformats()
    assert cluster["max_concurrent_downscale_layers"] == 5
    assert (cluster["slurm_time"], cluster["slurm_account"], cluster["slurm_partition"],
            cluster["slurm_worker_timeout"], cluster["slurm_sif_path"]) == (
        "02:30:00", "lab42", "gpu", 123, "/images/eubi.sif")
    assert (cluster["memory_budget"], cluster["decode_memory"],
            cluster["slurm_worker_memory"], cluster["region_size_mb"]) == (
        "12GB", "0.5GB", "16GB", 64.0)

    other = _fresh(page)
    try:
        other._load_config_to_ui(gui)
        assert other._writer.currentText() == "ozp"
        assert other._io_backend.currentText() == page._io_backend.currentText()
        assert other._export_acquisition.currentData() is True
        assert other._micro_check_pixels.isChecked() is has_bioformats()
        assert other._max_concurrent_downscale_layers.value() == 5
        assert other._slurm_account.text() == "lab42"
        assert other._slurm_worker_timeout.value() == 123
        assert other._memory_budget.value() == 12.0
        assert other._decode_memory.value() == 0.5
        assert other._region_size_mb.value() == 64.0
    finally:
        other.deleteLater()


def test_auto_values_stay_auto(page, tmp_path, capsys):
    stored, gui = _saved_and_loaded(page, tmp_path)               # the defaults
    capsys.readouterr()
    cluster = stored["cluster"]
    assert (cluster["memory_budget"], cluster["decode_memory"], cluster["region_size_mb"]) == (
        "auto", "auto", "auto")
    assert stored["conversion"]["writer"] == "auto"
    assert stored["conversion"]["export_acquisition_metadata"] is None
    assert page._memory_budget.text() == "auto" and page._region_size_mb.text() == "auto"


def test_the_run_receives_them(page):
    from eubi_bridge.qt_gui.workers.conversion_worker import _build_kwargs
    from eubi_bridge.utils.capabilities import has_bioformats
    _set(page)
    kwargs = _build_kwargs(page._ui_to_config())
    assert kwargs["writer"] == "ozp"
    assert kwargs["io_backend"] == page._io_backend.currentText()
    assert kwargs["export_acquisition_metadata"] is True
    assert kwargs["micro_check_pixels"] is has_bioformats()
    assert kwargs["max_concurrent_downscale_layers"] == 5
    assert kwargs["memory_budget"] == "12GB" and kwargs["decode_memory"] == "0.5GB"
    assert kwargs["slurm_account"] == "lab42"


def test_the_eubi_writer_pins_tensorstore(page):
    page._writer.setCurrentText("eubi")
    assert page._io_backend.currentText() == "tensorstore"
    assert not page._io_backend.isEnabled()
    page._writer.setCurrentText("auto")
    assert page._io_backend.isEnabled()


def test_an_older_payload_keeps_the_file_values(tmp_path, capsys):
    """A save from a page that predates a field keeps the file's value, not the default."""
    from eubi_bridge.qt_gui.server import config_manager as cm
    from eubi_bridge.ebridge import EuBIBridge
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    bridge = EuBIBridge(configpath=str(cfg_dir))
    bridge.configure.cluster(slurm_account="lab7", max_concurrent_downscale_layers=6)
    bridge.configure.conversion(writer="eubi")
    cm.action_save(str(cfg_dir), json.dumps({"cluster": {}, "reader": {}, "conversion": {}}))
    capsys.readouterr()
    stored = json.loads((cfg_dir / ".eubi_config.json").read_text())
    assert stored["cluster"]["slurm_account"] == "lab7"
    assert stored["cluster"]["max_concurrent_downscale_layers"] == 6
    assert stored["conversion"]["writer"] == "eubi"


def test_eubi_writer_settings_follow_the_writer(page):
    page._writer.setCurrentText("ozp")
    assert not page._queue_size.isEnabled()
    assert not page._max_concurrent_downscale_layers.isEnabled()
    page._writer.setCurrentText("eubi")
    assert page._queue_size.isEnabled() and page._max_concurrent_downscale_layers.isEnabled()
