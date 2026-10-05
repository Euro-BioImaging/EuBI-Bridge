"""The Convert page shows what this installation can do (micro-reader docs/
PLAN.md, eubi-bridge-lite phase C).

eubi-bridge-lite has neither Bio-Formats nor a dask cluster: the page
disables those options and says why -- also when a loaded config asks for
them -- instead of letting a conversion fail.  The full installation keeps
them all.  (Whether the GUI runs at all without the standard stack: the GUI
test files, run with it blocked, see tests/test_lite_core.py.)
"""
from __future__ import annotations

import pytest

from tests.conftest import qt_available

WHY = "Needs the full eubi-bridge: pip install eubi-bridge"

#: the QApplication, kept alive for the module (a widget outliving it aborts Qt)
_APP = None
_PAGES = []


@pytest.fixture(autouse=True)
def _close_pages():
    yield
    while _PAGES:
        _PAGES.pop().deleteLater()
    if _APP is not None:
        _APP.processEvents()


def _page(monkeypatch, bioformats: bool, cluster: bool):
    global _APP
    if not qt_available():
        pytest.skip("Qt is not available")
    from PyQt6.QtWidgets import QApplication

    from eubi_bridge.utils import capabilities
    monkeypatch.setattr(capabilities, "has_bioformats", lambda: bioformats)
    monkeypatch.setattr(capabilities, "has_cluster", lambda: cluster)
    _APP = QApplication.instance() or QApplication([])
    from eubi_bridge.qt_gui.pages.convert_page import ConvertPage
    _PAGES.append(ConvertPage())
    return _PAGES[-1]


def _items(combo):
    model = combo.model()
    return {combo.itemText(i): model.item(i).isEnabled() for i in range(combo.count())}


def test_lite_disables_bio_formats_and_the_cluster(monkeypatch):
    page = _page(monkeypatch, bioformats=False, cluster=False)
    assert _items(page._metadata_reader) == {"micro": True, "bfio": False, "bioio": False}
    assert page._metadata_reader.currentText() == "micro"
    assert page._use_micro_reader.isChecked() and not page._use_micro_reader.isEnabled()
    assert not page._force_bioformats.isChecked() and not page._force_bioformats.isEnabled()
    assert not page._bf_group.isEnabled()
    for box in (page._use_local_dask, page._use_slurm):
        assert not box.isChecked() and not box.isEnabled()
        assert box.toolTip().startswith(WHY)
    assert page._force_bioformats.toolTip().startswith(WHY)


def test_a_config_cannot_bring_them_back(monkeypatch):
    """A config written by the full eubi-bridge (bfio, standard pixels, SLURM)."""
    page = _page(monkeypatch, bioformats=False, cluster=False)
    page._load_config_to_ui({
        "cluster": {"useSlurm": True, "useLocalDask": True},
        "reader": {"useMicroReader": False, "forceBioformats": True},
        "metadata": {"metadataReader": "bfio"},
    })
    assert page._metadata_reader.currentText() == "micro"
    assert page._use_micro_reader.isChecked()
    assert not page._force_bioformats.isChecked()
    assert not page._use_slurm.isChecked() and not page._use_local_dask.isChecked()
    assert page._use_slurm.toolTip().count(WHY) == 1          # said once, not per load


def test_the_full_installation_keeps_everything(monkeypatch):
    page = _page(monkeypatch, bioformats=True, cluster=True)
    assert all(_items(page._metadata_reader).values())
    for widget in (page._use_micro_reader, page._force_bioformats, page._bf_group,
                   page._use_local_dask, page._use_slurm):
        assert widget.isEnabled()
    page._load_config_to_ui({"metadata": {"metadataReader": "bfio"},
                             "reader": {"useMicroReader": False}})
    assert page._metadata_reader.currentText() == "bfio"
    assert not page._use_micro_reader.isChecked()
