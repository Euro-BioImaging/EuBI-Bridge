"""The Convert page shows what this installation can do (micro-reader docs/
PLAN.md, eubi-bridge-lite phases C and D3).

eubi-bridge-lite has one reader and no Bio-Formats: the reader choices and
the Bio-Formats settings are hidden, and no visible text mentions
Bio-Formats, bfio, bioio, micro-reader or Java -- there is nothing to choose,
and a section that cannot be used only confuses (Bugra, 2026-10-05).  A
loaded config cannot bring any of it back.  The full installation keeps
everything.  (Whether the GUI runs at all without the standard stack: the GUI
test files, run with it blocked, see tests/test_lite_core.py.)
"""
from __future__ import annotations

import re

import pytest

from tests.conftest import qt_available

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


def _page(monkeypatch, bioformats: bool):
    global _APP
    if not qt_available():
        pytest.skip("Qt is not available")
    from PyQt6.QtWidgets import QApplication

    from eubi_bridge.utils import capabilities
    monkeypatch.setattr(capabilities, "has_bioformats", lambda: bioformats)
    monkeypatch.setattr(capabilities, "has_slurm", lambda: bioformats)   # lite has neither
    _APP = QApplication.instance() or QApplication([])
    from eubi_bridge.qt_gui.pages.convert_page import ConvertPage
    _PAGES.append(ConvertPage())
    return _PAGES[-1]


#: what a lite user should never read on the page
READER_WORDS = re.compile(r"bio-?formats|bfio|bioio|micro-?reader|\bjvm\b|\bjava\b", re.I)


def _shown(widget, page) -> bool:
    """Not hidden on purpose (switching tabs does not count)."""
    from PyQt6.QtWidgets import QStackedWidget
    while widget is not None and widget is not page:
        parent = widget.parentWidget()
        if widget.isHidden() and not isinstance(parent, QStackedWidget):
            return False
        widget = parent
    return True


def _reader_mentions(page) -> list[str]:
    """Every visible text on the page -- labels, titles, tooltips, combo
    items, in every tab -- that names a reader or Java."""
    from PyQt6.QtWidgets import QComboBox, QWidget
    found = []
    for widget in page.findChildren(QWidget):
        if not _shown(widget, page):
            continue
        texts = [getattr(widget, attr)() for attr in ("text", "title", "toolTip")
                 if callable(getattr(widget, attr, None))]
        if isinstance(widget, QComboBox):
            texts += [widget.itemText(i) for i in range(widget.count())]
        found += [t for t in texts if isinstance(t, str) and READER_WORDS.search(t)]
    return found


def _hidden_in_lite(page):
    return (page._reader_backend, page._metadata_reader_row, page._bf_group,
            page._micro_note, page._use_slurm, *page._slurm_rows)


def test_lite_hides_the_readers(monkeypatch):
    page = _page(monkeypatch, bioformats=False)
    assert all(widget.isHidden() for widget in _hidden_in_lite(page))
    # what the hidden controls still send: micro-reader, no Bio-Formats
    assert page._metadata_reader.currentText() == "micro"
    assert page._use_micro_reader.isChecked() and not page._force_bioformats.isChecked()
    # the read concurrency still matters, under a plain name
    assert page._micro_group.title() == "Reading"
    assert page._decode_threads_label.text() == "Read Threads:"
    assert page._decode_threads.isEnabled()


def test_lite_shows_nothing_about_readers(monkeypatch):
    assert _reader_mentions(_page(monkeypatch, bioformats=False)) == []


def test_the_check_sees_the_mentions_where_they_belong(monkeypatch):
    """The scan above is not vacuous: the full page has them."""
    assert len(_reader_mentions(_page(monkeypatch, bioformats=True))) >= 5


def test_a_config_cannot_bring_them_back(monkeypatch):
    """A config written by the full eubi-bridge (bfio, standard pixels, SLURM)."""
    page = _page(monkeypatch, bioformats=False)
    page._load_config_to_ui({
        "cluster": {"useSlurm": True, "useLocalDask": True},
        "reader": {"useMicroReader": False, "forceBioformats": True},
        "metadata": {"metadataReader": "bfio"},
    })
    assert page._metadata_reader.currentText() == "micro"
    assert page._use_micro_reader.isChecked()
    assert not page._force_bioformats.isChecked()
    assert all(widget.isHidden() for widget in _hidden_in_lite(page))
    assert not page._use_slurm.isChecked()                 # the config asked for SLURM
    assert _reader_mentions(page) == []


def test_the_full_installation_keeps_everything(monkeypatch):
    page = _page(monkeypatch, bioformats=True)
    # (the SLURM rows show once SLURM is ticked: tests/test_cluster_backends.py)
    assert not any(widget.isHidden() for widget in _hidden_in_lite(page)[:5])
    for widget in (page._use_micro_reader, page._force_bioformats, page._bf_group,
                   page._use_slurm):
        assert widget.isEnabled()
    assert page._micro_group.title() == "micro-reader Settings"
    page._load_config_to_ui({"metadata": {"metadataReader": "bfio"},
                             "reader": {"useMicroReader": False}})
    assert page._metadata_reader.currentText() == "bfio"
    assert not page._use_micro_reader.isChecked()
