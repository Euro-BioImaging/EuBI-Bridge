"""GUI widgets and windows that no suite touched.

A coverage sweep before 0.1.3 found these imported by no test at all:
``MainWindow``, ``LogWidget``, ``GroupedHeaderView``, ``SidebarBrowser`` and
``settings_dialog``.  Several are load-bearing -- the grouped header is what
makes a wide batch table readable, and the log is the only surface a running
conversion reports to.

The image/viewer/server modules are deliberately left out: they need real
pixel data and a display, which is a different kind of test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.conftest import qt_available


@pytest.fixture
def app():
    if not qt_available():
        pytest.skip("PyQt6 unavailable or no usable Qt platform plugin")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


class TestLogWidget:
    """The only surface a running conversion reports to."""

    def test_appended_lines_are_readable(self, app):
        from eubi_bridge.qt_gui.widgets.log_widget import LogWidget
        widget = LogWidget()
        widget.append_line("hello")
        # Lines are buffered and rendered on a timer, so a headless read has to
        # flush first; text() is the public accessor.
        widget._flush()
        assert "hello" in widget.text()
        widget.deleteLater()

    def test_clear_empties_it(self, app):
        from eubi_bridge.qt_gui.widgets.log_widget import LogWidget
        widget = LogWidget()
        widget.append_line("something")
        widget._flush()
        widget.clear()
        assert widget.text().strip() == ""
        widget.deleteLater()

    def test_many_lines_do_not_raise(self, app):
        """A long conversion emits thousands of lines.

        Each flush renders at most ``_MAX_BATCH`` of them so the UI stays
        responsive, so the tail arrives over several flushes rather than all at
        once -- draining it here is what a running event loop would do.
        """
        from eubi_bridge.qt_gui.widgets.log_widget import LogWidget
        widget = LogWidget()
        for index in range(500):
            widget.append_line(f"line {index}")
        for _ in range(10):
            widget._flush()
        assert "line 499" in widget.text()
        widget.deleteLater()

    def test_the_message_survives_markup(self, app):
        """Colour tags are handled at render time; text() is the raw buffer.

        What matters is that the words a user needs are all still there -- the
        log is the only place a failing conversion explains itself.
        """
        from eubi_bridge.qt_gui.widgets.log_widget import LogWidget
        widget = LogWidget()
        widget.append_line("[bold red]ERROR[/bold red] conversion failed")
        widget._flush()
        text = widget.text()
        assert "ERROR" in text
        assert "conversion failed" in text
        widget.deleteLater()


class TestGroupedHeaderView:
    """Three-row header (tab / group / parameter) over the batch queue."""

    def _table(self, hierarchy):
        from PyQt6.QtWidgets import QTableWidget
        from eubi_bridge.qt_gui.widgets.grouped_header import GroupedHeaderView
        table = QTableWidget(1, len(hierarchy))
        header = GroupedHeaderView(table)
        table.setHorizontalHeader(header)
        header.set_hierarchy(hierarchy)
        return table, header

    def test_it_accepts_a_hierarchy(self, app):
        table, header = self._table([
            ("Conversion", "Chunking", "z_chunk"),
            ("Conversion", "Chunking", "y_chunk"),
        ])
        assert header.count() == 2
        table.deleteLater()

    def test_it_is_taller_than_a_plain_header(self, app):
        """Three stacked rows need the extra height, or the labels are cut."""
        from PyQt6.QtWidgets import QTableWidget
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QHeaderView
        table, header = self._table([("Conversion", "Chunking", "z_chunk")])
        plain = QHeaderView(Qt.Orientation.Horizontal)
        assert header.sizeHint().height() > plain.sizeHint().height()
        table.deleteLater()
        plain.deleteLater()

    def test_empty_hierarchy_is_safe(self, app):
        """Separator columns carry ("", "", "") and must not break painting."""
        table, header = self._table([
            ("Conversion", "Chunking", "z_chunk"),
            ("", "", ""),
            ("Downscaling", "", "n_layers"),
        ])
        assert header.count() == 3
        table.deleteLater()

    def test_it_survives_being_repainted(self, app):
        """paintEvent draws the spanning rows itself; a raise would blank it."""
        table, header = self._table([
            ("Conversion", "Chunking", "z_chunk"),
            ("Conversion", "Chunking", "y_chunk"),
        ])
        table.show()
        app.processEvents()
        header.repaint()
        app.processEvents()
        table.deleteLater()


class TestSidebarBrowser:
    """Picks the input files, so its filters decide what gets converted."""

    @pytest.fixture
    def browser(self, app, tmp_path):
        from eubi_bridge.qt_gui.widgets.sidebar_browser import SidebarBrowser
        for name in ("a.tif", "b.tif", "c.czi", "thumb_a.tif"):
            (tmp_path / name).write_bytes(b"")
        widget = SidebarBrowser()
        widget.navigate_to(str(tmp_path))
        app.processEvents()
        yield widget
        widget.deleteLater()
        app.processEvents()

    def test_it_navigates_to_a_directory(self, browser, tmp_path):
        assert Path(browser.current_path()) == tmp_path

    def test_nothing_is_selected_to_begin_with(self, browser):
        assert browser.selected_paths() == []

    def test_clear_selection_is_safe_when_empty(self, browser):
        browser.clear_selection()
        assert browser.selected_paths() == []

    def test_filters_are_accepted(self, browser, app):
        """Include/exclude accept star patterns, as the Convert page passes them."""
        browser.set_filters("*.tif", "thumb*")
        app.processEvents()
        assert Path(browser.current_path()).exists()


class TestSettingsModule:
    """Theme and font settings, used by the docs screenshot script too."""

    def test_every_palette_has_a_stylesheet(self, app):
        from eubi_bridge.qt_gui.settings_dialog import PALETTES, STYLESHEETS
        assert PALETTES
        for name in PALETTES:
            assert name in STYLESHEETS or STYLESHEETS.get(name) is not None

    def test_the_current_theme_is_a_known_one(self, app):
        from eubi_bridge.qt_gui.settings_dialog import PALETTES, current_theme
        assert current_theme() in PALETTES

    def test_the_font_size_is_sane(self, app):
        from eubi_bridge.qt_gui.settings_dialog import current_font_size
        assert 6 <= current_font_size() <= 32

    def test_the_ui_scale_is_positive(self, app):
        from eubi_bridge.qt_gui.settings_dialog import current_ui_scale
        assert current_ui_scale() > 0

    def test_the_default_theme_is_dark(self, app):
        """The docs screenshots pin this name; renaming it breaks them."""
        from eubi_bridge.qt_gui.settings_dialog import PALETTES
        assert "Dark (default)" in PALETTES


class TestMainWindow:
    """The window that hosts both pages."""

    @pytest.fixture
    def window(self, app):
        from eubi_bridge.qt_gui.main_window import MainWindow
        widget = MainWindow()
        yield widget
        widget.deleteLater()
        app.processEvents()

    def test_it_builds(self, window):
        assert window.windowTitle() == "EuBI-Bridge"

    def test_it_has_convert_and_inspect_tabs(self, window):
        titles = []
        from PyQt6.QtWidgets import QTabWidget
        for tabs in window.findChildren(QTabWidget):
            titles += [tabs.tabText(i) for i in range(tabs.count())]
        assert any("Convert" in t for t in titles)
        assert any("Inspect" in t for t in titles)

    def test_it_accepts_an_initial_path(self, app, tmp_path):
        from eubi_bridge.qt_gui.main_window import MainWindow
        widget = MainWindow(initial_path=str(tmp_path))
        assert widget is not None
        widget.deleteLater()
        app.processEvents()
