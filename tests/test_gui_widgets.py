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


class TestSidebarBrowserDrop:
    """Dropping files is a second way to tick them, feeding the same selection."""

    @pytest.fixture
    def browser(self, app, tmp_path):
        from eubi_bridge.qt_gui.widgets.sidebar_browser import SidebarBrowser
        (tmp_path / "sub").mkdir()
        for name in ("a.tif", "b.czi"):
            (tmp_path / "sub" / name).write_bytes(b"")
        (tmp_path / "sub" / "plain").mkdir()
        (tmp_path / "sub" / "img.zarr").mkdir()
        (tmp_path / "sub" / "img.zarr" / "zarr.json").write_text("{}")
        widget = SidebarBrowser(mode="conversion", initial_path=str(tmp_path))
        app.processEvents()
        yield widget
        widget.deleteLater()
        app.processEvents()

    @staticmethod
    def _drop(widget, *paths):
        """Deliver a drop as Qt does, with forward-slash URLs as Explorer sends."""
        from PyQt6.QtCore import QMimeData, QPointF, Qt, QUrl
        from PyQt6.QtGui import QDropEvent
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(Path(p).as_posix())
                      if not str(p).startswith("http") else QUrl(str(p))
                      for p in paths])
        event = QDropEvent(QPointF(5, 5), Qt.DropAction.CopyAction, mime,
                           Qt.MouseButton.LeftButton,
                           Qt.KeyboardModifier.NoModifier)
        widget.dropEvent(event)
        return event

    @staticmethod
    def _check_state(widget, name):
        for row in range(widget._list.count()):
            item = widget._list.item(row)
            if item.text().endswith(name):
                return item.checkState()
        raise AssertionError(f"{name} not listed")

    def test_dropped_files_are_selected_and_shown_ticked(self, browser, tmp_path):
        from PyQt6.QtCore import Qt
        emitted = []
        browser.selection_changed.connect(emitted.append)
        target = tmp_path / "sub" / "a.tif"

        self._drop(browser, target)

        assert browser.selected_paths() == [str(target)]
        assert emitted and emitted[-1] == [str(target)]
        # The browser opens the folder the drop came from, with the tick shown.
        assert Path(browser.current_path()) == tmp_path / "sub"
        assert self._check_state(browser, "a.tif") == Qt.CheckState.Checked
        assert self._check_state(browser, "b.czi") == Qt.CheckState.Unchecked

    def test_a_file_ticked_by_hand_is_not_added_twice(self, browser, tmp_path):
        """Qt's C:/x form must match the listing's C:\\x form, or it duplicates."""
        from PyQt6.QtCore import Qt
        browser.navigate_to(str(tmp_path / "sub"))
        for row in range(browser._list.count()):
            item = browser._list.item(row)
            if item.text().endswith("a.tif"):
                item.setCheckState(Qt.CheckState.Checked)

        self._drop(browser, tmp_path / "sub" / "a.tif")

        assert len(browser.selected_paths()) == 1

    def test_drops_are_refused_while_filters_are_set(self, browser, tmp_path,
                                                     monkeypatch):
        from eubi_bridge.qt_gui.widgets import sidebar_browser
        warnings = []
        monkeypatch.setattr(sidebar_browser.QMessageBox, "warning",
                            lambda *args: warnings.append(args))
        browser.set_filters("*.tif", "")

        self._drop(browser, tmp_path / "sub" / "a.tif")

        assert browser.selected_paths() == []
        assert len(warnings) == 1

    def test_non_local_urls_are_ignored(self, browser):
        event = self._drop(browser, "https://example.org/a.tif")
        assert not event.isAccepted()
        assert browser.selected_paths() == []

    def test_only_input_and_inspect_browsers_take_drops(self, app, browser):
        """The list must not swallow drops itself, or they never reach the
        handler; the output browser has nothing to drop into."""
        from eubi_bridge.qt_gui.widgets.sidebar_browser import SidebarBrowser
        assert browser.acceptDrops()
        assert not browser._list.viewport().acceptDrops()
        output_browser = SidebarBrowser(mode="output")
        assert not output_browser.acceptDrops()
        output_browser.deleteLater()

    def test_a_plain_folder_cannot_be_ticked(self, browser, tmp_path):
        """Ticking one used to hand the folder to the reader, which failed."""
        from PyQt6.QtCore import Qt
        browser.navigate_to(str(tmp_path / "sub"))
        checkable = {}
        for row in range(browser._list.count()):
            item = browser._list.item(row)
            name = item.text().split(" ", 1)[1]
            checkable[name] = bool(item.flags() & Qt.ItemFlag.ItemIsUserCheckable)
        assert checkable == {"plain": False, "img.zarr": True,
                             "a.tif": True, "b.czi": True}

    def test_select_all_skips_plain_folders(self, browser, tmp_path):
        browser.navigate_to(str(tmp_path / "sub"))
        browser._on_select_all()
        assert sorted(Path(p).name for p in browser.selected_paths()) == [
            "a.tif", "b.czi", "img.zarr"]

    def test_dropped_folders_are_skipped_but_files_kept(self, browser, tmp_path,
                                                        monkeypatch):
        from eubi_bridge.qt_gui.widgets import sidebar_browser
        warnings = []
        monkeypatch.setattr(sidebar_browser.QMessageBox, "warning",
                            lambda *args: warnings.append(args))

        self._drop(browser, tmp_path / "sub" / "plain",
                   tmp_path / "sub" / "a.tif")

        assert browser.selected_paths() == [str(tmp_path / "sub" / "a.tif")]
        assert len(warnings) == 1 and "plain" in warnings[0][2]

    def test_a_dropped_store_is_selected(self, browser, tmp_path):
        """An OME-Zarr store is a folder on disk but a dataset to convert."""
        self._drop(browser, tmp_path / "sub" / "img.zarr")
        assert browser.selected_paths() == [str(tmp_path / "sub" / "img.zarr")]


class TestInspectBrowserDrop:
    """On the Inspect page a drop opens one store, as clicking it does."""

    @pytest.fixture
    def browser(self, app, tmp_path):
        from eubi_bridge.qt_gui.widgets.sidebar_browser import SidebarBrowser
        for name in ("one.zarr", "two.zarr"):
            (tmp_path / name).mkdir()
            (tmp_path / name / "zarr.json").write_text("{}")
        (tmp_path / "a.tif").write_bytes(b"")
        widget = SidebarBrowser(mode="zarr", initial_path=str(Path.home()))
        opened = []
        widget.zarr_selected.connect(opened.append)
        widget.opened = opened
        yield widget
        widget.deleteLater()
        app.processEvents()

    @pytest.fixture
    def warnings(self, monkeypatch):
        from eubi_bridge.qt_gui.widgets import sidebar_browser
        seen = []
        monkeypatch.setattr(sidebar_browser.QMessageBox, "warning",
                            lambda *args: seen.append(args))
        return seen

    def test_a_dropped_store_is_opened(self, browser, tmp_path, warnings):
        TestSidebarBrowserDrop._drop(browser, tmp_path / "one.zarr")
        assert browser.opened == [str(tmp_path / "one.zarr")]
        assert Path(browser.current_path()) == tmp_path
        assert not warnings

    def test_a_plain_file_is_refused(self, browser, tmp_path, warnings):
        TestSidebarBrowserDrop._drop(browser, tmp_path / "a.tif")
        assert browser.opened == []
        assert len(warnings) == 1

    def test_several_stores_are_refused(self, browser, tmp_path, warnings):
        TestSidebarBrowserDrop._drop(browser, tmp_path / "one.zarr",
                                     tmp_path / "two.zarr")
        assert browser.opened == []
        assert len(warnings) == 1

    @staticmethod
    def _marked(browser) -> list:
        """Names of the listed stores shown as open in the viewer."""
        return [browser._list.item(row).text().split(" ", 1)[1]
                for row in range(browser._list.count())
                if browser._list.item(row).font().bold()]

    @staticmethod
    def _item(browser, name):
        for row in range(browser._list.count()):
            if browser._list.item(row).text().endswith(name):
                return browser._list.item(row)
        raise AssertionError(f"{name} not listed")

    def test_the_dropped_store_is_marked(self, browser, tmp_path):
        """After the jump to its folder, the store must be recognisable."""
        TestSidebarBrowserDrop._drop(browser, tmp_path / "one.zarr")
        assert self._marked(browser) == ["one.zarr"]
        assert browser._list.currentItem() is self._item(browser, "one.zarr")

    def test_the_marker_follows_a_click(self, browser, tmp_path):
        TestSidebarBrowserDrop._drop(browser, tmp_path / "one.zarr")
        browser._on_single_click(self._item(browser, "two.zarr"))
        browser._on_click_confirmed()        # the single-click timer's job
        assert self._marked(browser) == ["two.zarr"]
        assert browser.opened[-1] == str(tmp_path / "two.zarr")

    def test_the_marker_survives_navigation(self, browser, tmp_path):
        TestSidebarBrowserDrop._drop(browser, tmp_path / "one.zarr")
        browser.navigate_to(str(Path.home()))
        browser.navigate_to(str(tmp_path))
        assert self._marked(browser) == ["one.zarr"]

    def test_a_store_past_the_first_page_is_shown(self, app, tmp_path):
        """Folders list first, so enough of them push the store to page two."""
        from eubi_bridge.qt_gui.core.file_service import PAGE_SIZE
        from eubi_bridge.qt_gui.widgets.sidebar_browser import SidebarBrowser
        for index in range(PAGE_SIZE + 3):
            (tmp_path / f"a{index:04d}").mkdir()
        (tmp_path / "zz.zarr").mkdir()
        (tmp_path / "zz.zarr" / "zarr.json").write_text("{}")
        widget = SidebarBrowser(mode="zarr", initial_path=str(Path.home()))

        TestSidebarBrowserDrop._drop(widget, tmp_path / "zz.zarr")

        assert widget._page == 1
        assert self._marked(widget) == ["zz.zarr"]
        widget.deleteLater()
        app.processEvents()


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
