"""
Main window — QMainWindow with Convert and Inspect tabs.
"""
from __future__ import annotations

import os

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from eubi_bridge.qt_gui.pages.convert_page import ConvertPage
from eubi_bridge.qt_gui.pages.inspect_page import InspectPage
from eubi_bridge.qt_gui.settings_dialog import SettingsDialog, apply_settings, current_font_size, current_theme

try:
    from eubi_flow.qt_gui.pages.flow_page import FlowPage
    _FLOW_AVAILABLE = True
except ModuleNotFoundError:
    _FLOW_AVAILABLE = False
    _FLOW_MISSING_MSG = (
        "The 'eubi-flow' package is not installed.\n"
        "To enable the Flow tab:  pip install eubi-flow\n"
        "Or:  pip install \"eubi-bridge[flow]\""
    )

try:
    from eubi_annotate.qt_gui.pages.process_page import ProcessPage
    _ANNOTATE_AVAILABLE = True
except ModuleNotFoundError:
    _ANNOTATE_AVAILABLE = False
    _ANNOTATE_MISSING_MSG = (
        "The 'eubi-annotate' package is not installed.\n"
        "To enable the Process tab:  pip install eubi-annotate\n"
        "Or:  pip install \"eubi-bridge[annotate]\""
    )

# Logo bundled inside the qt_gui package directory
_LOGO_PATH = os.path.join(os.path.dirname(__file__), "eurobioimaging-logo.webp")


class MainWindow(QMainWindow):
    """Top-level window with Convert and Inspect tabs."""

    def __init__(self, initial_path: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("EuBI-Bridge")
        self.resize(1280, 800)
        self._center_on_screen()

        # ── Central widget: header + tabs ─────────────────────────────────────
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        # Header bar with logo + title — use palette colour so it blends with the theme
        header = QWidget()
        header.setFixedHeight(40)
        header.setStyleSheet("border-bottom: 1px solid #555;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 4, 8, 4)
        header_layout.setSpacing(8)

        logo_label = QLabel()
        logo_path = os.path.abspath(_LOGO_PATH)
        if os.path.exists(logo_path):
            pix = QPixmap(logo_path)
            if not pix.isNull():
                logo_label.setPixmap(
                    pix.scaledToHeight(36, Qt.TransformationMode.SmoothTransformation)
                )
        header_layout.addWidget(logo_label)

        title_label = QLabel("")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        settings_btn = QPushButton("\u2699 Settings")
        settings_btn.setFixedHeight(26)
        settings_btn.setToolTip("Open application settings (theme, font size)")
        settings_btn.clicked.connect(self._on_settings)
        header_layout.addWidget(settings_btn)

        central_layout.addWidget(header)

        self._tabs = QTabWidget()
        central_layout.addWidget(self._tabs)

        self.setCentralWidget(central)

        self._convert_page = ConvertPage()
        self._inspect_page = InspectPage()

        self._tabs.addTab(self._convert_page, "Convert")
        self._tabs.addTab(self._inspect_page, "Inspect")

        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        self._inspect_page.status_changed.connect(status_bar.showMessage)

        if _ANNOTATE_AVAILABLE:
            self._process_page = ProcessPage()
            self._tabs.addTab(self._process_page, "Process")
            self._process_page.status_changed.connect(status_bar.showMessage)
        else:
            status_bar.showMessage(_ANNOTATE_MISSING_MSG, 10000)

        if _FLOW_AVAILABLE:
            self._flow_page = FlowPage()
            self._tabs.addTab(self._flow_page, "Flow")
            self._flow_page.status_changed.connect(status_bar.showMessage)
        else:
            status_bar.showMessage(_FLOW_MISSING_MSG, 10000)

        if initial_path:
            self._inspect_page._browser.navigate_to(initial_path)
            self._tabs.setCurrentIndex(1)

    def _center_on_screen(self) -> None:
        from PyQt6.QtWidgets import QApplication
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        frame = self.frameGeometry()
        frame.moveCenter(available.center())
        # Clamp to available area so the window is never off-screen
        x = max(available.left(), min(frame.left(), available.right()  - self.width()))
        y = max(available.top(),  min(frame.top(),  available.bottom() - self.height()))
        self.move(x, y)

    def _on_settings(self):
        dlg = SettingsDialog(parent=self)
        dlg.exec()
