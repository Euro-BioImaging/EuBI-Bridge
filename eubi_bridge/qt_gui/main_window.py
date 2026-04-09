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

# Logo bundled inside the qt_gui package directory
_LOGO_PATH = os.path.join(os.path.dirname(__file__), "eurobioimaging-logo.webp")


class MainWindow(QMainWindow):
    """Top-level window with Convert and Inspect tabs."""

    def __init__(self, initial_path: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("EuBI-Bridge")
        self.resize(1280, 800)

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

        if initial_path:
            self._inspect_page._browser.navigate_to(initial_path)
            self._tabs.setCurrentIndex(1)

    def _on_settings(self):
        dlg = SettingsDialog(parent=self)
        dlg.exec()
