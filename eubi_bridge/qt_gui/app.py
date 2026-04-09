"""
EuBI-Bridge Qt GUI entry point.

Launch with:  eubi-gui-qt [optional-zarr-path]
"""
from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from eubi_bridge.qt_gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    from eubi_bridge.qt_gui.settings_dialog import (
        PALETTES, STYLESHEETS, current_font_size, _current_theme,
    )
    app.setPalette(PALETTES[_current_theme])
    app.setStyleSheet(STYLESHEETS[_current_theme])

    font = app.font()
    font.setPointSize(current_font_size())
    app.setFont(font)

    initial_path = sys.argv[1] if len(sys.argv) > 1 else ""
    window = MainWindow(initial_path=initial_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
