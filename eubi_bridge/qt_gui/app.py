"""
EuBI-Bridge Qt GUI entry point.

Launch with:  eubi-gui-qt [optional-zarr-path]
"""
from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from eubi_bridge.qt_gui.main_window import MainWindow


def main():
    import locale
    import os

    # Ensure a UTF-8 locale is active so that Python's stdout encoding and
    # Rich's terminal detection both work correctly in VNC / remote-desktop
    # sessions where the inherited locale is often plain ASCII ("C" / POSIX).
    # setdefault leaves the value alone when the user has already set it.
    os.environ.setdefault("LC_ALL", "C.UTF-8")
    os.environ.setdefault("LANG",   "C.UTF-8")
    try:
        locale.setlocale(locale.LC_ALL, "")
    except locale.Error:
        pass

    # QT_SCALE_FACTOR must be set before QApplication is constructed.
    # Read it from the persisted settings so the scale survives restarts.
    from eubi_bridge.qt_gui.settings_dialog import current_ui_scale
    os.environ.setdefault("QT_SCALE_FACTOR", str(current_ui_scale()))

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
