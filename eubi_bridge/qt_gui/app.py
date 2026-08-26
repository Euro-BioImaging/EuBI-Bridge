"""
EuBI-Bridge Qt GUI entry point.

Launch with:  eubi-gui-qt [optional-zarr-path]
"""
from __future__ import annotations

import sys


def _import_qt():
    """Import Qt, turning a missing-system-library failure into advice.

    PyQt6 installs cleanly on a headless Linux box but cannot load its system
    libraries there, so the bare ImportError names ``libEGL.so.1`` and nothing
    else.  The CLI has no such dependency, which is worth saying.
    """
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError as exc:
        raise SystemExit(
            f"EuBI-Bridge could not start its graphical interface: {exc}\n"
            "\n"
            "Qt needs system libraries that are not installed. On Debian or "
            "Ubuntu:\n"
            "    sudo apt install libegl1 libgl1 libxkbcommon0 libdbus-1-3\n"
            "\n"
            "The 'eubi' command-line interface needs none of these and works "
            "as normal."
        ) from exc
    return QApplication


def main():
    import locale
    import os

    QApplication = _import_qt()
    from eubi_bridge.qt_gui.main_window import MainWindow

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
