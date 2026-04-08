"""
EuBI-Bridge Qt GUI entry point.

Launch with:  eubi-gui-qt [optional-zarr-path]
"""
from __future__ import annotations

import sys

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

from eubi_bridge.qt_gui.main_window import MainWindow


def _dark_palette(app: QApplication) -> QPalette:
    """Return a dark QPalette matching the prototype viewer."""
    p = QPalette()
    dark   = QColor(30,  30,  30)
    mid    = QColor(45,  45,  45)
    light  = QColor(60,  60,  60)
    text   = QColor(210, 210, 210)
    accent = QColor(66,  135, 245)
    disabled = QColor(120, 120, 120)

    p.setColor(QPalette.ColorRole.Window,          dark)
    p.setColor(QPalette.ColorRole.WindowText,      text)
    p.setColor(QPalette.ColorRole.Base,            mid)
    p.setColor(QPalette.ColorRole.AlternateBase,   QColor(38, 38, 38))
    p.setColor(QPalette.ColorRole.ToolTipBase,     dark)
    p.setColor(QPalette.ColorRole.ToolTipText,     text)
    p.setColor(QPalette.ColorRole.Text,            text)
    p.setColor(QPalette.ColorRole.Button,          light)
    p.setColor(QPalette.ColorRole.ButtonText,      text)
    p.setColor(QPalette.ColorRole.BrightText,      QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.Highlight,       accent)
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.Link,            accent)

    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,       disabled)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled)

    return p


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(_dark_palette(app))
    app.setStyleSheet(
        "QToolTip { color: #ddd; background: #333; border: 1px solid #555; }"
        "QGroupBox { border: 1px solid #555; border-radius: 3px; margin-top: 6px; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 6px; }"
        "QTabWidget::pane { border: 1px solid #555; }"
        "QScrollArea { border: none; }"
    )

    # Apply initial font size from settings module
    from eubi_bridge.qt_gui.settings_dialog import current_font_size
    font = app.font()
    font.setPointSize(current_font_size())
    app.setFont(font)

    initial_path = sys.argv[1] if len(sys.argv) > 1 else ""
    window = MainWindow(initial_path=initial_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
