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
    dark   = QColor(40,  40,  42)
    mid    = QColor(55,  55,  58)
    light  = QColor(72,  72,  76)
    text   = QColor(235, 235, 235)
    accent = QColor(60,  160, 255)
    disabled = QColor(130, 130, 130)

    p.setColor(QPalette.ColorRole.Window,          dark)
    p.setColor(QPalette.ColorRole.WindowText,      text)
    p.setColor(QPalette.ColorRole.Base,            mid)
    p.setColor(QPalette.ColorRole.AlternateBase,   QColor(50, 50, 53))
    p.setColor(QPalette.ColorRole.ToolTipBase,     QColor(50, 50, 53))
    p.setColor(QPalette.ColorRole.ToolTipText,     text)
    p.setColor(QPalette.ColorRole.Text,            text)
    p.setColor(QPalette.ColorRole.Button,          light)
    p.setColor(QPalette.ColorRole.ButtonText,      text)
    p.setColor(QPalette.ColorRole.BrightText,      QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.Highlight,       accent)
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.Link,            accent)
    p.setColor(QPalette.ColorRole.Mid,             QColor(80,  80,  84))
    p.setColor(QPalette.ColorRole.Shadow,          QColor(20,  20,  20))

    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,       disabled)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled)

    return p


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(_dark_palette(app))
    app.setStyleSheet(
        "QToolTip { color: #ebebeb; background: #383838; border: 1px solid #60a0ff; }"
        "QGroupBox { border: 1px solid #666; border-radius: 4px; margin-top: 6px; font-weight: bold; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #b0d0ff; }"
        "QTabWidget::pane { border: 1px solid #666; }"
        "QTabBar::tab { background: #484848; color: #ddd; padding: 4px 10px; border: 1px solid #666; border-bottom: none; }"
        "QTabBar::tab:selected { background: #383a3e; color: #fff; border-bottom: 2px solid #3ca0ff; }"
        "QTabBar::tab:hover:!selected { background: #525256; }"
        "QPushButton { background: #505054; color: #ebebeb; border: 1px solid #686870; border-radius: 3px; padding: 3px 8px; }"
        "QPushButton:hover { background: #5a5a60; border-color: #3ca0ff; }"
        "QPushButton:pressed { background: #404044; }"
        "QPushButton:disabled { color: #888; border-color: #555; }"
        "QComboBox { background: #505054; color: #ebebeb; border: 1px solid #686870; border-radius: 3px; padding: 2px 6px; }"
        "QComboBox:hover { border-color: #3ca0ff; }"
        "QComboBox QAbstractItemView { background: #484848; color: #ebebeb; selection-background-color: #3ca0ff; }"
        "QLineEdit, QSpinBox, QDoubleSpinBox { background: #505054; color: #ebebeb; border: 1px solid #686870; border-radius: 3px; padding: 2px 4px; }"
        "QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #3ca0ff; }"
        "QScrollArea { border: none; }"
        "QSlider::groove:horizontal { background: #555; height: 4px; border-radius: 2px; }"
        "QSlider::handle:horizontal { background: #3ca0ff; width: 12px; height: 12px; margin: -4px 0; border-radius: 6px; }"
        "QSlider::sub-page:horizontal { background: #3ca0ff; border-radius: 2px; }"
        "QCheckBox { color: #ebebeb; }"
        "QCheckBox::indicator:checked { background: #3ca0ff; border: 1px solid #3ca0ff; border-radius: 2px; }"
        "QCheckBox::indicator:unchecked { background: #505054; border: 1px solid #686870; border-radius: 2px; }"
        "QLabel { color: #ebebeb; }"
        "QScrollBar:vertical { background: #484848; width: 10px; }"
        "QScrollBar::handle:vertical { background: #686870; border-radius: 5px; min-height: 20px; }"
        "QScrollBar::handle:vertical:hover { background: #3ca0ff; }"
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
