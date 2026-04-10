"""
Settings dialog — font size, colour palette, and UI scale.

Usage:
    dlg = SettingsDialog(parent=window)
    dlg.exec()          # apply() is called automatically on Accept

UI scale note:
    QT_SCALE_FACTOR must be set before QApplication is constructed, so a scale
    change is written to disk and takes effect on the next launch.
"""
from __future__ import annotations

import json
import os
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# ── Palette definitions ────────────────────────────────────────────────────────

def _make_dark_palette(accent_hex: str = "#3ca0ff") -> QPalette:
    """Bright dark palette — the default EuBI-Bridge theme."""
    p = QPalette()
    accent = QColor(accent_hex)
    p.setColor(QPalette.ColorRole.Window,          QColor(40,  40,  42))
    p.setColor(QPalette.ColorRole.WindowText,      QColor(235, 235, 235))
    p.setColor(QPalette.ColorRole.Base,            QColor(55,  55,  58))
    p.setColor(QPalette.ColorRole.AlternateBase,   QColor(50,  50,  53))
    p.setColor(QPalette.ColorRole.ToolTipBase,     QColor(50,  50,  53))
    p.setColor(QPalette.ColorRole.ToolTipText,     QColor(235, 235, 235))
    p.setColor(QPalette.ColorRole.Text,            QColor(235, 235, 235))
    p.setColor(QPalette.ColorRole.Button,          QColor(72,  72,  76))
    p.setColor(QPalette.ColorRole.ButtonText,      QColor(235, 235, 235))
    p.setColor(QPalette.ColorRole.BrightText,      QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.Highlight,       accent)
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.Link,            accent)
    p.setColor(QPalette.ColorRole.Mid,             QColor(80,  80,  84))
    p.setColor(QPalette.ColorRole.Shadow,          QColor(20,  20,  20))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,       QColor(130, 130, 130))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(130, 130, 130))
    return p


def _make_dark_blue_palette() -> QPalette:
    """Dark palette with a blue tint."""
    p = QPalette()
    accent = QColor("#4287f5")
    p.setColor(QPalette.ColorRole.Window,          QColor(22,  30,  46))
    p.setColor(QPalette.ColorRole.WindowText,      QColor(210, 220, 235))
    p.setColor(QPalette.ColorRole.Base,            QColor(30,  42,  64))
    p.setColor(QPalette.ColorRole.AlternateBase,   QColor(25,  35,  54))
    p.setColor(QPalette.ColorRole.ToolTipBase,     QColor(22,  30,  46))
    p.setColor(QPalette.ColorRole.ToolTipText,     QColor(210, 220, 235))
    p.setColor(QPalette.ColorRole.Text,            QColor(210, 220, 235))
    p.setColor(QPalette.ColorRole.Button,          QColor(40,  55,  80))
    p.setColor(QPalette.ColorRole.ButtonText,      QColor(210, 220, 235))
    p.setColor(QPalette.ColorRole.BrightText,      QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.Highlight,       accent)
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.Link,            accent)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,       QColor(100, 110, 130))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(100, 110, 130))
    return p


def _make_light_palette() -> QPalette:
    """Light palette."""
    p = QPalette()
    accent = QColor("#1a6ed8")
    p.setColor(QPalette.ColorRole.Window,          QColor(240, 240, 240))
    p.setColor(QPalette.ColorRole.WindowText,      QColor(20,  20,  20))
    p.setColor(QPalette.ColorRole.Base,            QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.AlternateBase,   QColor(230, 230, 230))
    p.setColor(QPalette.ColorRole.ToolTipBase,     QColor(255, 255, 220))
    p.setColor(QPalette.ColorRole.ToolTipText,     QColor(20,  20,  20))
    p.setColor(QPalette.ColorRole.Text,            QColor(20,  20,  20))
    p.setColor(QPalette.ColorRole.Button,          QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.ButtonText,      QColor(20,  20,  20))
    p.setColor(QPalette.ColorRole.BrightText,      QColor(0,   0,   0))
    p.setColor(QPalette.ColorRole.Highlight,       accent)
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.Link,            accent)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,       QColor(150, 150, 150))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(150, 150, 150))
    return p


PALETTES: dict[str, QPalette] = {
    "Dark (default)":  _make_dark_palette(),
    "Dark Blue":       _make_dark_blue_palette(),
    "Light":           _make_light_palette(),
}

_DARK_STYLESHEET = (
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
    "QCheckBox::indicator:unchecked { background: #505054; border: 1px solid #686870; border-radius: 2px; }"
    "QLabel { color: #ebebeb; }"
    "QScrollBar:vertical { background: #484848; width: 10px; }"
    "QScrollBar::handle:vertical { background: #686870; border-radius: 5px; min-height: 20px; }"
    "QScrollBar::handle:vertical:hover { background: #3ca0ff; }"
)

_BASE_STYLESHEET = (
    "QToolTip { color: #ddd; background: #333; border: 1px solid #555; }"
    "QGroupBox { border: 1px solid #555; border-radius: 3px; margin-top: 6px; }"
    "QGroupBox::title { subcontrol-origin: margin; left: 6px; }"
    "QTabWidget::pane { border: 1px solid #555; }"
    "QScrollArea { border: none; }"
)

_LIGHT_STYLESHEET = (
    "QToolTip { color: #222; background: #fffde7; border: 1px solid #bbb; }"
    "QGroupBox { border: 1px solid #bbb; border-radius: 3px; margin-top: 6px; }"
    "QGroupBox::title { subcontrol-origin: margin; left: 6px; }"
    "QTabWidget::pane { border: 1px solid #bbb; }"
    "QScrollArea { border: none; }"
)

STYLESHEETS: dict[str, str] = {
    "Dark (default)": _DARK_STYLESHEET,
    "Dark Blue":      _BASE_STYLESHEET,
    "Light":          _LIGHT_STYLESHEET,
}

# ── Settings state (module-level singleton) ────────────────────────────────────

_current_theme = "Dark (default)"
_current_font_size = 9   # pt
_current_ui_scale = 1.0  # stored; applied via QT_SCALE_FACTOR at next launch

_UI_SCALE_OPTIONS = ["100%", "125%", "150%", "175%", "200%"]
_UI_SCALE_VALUES  = [1.0,    1.25,   1.5,    1.75,   2.0]

# ── Persistence ───────────────────────────────────────────────────────────────

def _settings_path() -> str:
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    folder = os.path.join(base, "EuBI-Bridge")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "settings.json")


def _load_settings() -> None:
    global _current_theme, _current_font_size, _current_ui_scale
    try:
        with open(_settings_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        _current_theme     = data.get("theme",     _current_theme)
        _current_font_size = int(data.get("font_size", _current_font_size))
        _current_ui_scale  = float(data.get("ui_scale",  _current_ui_scale))
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass


def _save_settings() -> None:
    try:
        with open(_settings_path(), "w", encoding="utf-8") as fh:
            json.dump({
                "theme":     _current_theme,
                "font_size": _current_font_size,
                "ui_scale":  _current_ui_scale,
            }, fh, indent=2)
    except OSError:
        pass


_load_settings()  # populate globals from disk at import time


def current_theme() -> str:
    return _current_theme


def current_font_size() -> int:
    return _current_font_size


def current_ui_scale() -> float:
    return _current_ui_scale


def apply_settings(theme: str, font_size: int, ui_scale: float | None = None) -> None:
    """Apply *theme* and *font_size* to the running QApplication.
    *ui_scale* is persisted but takes effect only on the next launch."""
    global _current_theme, _current_font_size, _current_ui_scale
    _current_theme = theme
    _current_font_size = font_size
    if ui_scale is not None:
        _current_ui_scale = ui_scale
    _save_settings()

    app = QApplication.instance()
    if not isinstance(app, QApplication):
        return

    palette = PALETTES.get(theme, PALETTES["Dark (default)"])
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEETS.get(theme, _BASE_STYLESHEET))

    font = app.font()
    font.setPointSize(font_size)
    app.setFont(font)

    # Flush the queued ApplicationFontChange / ApplicationPaletteChange events
    # so child widgets update their metrics immediately (not on the next click).
    app.processEvents()

    # Re-polish every widget so the Fusion style picks up the new palette/font.
    for w in app.allWidgets():
        app.style().unpolish(w)
        app.style().polish(w)
        w.update()


# ── Dialog ─────────────────────────────────────────────────────────────────────

class SettingsDialog(QDialog):
    """Modal settings dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(320)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)

        outer = QVBoxLayout(self)
        outer.setSpacing(12)

        # ── Appearance group ──────────────────────────────────────────────────
        ap_group = QGroupBox("Appearance")
        ap_layout = QFormLayout(ap_group)
        ap_layout.setSpacing(8)

        self._theme_combo = QComboBox()
        for name in PALETTES:
            self._theme_combo.addItem(name)
        idx = self._theme_combo.findText(_current_theme)
        if idx >= 0:
            self._theme_combo.setCurrentIndex(idx)
        ap_layout.addRow("Colour theme:", self._theme_combo)

        # Font size slider + live label
        font_row_widget = self._make_font_row()
        ap_layout.addRow("Font size:", font_row_widget)

        # UI scale dropdown
        self._scale_combo = QComboBox()
        for label in _UI_SCALE_OPTIONS:
            self._scale_combo.addItem(label)
        # Select the option closest to the current scale
        closest = min(range(len(_UI_SCALE_VALUES)),
                      key=lambda i: abs(_UI_SCALE_VALUES[i] - _current_ui_scale))
        self._scale_combo.setCurrentIndex(closest)
        ap_layout.addRow("UI scale:", self._scale_combo)

        scale_note = QLabel("\u26a0\ufe0f Scale change takes effect after restart.")
        scale_note.setStyleSheet("font-size: 9px; color: #aaa;")
        ap_layout.addRow("", scale_note)

        outer.addWidget(ap_group)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        preview_btn = QPushButton("Preview")
        preview_btn.setToolTip("Apply settings immediately without closing")
        preview_btn.clicked.connect(self._on_preview)
        btn_row.addWidget(preview_btn)

        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._on_accept)
        btn_row.addWidget(ok_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._on_reject)
        btn_row.addWidget(cancel_btn)

        outer.addLayout(btn_row)

        # Snapshot of state at dialog open — used to revert on Cancel
        self._snapshot_theme = _current_theme
        self._snapshot_font  = _current_font_size
        self._snapshot_scale = _current_ui_scale

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_font_row(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self._font_slider = QSlider(Qt.Orientation.Horizontal)
        self._font_slider.setRange(7, 16)
        self._font_slider.setValue(_current_font_size)
        self._font_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._font_slider.setTickInterval(1)
        self._font_slider.setFixedWidth(160)

        self._font_label = QLabel(f"{_current_font_size} pt")
        self._font_label.setMinimumWidth(36)
        self._font_slider.valueChanged.connect(
            lambda v: self._font_label.setText(f"{v} pt")
        )
        lay.addWidget(self._font_slider)
        lay.addWidget(self._font_label)
        return w

    def _selected_scale(self) -> float:
        return _UI_SCALE_VALUES[self._scale_combo.currentIndex()]

    def _on_preview(self):
        apply_settings(self._theme_combo.currentText(), self._font_slider.value(),
                       self._selected_scale())

    def _on_accept(self):
        apply_settings(self._theme_combo.currentText(), self._font_slider.value(),
                       self._selected_scale())
        self.accept()

    def _on_reject(self):
        # Revert to the state before the dialog was opened
        apply_settings(self._snapshot_theme, self._snapshot_font, self._snapshot_scale)
        self.reject()
