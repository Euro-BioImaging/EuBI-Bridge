"""
Settings dialog — font size + colour palette.

Usage:
    dlg = SettingsDialog(parent=window)
    dlg.exec()          # apply() is called automatically on Accept
"""
from __future__ import annotations

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

def _make_dark_palette(accent_hex: str = "#4287f5") -> QPalette:
    """Original dark palette (default)."""
    p = QPalette()
    accent = QColor(accent_hex)
    p.setColor(QPalette.ColorRole.Window,          QColor(30,  30,  30))
    p.setColor(QPalette.ColorRole.WindowText,      QColor(210, 210, 210))
    p.setColor(QPalette.ColorRole.Base,            QColor(45,  45,  45))
    p.setColor(QPalette.ColorRole.AlternateBase,   QColor(38,  38,  38))
    p.setColor(QPalette.ColorRole.ToolTipBase,     QColor(30,  30,  30))
    p.setColor(QPalette.ColorRole.ToolTipText,     QColor(210, 210, 210))
    p.setColor(QPalette.ColorRole.Text,            QColor(210, 210, 210))
    p.setColor(QPalette.ColorRole.Button,          QColor(60,  60,  60))
    p.setColor(QPalette.ColorRole.ButtonText,      QColor(210, 210, 210))
    p.setColor(QPalette.ColorRole.BrightText,      QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.Highlight,       accent)
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.Link,            accent)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,       QColor(120, 120, 120))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(120, 120, 120))
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
    "Dark (default)": _BASE_STYLESHEET,
    "Dark Blue":      _BASE_STYLESHEET,
    "Light":          _LIGHT_STYLESHEET,
}

# ── Settings state (module-level singleton) ────────────────────────────────────

_current_theme = "Dark (default)"
_current_font_size = 9   # pt


def current_theme() -> str:
    return _current_theme


def current_font_size() -> int:
    return _current_font_size


def apply_settings(theme: str, font_size: int) -> None:
    """Apply *theme* and *font_size* to the running QApplication."""
    global _current_theme, _current_font_size
    _current_theme = theme
    _current_font_size = font_size

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

    def _on_preview(self):
        apply_settings(self._theme_combo.currentText(), self._font_slider.value())

    def _on_accept(self):
        apply_settings(self._theme_combo.currentText(), self._font_slider.value())
        self.accept()

    def _on_reject(self):
        # Revert to the state before the dialog was opened
        apply_settings(self._snapshot_theme, self._snapshot_font)
        self.reject()
