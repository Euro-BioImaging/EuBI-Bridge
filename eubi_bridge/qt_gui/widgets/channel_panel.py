"""
Per-channel intensity controls panel.

Each channel gets a collapsible section with:
  - visibility toggle, color swatch, label
  - min/max display-range spinboxes
  - Auto button (requests histogram computation) + Reset button
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

_DEFAULT_COLORS = [
    "#FFFFFF", "#FF0000", "#00FF00", "#0000FF",
    "#FFFF00", "#FF00FF", "#00FFFF", "#FF8000",
]


def _color_btn_style(hex_color: str) -> str:
    return (
        f"background-color: {hex_color}; "
        "border: 1px solid #666; border-radius: 3px;"
    )


class _ChannelRow(QWidget):
    """One collapsible channel section."""

    changed     = pyqtSignal()       # any control changed
    auto_clicked = pyqtSignal(int)   # Auto button → parent triggers MinMaxWorker

    def __init__(self, ch: dict, parent=None):
        super().__init__(parent)
        self._ch    = dict(ch)
        self._idx   = ch["index"]
        self._orig_min = ch.get("intensityMin")
        self._orig_max = ch.get("intensityMax")
        self._building = False
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 1, 0, 1)
        outer.setSpacing(0)

        # ── Header row (always visible) ───────────────────────────────────────
        header_widget = QWidget()
        header_widget.setStyleSheet("background: #3a3a3a; border-radius: 3px;")
        header = QHBoxLayout(header_widget)
        header.setContentsMargins(4, 3, 4, 3)
        header.setSpacing(4)

        self._expand_btn = QPushButton("▶")
        self._expand_btn.setFixedSize(24, 24)
        self._expand_btn.setStyleSheet(
            "QPushButton { border: none; background: transparent; font-size: 14px; color: #ccc; }"
            "QPushButton:hover { color: #fff; }"
        )
        self._expand_btn.setToolTip("Expand / collapse")
        self._expand_btn.clicked.connect(self._on_toggle)
        header.addWidget(self._expand_btn)

        self._vis_cb = QCheckBox()
        self._vis_cb.setChecked(self._ch.get("visible", True))
        self._vis_cb.setToolTip("Toggle channel visibility")
        self._vis_cb.stateChanged.connect(self._on_vis_changed)
        header.addWidget(self._vis_cb)

        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(16, 16)
        self._color_btn.setStyleSheet(_color_btn_style(self._ch.get("color", "#FFFFFF")))
        self._color_btn.setToolTip("Change channel color")
        self._color_btn.clicked.connect(self._on_color_clicked)
        header.addWidget(self._color_btn)

        self._header_label = QLabel(f"Ch {self._idx}: {self._ch.get('label', '')}")
        self._header_label.setStyleSheet("font-size: 10px;")
        header.addWidget(self._header_label)
        header.addStretch()

        outer.addWidget(header_widget)

        # ── Body (shown when expanded) ────────────────────────────────────────
        self._body = QWidget()
        self._body.setStyleSheet("background: #2d2d2d;")
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(8, 4, 8, 4)
        body_layout.setSpacing(4)

        # Label editor
        label_row = QHBoxLayout()
        label_row.addWidget(QLabel("Name:"))
        self._label_edit = QLineEdit(self._ch.get("label", ""))
        self._label_edit.setPlaceholderText("channel name")
        self._label_edit.textChanged.connect(self._on_label_changed)
        label_row.addWidget(self._label_edit)
        body_layout.addLayout(label_row)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Min:"))
        self._min_spin = QDoubleSpinBox()
        self._min_spin.setRange(-1e9, 1e9)
        self._min_spin.setDecimals(2)
        self._min_spin.setFixedWidth(80)
        if self._ch.get("intensityMin") is not None:
            self._min_spin.setValue(float(self._ch["intensityMin"]))
        self._min_spin.valueChanged.connect(self._on_range_changed)
        range_row.addWidget(self._min_spin)

        range_row.addWidget(QLabel("Max:"))
        self._max_spin = QDoubleSpinBox()
        self._max_spin.setRange(-1e9, 1e9)
        self._max_spin.setDecimals(2)
        self._max_spin.setFixedWidth(80)
        if self._ch.get("intensityMax") is not None:
            self._max_spin.setValue(float(self._ch["intensityMax"]))
        self._max_spin.valueChanged.connect(self._on_range_changed)
        range_row.addWidget(self._max_spin)
        body_layout.addLayout(range_row)

        btn_row = QHBoxLayout()
        auto_btn = QPushButton("Auto")
        auto_btn.setFixedHeight(22)
        auto_btn.setToolTip("Compute 1st/99th percentile from data")
        auto_btn.clicked.connect(lambda: self.auto_clicked.emit(self._idx))
        btn_row.addWidget(auto_btn)

        reset_btn = QPushButton("Reset")
        reset_btn.setFixedHeight(22)
        reset_btn.setToolTip("Revert to stored window values")
        reset_btn.clicked.connect(self._on_reset)
        btn_row.addWidget(reset_btn)
        body_layout.addLayout(btn_row)

        self._body.setVisible(False)
        outer.addWidget(self._body)

    def _on_toggle(self):
        expanded = not self._body.isVisible()
        self._body.setVisible(expanded)
        self._expand_btn.setText("▼" if expanded else "▶")

    def _on_label_changed(self, text: str):
        self._ch["label"] = text
        self._header_label.setText(f"Ch {self._idx}: {text}")
        if not self._building:
            self.changed.emit()

    def _on_vis_changed(self, state: int):
        self._ch["visible"] = state == Qt.CheckState.Checked.value
        if not self._building:
            self.changed.emit()

    def _on_color_clicked(self):
        current = QColor(self._ch.get("color", "#FFFFFF"))
        color = QColorDialog.getColor(current, self, "Channel color")
        if color.isValid():
            hex_c = color.name()
            self._ch["color"] = hex_c
            self._color_btn.setStyleSheet(_color_btn_style(hex_c))
            self.changed.emit()

    def _on_range_changed(self):
        if self._building:
            return
        self._ch["intensityMin"] = self._min_spin.value()
        self._ch["intensityMax"] = self._max_spin.value()
        self.changed.emit()

    def _on_reset(self):
        self._building = True
        if self._orig_min is not None:
            self._min_spin.setValue(float(self._orig_min))
            self._ch["intensityMin"] = float(self._orig_min)
        if self._orig_max is not None:
            self._max_spin.setValue(float(self._orig_max))
            self._ch["intensityMax"] = float(self._orig_max)
        self._building = False
        self.changed.emit()

    def set_range(self, vmin: float, vmax: float):
        """Called by MinMaxWorker result — updates spinboxes without re-emitting."""
        self._building = True
        self._min_spin.setValue(vmin)
        self._max_spin.setValue(vmax)
        self._ch["intensityMin"] = vmin
        self._ch["intensityMax"] = vmax
        self._building = False
        self.changed.emit()

    def commit_reset_baseline(self):
        """Update the reset baseline to the current spinbox values.

        Call this after a successful save so Reset reverts to the saved
        values rather than the original load-time values.
        """
        self._orig_min = self._ch.get("intensityMin")
        self._orig_max = self._ch.get("intensityMax")

    def channel_data(self) -> dict:
        return dict(self._ch)


class ChannelPanel(QWidget):
    """Scrollable list of per-channel collapsible controls.

    Signals:
        channels_changed(list[dict])  — emitted (debounced) on any change
        auto_requested(int)           — channel index whose Auto button was clicked
    """

    channels_changed = pyqtSignal(list)
    auto_requested   = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[_ChannelRow] = []

        inner = QWidget()
        self._inner_layout = QVBoxLayout(inner)
        self._inner_layout.setContentsMargins(0, 0, 0, 0)
        self._inner_layout.setSpacing(0)
        self._inner_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

    def set_channels(self, channels: list[dict]):
        """Rebuild the panel for a new list of channel dicts."""
        # Remove old rows
        for row in self._rows:
            self._inner_layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()

        # Remove stretch
        item = self._inner_layout.takeAt(self._inner_layout.count() - 1)
        del item

        for ch in channels:
            row = _ChannelRow(ch, self)
            row.changed.connect(self._on_changed)
            row.auto_clicked.connect(self.auto_requested)
            self._inner_layout.addWidget(row)
            self._rows.append(row)

        self._inner_layout.addStretch()

    def set_channel_range(self, channel_idx: int, vmin: float, vmax: float):
        """Apply auto-computed range to a channel row."""
        for row in self._rows:
            if row._idx == channel_idx:
                row.set_range(vmin, vmax)
                break

    def commit_reset_baseline(self):
        """Commit current values as the new reset baseline for all rows."""
        for row in self._rows:
            row.commit_reset_baseline()

    def channel_data(self) -> list[dict]:
        return [row.channel_data() for row in self._rows]

    def _on_changed(self):
        self.channels_changed.emit(self.channel_data())
