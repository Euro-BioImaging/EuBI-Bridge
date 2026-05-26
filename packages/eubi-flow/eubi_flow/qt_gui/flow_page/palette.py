"""WavePalette — drag source listing all registered wave processors.

The sidebar is structured as nested collapsible sections:

    ▼ Processors
      ▼ Filters
          gaussian_filter  (draggable)
          ...
      ▼ Thresholding
          ...
      ▼ Numerical
          ...
"""
from __future__ import annotations

from PyQt6.QtCore import QMimeData, Qt
from PyQt6.QtGui import QColor, QDrag, QFont, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Collapsible section helper
# ---------------------------------------------------------------------------

class _CollapsibleSection(QWidget):
    """Arrow-toggle section: click the title to show/hide content."""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._toggle = QToolButton()
        self._toggle.setCheckable(True)
        self._toggle.setChecked(True)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.ArrowType.DownArrow)
        self._toggle.setText(f"  {title}")
        self._toggle.setStyleSheet(
            "QToolButton { border: none; font-weight: bold; padding: 2px 0px; }"
        )
        self._toggle.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._toggle.clicked.connect(self._on_toggle)
        outer.addWidget(self._toggle)

        self._content = QWidget()
        self._cl = QVBoxLayout(self._content)
        self._cl.setContentsMargins(4, 0, 0, 2)
        self._cl.setSpacing(2)
        outer.addWidget(self._content)

    def _on_toggle(self, checked: bool) -> None:
        self._content.setVisible(checked)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )

    def add_widget(self, w: QWidget) -> None:
        self._cl.addWidget(w)


# ---------------------------------------------------------------------------
# Draggable list
# ---------------------------------------------------------------------------

class _DraggableList(QListWidget):
    """QListWidget that initiates a drag carrying the processor name."""

    def startDrag(self, supported_actions) -> None:
        item = self.currentItem()
        if item is None:
            return
        if not (item.flags() & Qt.ItemFlag.ItemIsSelectable):
            return
        wave_name = item.text().strip()

        pix = QPixmap(90, 22)
        pix.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor("#3ca0ff"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(0, 0, 90, 22, 6, 6)
        f = QFont()
        f.setPointSize(8)
        f.setBold(True)
        painter.setFont(f)
        painter.setPen(QColor("white"))
        painter.drawText(0, 0, 90, 22, Qt.AlignmentFlag.AlignCenter, wave_name)
        painter.end()

        mime = QMimeData()
        mime.setData("application/x-wave-name", wave_name.encode("utf-8"))

        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.setPixmap(pix)
        drag.setHotSpot(pix.rect().center())
        drag.exec(Qt.DropAction.CopyAction)


# ---------------------------------------------------------------------------
# WavePalette
# ---------------------------------------------------------------------------

class WavePalette(QWidget):
    """Left sidebar: nested collapsible sections → draggable processor items."""

    _GROUPS: dict[str, list[str]] = {
        "Filters": [
            "gaussian_filter",
            "median_filter",
            "uniform_filter",
            "maximum_filter",
            "minimum_filter",
            "percentile_filter",
            "rank_filter",
        ],
        "Gradient": [
            "laplace",
            "gaussian_laplace",
            "gaussian_gradient_magnitude",
            "sobel",
            "prewitt",
        ],
        "Fourier": [
            "fourier_gaussian",
            "fourier_uniform",
            "fourier_ellipsoid",
        ],
        "Morphology": [
            "binary_erosion",
            "binary_dilation",
            "binary_opening",
            "binary_closing",
            "binary_fill_holes",
            "grey_erosion",
            "grey_dilation",
            "grey_opening",
            "grey_closing",
            "distance_transform_edt",
        ],
        "Thresholding": [
            "threshold_fixed",
            "threshold_otsu",
            "threshold_percentile",
        ],
        "Reductions": [
            "max_projection",
            "mean_projection",
            "sum_projection",
        ],
        "Reshape": [
            "new_axis",
        ],
        "Numerical": [
            "add_scalar",
            "subtract_scalar",
            "multiply_scalar",
            "divide_scalar",
            "abs_val",
            "sqrt",
            "square",
            "power",
            "log1p",
            "exp",
            "clip",
            "normalize_minmax",
            "invert",
            "cast_dtype",
        ],
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(200)

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(0)

        # ── Outer "Processors" collapsible ────────────────────────────
        outer = _CollapsibleSection("Processors")
        outer._toggle.setStyleSheet(
            "QToolButton { border: none; font-weight: bold; "
            "font-size: 10px; color: #aaaaaa; padding: 3px 0px; }"
        )
        root.addWidget(outer, stretch=1)

        # Scroll area fills the outer section
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc_widget = QWidget()
        sc_layout = QVBoxLayout(sc_widget)
        sc_layout.setContentsMargins(0, 2, 0, 2)
        sc_layout.setSpacing(4)
        scroll.setWidget(sc_widget)
        outer.add_widget(scroll)

        # ── One collapsible per category ──────────────────────────────
        for group_name, proc_names in self._GROUPS.items():
            section = _CollapsibleSection(group_name)
            lst = _DraggableList()
            lst.setDragEnabled(True)
            lst.setDragDropMode(QListWidget.DragDropMode.DragOnly)
            lst.setDefaultDropAction(Qt.DropAction.CopyAction)
            lst.setFrameShape(QListWidget.Shape.NoFrame)
            lst.setSpacing(1)
            lst.setSizeAdjustPolicy(
                QListWidget.SizeAdjustPolicy.AdjustToContents
            )
            for proc_name in proc_names:
                item = QListWidgetItem(proc_name)
                item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsDragEnabled
                )
                lst.addItem(item)
            section.add_widget(lst)
            sc_layout.addWidget(section)

        sc_layout.addStretch()
