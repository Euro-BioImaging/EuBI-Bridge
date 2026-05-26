"""Canvas items: HeaveItem, WaveItem, EdgeItem, PortItem, RubberBandEdge.

All nodes use custom ``paint()`` — no QGraphicsProxyWidget — so they stay
fast, respect the active QPalette, and handle hover/selection cleanly.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainterPath,
    QPen,
)
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QStyle,
)

if TYPE_CHECKING:
    from eubi_flow.models import HeaveSpec, WaveSpec

# ---------------------------------------------------------------------------
# Status / accent colours (module-level constants — one place to update)
# ---------------------------------------------------------------------------

_C_INPUT   = QColor("#3ca0ff")   # heave_000 (input)
_C_PENDING = QColor("#666666")   # not yet written / not yet run
_C_RUNNING = QColor("#e6a020")   # wave currently executing
_C_DONE    = QColor("#4caf50")   # written / completed
_C_FAILED  = QColor("#e53935")   # error

_C_NODE_BG_HEAVE = QColor(36, 36, 48)
_C_NODE_BG_WAVE  = QColor(44, 42, 36)
_C_NODE_BORDER   = QColor(70, 70, 80)
_C_TEXT_PRIMARY  = QColor(220, 220, 225)
_C_TEXT_SECONDARY = QColor(150, 150, 158)
_C_SELECTION     = QColor("#3ca0ff")

NODE_W = 160
HEAVE_H = 80
WAVE_H  = 90

_FLAG_MOVABLE    = QGraphicsItem.GraphicsItemFlag.ItemIsMovable
_FLAG_SELECTABLE = QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
_FLAG_GEOM       = QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
_CHANGE_POS      = QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged


# ---------------------------------------------------------------------------
# PortItem
# ---------------------------------------------------------------------------

class PortItem(QGraphicsEllipseItem):
    """8×8 px connection dot on the edge of a node.

    Placed as a child of HeaveItem or WaveItem; its local origin (0, 0) is
    the visual centre of the port circle.
    """

    _R = 4   # radius

    def __init__(self, port_type: str, parent_node: QGraphicsItem) -> None:
        r = self._R
        super().__init__(QRectF(-r, -r, r * 2, r * 2), parent_node)
        self.port_type = port_type   # "in" or "out"
        self._hovered = False
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setZValue(2)
        self._refresh_brush()

    # ------------------------------------------------------------------

    def center_scene_pos(self) -> QPointF:
        """Scene position of the port centre."""
        return self.mapToScene(QPointF(0, 0))

    def hoverEnterEvent(self, event) -> None:
        self._hovered = True
        self._refresh_brush()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self._hovered = False
        self._refresh_brush()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            scene = self.scene()
            if scene and hasattr(scene, "_start_edge_drag"):
                scene._start_edge_drag(self)
                event.accept()
                return
        super().mousePressEvent(event)

    def _refresh_brush(self) -> None:
        col = _C_INPUT if self._hovered else QColor(85, 85, 95)
        self.setBrush(QBrush(col))
        self.setPen(QPen(QColor(130, 130, 140), 1))


# ---------------------------------------------------------------------------
# HeaveItem
# ---------------------------------------------------------------------------

class HeaveItem(QGraphicsRectItem):
    """Visual node representing one OME-Zarr heave artifact."""

    def __init__(self, heave_spec: "HeaveSpec") -> None:
        super().__init__(QRectF(0, 0, NODE_W, HEAVE_H))
        self.heave_spec = heave_spec
        self._status = "input" if heave_spec.heave_id == "heave_000" else "pending"
        self._edges: list[EdgeItem] = []

        self.setFlags(_FLAG_MOVABLE | _FLAG_SELECTABLE | _FLAG_GEOM)
        self.setAcceptHoverEvents(True)
        self.setZValue(1)

        # Ports — heave_000 has no input port
        cy = HEAVE_H / 2
        self.out_port = PortItem("out", self)
        self.out_port.setPos(NODE_W, cy)
        if heave_spec.heave_id != "heave_000":
            self.in_port: PortItem | None = PortItem("in", self)
            self.in_port.setPos(0, cy)
        else:
            self.in_port = None

    # ------------------------------------------------------------------

    def update_from_spec(self, spec: "HeaveSpec") -> None:
        self.heave_spec = spec
        self.update()

    def set_status(self, status: str) -> None:
        self._status = status
        self.update()

    def _status_color(self) -> QColor:
        return {
            "input":   _C_INPUT,
            "pending": _C_PENDING,
            "done":    _C_DONE,
            "failed":  _C_FAILED,
        }.get(self._status, _C_PENDING)

    def _meta_line(self) -> str:
        spec = self.heave_spec
        parts = []
        if spec.axes:
            parts.append(spec.axes)
        if spec.shape:
            parts.append("×".join(str(s) for s in spec.shape))
        if spec.dtype:
            parts.append(spec.dtype)
        return "  ".join(parts) if parts else "—"

    # ------------------------------------------------------------------

    def paint(self, painter, option, widget=None) -> None:
        r = self.rect()

        # Background
        painter.setBrush(QBrush(_C_NODE_BG_HEAVE))
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        painter.setPen(QPen(_C_SELECTION if selected else _C_NODE_BORDER,
                            2 if selected else 1))
        painter.drawRoundedRect(r, 6, 6)

        # Left accent bar
        accent = QRectF(r.left() + 1, r.top() + 8, 4, r.height() - 16)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self._status_color()))
        painter.drawRoundedRect(accent, 2, 2)

        # Title (heave_id)
        tf = QFont()
        tf.setBold(True)
        tf.setPointSize(9)
        painter.setFont(tf)
        painter.setPen(QPen(_C_TEXT_PRIMARY))
        title_rect = QRectF(14, 10, NODE_W - 20, 20)
        painter.drawText(title_rect,
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         self.heave_spec.heave_id)

        # Metadata
        sf = QFont()
        sf.setPointSize(8)
        painter.setFont(sf)
        painter.setPen(QPen(_C_TEXT_SECONDARY))
        meta_rect = QRectF(14, 34, NODE_W - 20, 36)
        painter.drawText(meta_rect,
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop |
                         Qt.TextFlag.TextWordWrap,
                         self._meta_line())

    def itemChange(self, change, value):
        if change == _CHANGE_POS:
            for edge in self._edges:
                edge.update_path()
            scene = self.scene()
            if scene and hasattr(scene, "notify_layout_dirty"):
                scene.notify_layout_dirty()
        return super().itemChange(change, value)


# ---------------------------------------------------------------------------
# WaveItem
# ---------------------------------------------------------------------------

class WaveItem(QGraphicsRectItem):
    """Visual node representing one processing wave."""

    def __init__(self, wave_spec: "WaveSpec") -> None:
        super().__init__(QRectF(0, 0, NODE_W, WAVE_H))
        self.wave_spec = wave_spec
        self._status = wave_spec.status
        self._edges: list[EdgeItem] = []

        self.setFlags(_FLAG_MOVABLE | _FLAG_SELECTABLE | _FLAG_GEOM)
        self.setAcceptHoverEvents(True)
        self.setZValue(1)

        cy = WAVE_H / 2
        self.in_port = PortItem("in", self)
        self.in_port.setPos(0, cy)
        self.out_port = PortItem("out", self)
        self.out_port.setPos(NODE_W, cy)

    # ------------------------------------------------------------------

    def update_from_spec(self, spec: "WaveSpec") -> None:
        self.wave_spec = spec
        self._status = spec.status
        self.update()

    def update_status(self, status: str) -> None:
        self._status = status
        self.update()

    def _status_color(self) -> QColor:
        return {
            "pending": _C_PENDING,
            "running": _C_RUNNING,
            "done":    _C_DONE,
            "failed":  _C_FAILED,
        }.get(self._status, _C_PENDING)

    def _params_summary(self) -> str:
        params = self.wave_spec.params
        if not params:
            return ""
        items = [f"{k}={v}" for k, v in list(params.items())[:2]]
        suffix = " …" if len(params) > 2 else ""
        return "  ".join(items) + suffix

    # ------------------------------------------------------------------

    def paint(self, painter, option, widget=None) -> None:
        r = self.rect()

        # Background with chamfered top corners via QPainterPath
        path = QPainterPath()
        ch = 10  # chamfer size
        path.moveTo(r.left() + ch, r.top())
        path.lineTo(r.right() - ch, r.top())
        path.lineTo(r.right(), r.top() + ch)
        path.lineTo(r.right(), r.bottom())
        path.lineTo(r.left(), r.bottom())
        path.lineTo(r.left(), r.top() + ch)
        path.closeSubpath()

        painter.setBrush(QBrush(_C_NODE_BG_WAVE))
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        painter.setPen(QPen(_C_SELECTION if selected else _C_NODE_BORDER,
                            2 if selected else 1))
        painter.drawPath(path)

        # Left accent bar
        accent = QRectF(r.left() + 1, r.top() + ch, 4, r.height() - ch - 8)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self._status_color()))
        painter.drawRoundedRect(accent, 2, 2)

        # Status dot (top-right)
        dot_x = r.right() - 14
        dot_y = r.top() + 8
        painter.setBrush(QBrush(self._status_color()))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(dot_x, dot_y, 8, 8))

        # Processor name (bold)
        tf = QFont()
        tf.setBold(True)
        tf.setPointSize(9)
        painter.setFont(tf)
        painter.setPen(QPen(_C_TEXT_PRIMARY))
        name_rect = QRectF(14, 10, NODE_W - 35, 20)
        painter.drawText(name_rect,
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         self.wave_spec.name)

        # Wave ID (small, secondary)
        sf = QFont()
        sf.setPointSize(7)
        painter.setFont(sf)
        painter.setPen(QPen(_C_TEXT_SECONDARY))
        id_rect = QRectF(14, 30, NODE_W - 20, 16)
        painter.drawText(id_rect,
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         self.wave_spec.wave_id)

        # Params summary (small)
        summary = self._params_summary()
        if summary:
            painter.setPen(QPen(QColor(130, 160, 130)))
            p_rect = QRectF(14, 50, NODE_W - 20, 30)
            painter.drawText(p_rect,
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop |
                             Qt.TextFlag.TextWordWrap,
                             summary)

    def itemChange(self, change, value):
        if change == _CHANGE_POS:
            for edge in self._edges:
                edge.update_path()
            scene = self.scene()
            if scene and hasattr(scene, "notify_layout_dirty"):
                scene.notify_layout_dirty()
        return super().itemChange(change, value)


# ---------------------------------------------------------------------------
# EdgeItem
# ---------------------------------------------------------------------------

class EdgeItem(QGraphicsPathItem):
    """Cubic Bezier edge between two PortItems."""

    _CTRL_OFFSET = 70   # horizontal tangent length

    def __init__(
        self,
        source_port: PortItem,
        dest_port: PortItem | None = None,
    ) -> None:
        super().__init__()
        self.source_port = source_port
        self.dest_port = dest_port
        self._dest_pos: QPointF | None = None
        self.setZValue(0)
        pen = QPen(QColor(100, 100, 110), 2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.setPen(pen)
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.update_path()

    # ------------------------------------------------------------------

    def set_dest_pos(self, pos: QPointF) -> None:
        """Move the floating end of a rubber-band edge."""
        self._dest_pos = pos
        self.update_path()

    def update_path(self) -> None:
        src = self.source_port.center_scene_pos()
        if self.dest_port is not None:
            dst = self.dest_port.center_scene_pos()
        elif self._dest_pos is not None:
            dst = self._dest_pos
        else:
            return

        off = self._CTRL_OFFSET
        path = QPainterPath(src)
        path.cubicTo(src + QPointF(off, 0),
                     dst - QPointF(off, 0),
                     dst)

        # Arrowhead at dst
        dx = dst.x() - (dst.x() - off)
        dy = 0.0
        import math
        angle = math.atan2(dst.y() - (dst.y()), off)
        arrow_len = 8
        arrow_half = 4
        tip = dst
        base_mid = tip - QPointF(arrow_len, 0)
        arrow = QPainterPath()
        arrow.moveTo(tip)
        arrow.lineTo(base_mid + QPointF(0, arrow_half))
        arrow.lineTo(base_mid - QPointF(0, arrow_half))
        arrow.closeSubpath()
        path.addPath(arrow)

        self.setPath(path)

    def paint(self, painter, option, widget=None) -> None:
        painter.setPen(self.pen())
        painter.setBrush(QBrush(self.pen().color()))
        painter.drawPath(self.path())


# ---------------------------------------------------------------------------
# RubberBandEdge — temporary edge during drag
# ---------------------------------------------------------------------------

class RubberBandEdge(EdgeItem):
    """Temporary edge drawn while the user drags from a port."""

    def __init__(self, source_port: PortItem) -> None:
        super().__init__(source_port, dest_port=None)
        pen = QPen(QColor("#3ca0ff"), 2, Qt.PenStyle.DashLine)
        self.setPen(pen)
