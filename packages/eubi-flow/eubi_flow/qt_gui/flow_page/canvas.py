"""FlowScene and FlowCanvas — the interactive DAG canvas.

FlowScene manages the collection of HeaveItems, WaveItems, and EdgeItems
derived from a loaded FlowSpec.  FlowCanvas wraps it in a QGraphicsView
with zoom/pan and drag-and-drop acceptance.
"""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import QEvent, QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen, QTransform
from PyQt6.QtWidgets import QGraphicsScene, QGraphicsView

from eubi_flow.qt_gui.flow_page.items import (
    EdgeItem,
    HeaveItem,
    PortItem,
    RubberBandEdge,
    WaveItem,
)
from eubi_flow.qt_gui.flow_page.layout_store import LayoutStore

if TYPE_CHECKING:
    from eubi_flow.models import FlowSpec

# Layout constants
_COL_SPACING = 240
_ROW_SPACING = 120
_ORIGIN_X    = 60
_ORIGIN_Y    = 60


class FlowScene(QGraphicsScene):
    """QGraphicsScene that owns and manages all flow DAG items."""

    item_selected    = pyqtSignal(object)      # HeaveItem | WaveItem | None
    layout_dirty_changed = pyqtSignal()        # any node moved
    node_dropped     = pyqtSignal(str, QPointF)  # wave_name, scene_pos

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._flow: Optional["FlowSpec"] = None
        self._flow_name: Optional[str] = None
        self._layout_store = LayoutStore()

        self._heave_items: dict[str, HeaveItem] = {}
        self._wave_items:  dict[str, WaveItem]  = {}
        self._edge_items:  list[EdgeItem]       = []
        self._rubber_band: Optional[RubberBandEdge] = None

        self.selectionChanged.connect(self._on_selection_changed)
        self.setBackgroundBrush(QBrush(QColor(28, 28, 32)))

    # ------------------------------------------------------------------
    # Load / rebuild
    # ------------------------------------------------------------------

    def load_flow(self, name: str, flow: "FlowSpec") -> None:
        """Build all canvas items from *flow* and apply saved positions."""
        self._flow_name = name
        self._flow = flow
        self.rebuild_from_spec()

        positions = self._layout_store.load(name)
        if positions:
            self.apply_layout(positions)
        else:
            self.auto_layout()

    def rebuild_from_spec(self) -> None:
        """Clear the scene and recreate all items from _flow."""
        if self._flow is None:
            return
        self.clear()
        self._heave_items.clear()
        self._wave_items.clear()
        self._edge_items.clear()
        self._rubber_band = None

        for heave_id, spec in self._flow.heaves.items():
            self._add_heave(spec)

        for wave in self._flow.waves:
            self._add_wave(wave)

    def _add_heave(self, spec) -> HeaveItem:
        item = HeaveItem(spec)
        self._heave_items[spec.heave_id] = item
        self.addItem(item)
        return item

    def _add_wave(self, wave) -> WaveItem:
        item = WaveItem(wave)
        self._wave_items[wave.wave_id] = item
        self.addItem(item)

        # Create edges: input heave(s) → wave
        for hid in wave.input_heave_ids:
            if hid in self._heave_items:
                src_port = self._heave_items[hid].out_port
                dst_port = item.in_port
                self._add_edge(src_port, dst_port, self._heave_items[hid], item)

        # Create edge: wave → output heave
        if wave.output_heave_id in self._heave_items:
            src_port = item.out_port
            dst_port = self._heave_items[wave.output_heave_id].in_port
            if dst_port is not None:
                self._add_edge(src_port, dst_port, item,
                               self._heave_items[wave.output_heave_id])

        return item

    def _add_edge(
        self,
        src_port: PortItem,
        dst_port: PortItem,
        src_node,
        dst_node,
    ) -> EdgeItem:
        edge = EdgeItem(src_port, dst_port)
        self._edge_items.append(edge)
        self.addItem(edge)
        src_node._edges.append(edge)
        dst_node._edges.append(edge)
        edge.update_path()
        return edge

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def auto_layout(self) -> None:
        """BFS from heave_000; place nodes left-to-right, fan-out stacked."""
        if self._flow is None:
            return

        col_row_count: dict[int, int] = {}
        positions: dict[str, tuple[float, float]] = {}
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque()

        queue.append(("heave_000", 0))
        visited.add("heave_000")

        while queue:
            node_id, col = queue.popleft()
            row = col_row_count.get(col, 0)
            col_row_count[col] = row + 1
            positions[node_id] = (
                _ORIGIN_X + col * _COL_SPACING,
                _ORIGIN_Y + row * _ROW_SPACING,
            )

            if node_id in self._heave_items:
                for wave in self._flow.waves:
                    if node_id in wave.input_heave_ids and wave.wave_id not in visited:
                        visited.add(wave.wave_id)
                        queue.append((wave.wave_id, col + 1))
            elif node_id in self._wave_items:
                wave = next(
                    (w for w in self._flow.waves if w.wave_id == node_id), None
                )
                if wave and wave.output_heave_id not in visited:
                    visited.add(wave.output_heave_id)
                    queue.append((wave.output_heave_id, col + 1))

        self.apply_layout(positions)

    def apply_layout(self, positions: dict[str, tuple[float, float]]) -> None:
        for node_id, (x, y) in positions.items():
            if node_id in self._heave_items:
                self._heave_items[node_id].setPos(x, y)
            elif node_id in self._wave_items:
                self._wave_items[node_id].setPos(x, y)
        for edge in self._edge_items:
            edge.update_path()

    def save_layout(self) -> None:
        if self._flow_name is None:
            return
        positions: dict[str, tuple[float, float]] = {}
        for nid, item in self._heave_items.items():
            p = item.pos()
            positions[nid] = (p.x(), p.y())
        for nid, item in self._wave_items.items():
            p = item.pos()
            positions[nid] = (p.x(), p.y())
        self._layout_store.save(self._flow_name, positions)

    def current_positions(self) -> dict[str, tuple[float, float]]:
        pos = {}
        for nid, item in self._heave_items.items():
            pos[nid] = (item.pos().x(), item.pos().y())
        for nid, item in self._wave_items.items():
            pos[nid] = (item.pos().x(), item.pos().y())
        return pos

    # ------------------------------------------------------------------
    # Status updates (called by FlowRunWorker)
    # ------------------------------------------------------------------

    def update_wave_status(self, wave_id: str, status: str) -> None:
        if wave_id in self._wave_items:
            self._wave_items[wave_id].update_status(status)

    def reset_all_statuses(self) -> None:
        for item in self._wave_items.values():
            item.update_status("pending")
        for nid, item in self._heave_items.items():
            item.set_status("input" if nid == "heave_000" else "pending")

    def update_wave_item(self, wave_id: str) -> None:
        """Refresh a WaveItem from the current FlowSpec (after params change)."""
        if self._flow is None or wave_id not in self._wave_items:
            return
        wave = next((w for w in self._flow.waves if w.wave_id == wave_id), None)
        if wave:
            self._wave_items[wave_id].update_from_spec(wave)

    # ------------------------------------------------------------------
    # Edge drag (rubber-band connections)
    # ------------------------------------------------------------------

    def _start_edge_drag(self, port: PortItem) -> None:
        if self._rubber_band:
            self.removeItem(self._rubber_band)
        self._rubber_band = RubberBandEdge(port)
        self.addItem(self._rubber_band)

    def _finish_edge_drag(self, target: PortItem | None) -> None:
        if self._rubber_band:
            self.removeItem(self._rubber_band)
            self._rubber_band = None
        # TODO: wire target port to a new connection (step 12)

    def mouseMoveEvent(self, event) -> None:
        if self._rubber_band:
            self._rubber_band.set_dest_pos(event.scenePos())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._rubber_band:
            transform = self.views()[0].transform() if self.views() else QTransform()
            item = self.itemAt(event.scenePos(), transform)
            target = item if isinstance(item, PortItem) else None
            self._finish_edge_drag(target)
        super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------
    # Drag-and-drop from WavePalette
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasFormat("application/x-wave-name"):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasFormat("application/x-wave-name"):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        if event.mimeData().hasFormat("application/x-wave-name"):
            wave_name = event.mimeData().data("application/x-wave-name")\
                            .data().decode("utf-8")
            self.node_dropped.emit(wave_name, event.scenePos())
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _on_selection_changed(self) -> None:
        items = self.selectedItems()
        if not items:
            self.item_selected.emit(None)
            return
        item = items[0]
        if isinstance(item, (HeaveItem, WaveItem)):
            self.item_selected.emit(item)
        else:
            self.item_selected.emit(None)

    # ------------------------------------------------------------------
    # Position-dirty signal propagation
    # ------------------------------------------------------------------

    def notify_layout_dirty(self) -> None:
        self.layout_dirty_changed.emit()


# ---------------------------------------------------------------------------
# FlowCanvas
# ---------------------------------------------------------------------------

class FlowCanvas(QGraphicsView):
    """QGraphicsView wrapper: zoom, pan, drop acceptance."""

    def __init__(self, scene: FlowScene, parent=None) -> None:
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.SmartViewportUpdate
        )
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        # Install on the viewport: QGraphicsView routes all viewport input
        # events through its own internal filter before calling the view's
        # virtual methods, so dragEnterEvent / dropEvent on the view are never
        # reached for external drags.  The event filter intercepts them first.
        self.viewport().installEventFilter(self)
        self.setStyleSheet("border: none;")

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1 / 1.15
        self.scale(factor, factor)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            scene = self.scene()
            if isinstance(scene, FlowScene) and scene._rubber_band:
                scene._finish_edge_drag(None)
                return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            # Deletion handled by FlowPage
            self.scene().notify_layout_dirty()
        super().keyPressEvent(event)

    def fit_view(self) -> None:
        rect = self.scene().itemsBoundingRect()
        if not rect.isEmpty():
            self.fitInView(rect.adjusted(-20, -20, 20, 20),
                           Qt.AspectRatioMode.KeepAspectRatio)

    # ------------------------------------------------------------------
    # Viewport event filter — catches drag events before Qt's internal
    # QGraphicsView routing swallows them.
    # ------------------------------------------------------------------

    def eventFilter(self, source, event) -> bool:
        if source is self.viewport():
            t = event.type()
            if t == QEvent.Type.DragEnter:
                if event.mimeData().hasFormat("application/x-wave-name"):
                    event.acceptProposedAction()
                    return True
            elif t == QEvent.Type.DragMove:
                if event.mimeData().hasFormat("application/x-wave-name"):
                    event.acceptProposedAction()
                    return True
            elif t == QEvent.Type.Drop:
                if event.mimeData().hasFormat("application/x-wave-name"):
                    wave_name = (event.mimeData()
                                 .data("application/x-wave-name")
                                 .data().decode("utf-8"))
                    scene_pos = self.mapToScene(event.position().toPoint())
                    sc = self.scene()
                    if isinstance(sc, FlowScene):
                        sc.node_dropped.emit(wave_name, scene_pos)
                    event.acceptProposedAction()
                    return True
        return super().eventFilter(source, event)
