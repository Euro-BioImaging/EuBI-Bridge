"""
Image display widget — renders numpy RGB arrays with anisotropy correction,
pan support, and a timing overlay.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QSizePolicy, QWidget


class ImageWidget(QWidget):
    """Displays a rendered zarr plane.

    Signals:
        pan_changed(delta_row, delta_col)  — full-resolution coordinate deltas
        pan_released()                     — left mouse button released after drag
        wheel_scrolled(delta)              — +1 / -1 from mouse wheel
    """

    pan_changed    = pyqtSignal(float, float)  # delta_row, delta_col
    pan_released   = pyqtSignal()              # mouse-up after drag
    wheel_scrolled = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._aspect  = 1.0     # physical h/w ratio for anisotropy correction
        self._elapsed = 0.0     # last render time (ms), shown as overlay
        self._drag_start: tuple[float, float] | None = None
        # Axis labels: (horizontal, vertical, through-plane)
        self._ax_h: str = "X"
        self._ax_v: str = "Y"
        self._ax_t: str = "Z"

        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_axes(self, h_axis: str, v_axis: str, through_axis: str):
        """Update orientation labels (call before or after set_frame)."""
        self._ax_h = h_axis.upper()
        self._ax_v = v_axis.upper()
        self._ax_t = through_axis.upper()
        self.update()

    def set_frame(self, rgb: np.ndarray, aspect: float, elapsed: float):
        """Update the displayed image.  Called from the main thread."""
        h, w = rgb.shape[:2]
        rgb_c = np.ascontiguousarray(rgb)
        qimg = QImage(rgb_c.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._pixmap  = QPixmap.fromImage(qimg.copy())
        self._aspect  = max(0.01, aspect)
        self._elapsed = elapsed
        self.update()

    def add_pan_offset(self, dy_screen: float, dx_screen: float):
        """No-op: pan feedback is handled by immediate re-renders, not offsets."""
        pass

    def clear(self):
        self._pixmap = None
        self.update()

    # ── Internal geometry helper ──────────────────────────────────────────────

    def _display_rect(self) -> tuple[float, float, float, float]:
        """(x, y, display_w, display_h) of the pixmap in *logical* widget coordinates.

        Pixels are shown 1:1 in *device* (physical) pixels — i.e. one data pixel
        maps to exactly one screen pixel regardless of QT_SCALE_FACTOR.  In
        logical coordinates the image therefore occupies pixmap_px / DPR pixels.
        The only additional transform is anisotropy correction on the y axis.
        The image is centred in the widget; smaller images get black borders.
        """
        if self._pixmap is None:
            return 0, 0, self.width(), self.height()
        dpr = self.devicePixelRatioF()
        img_w_log = self._pixmap.width()  / dpr
        img_h_log = max(1.0, self._pixmap.height() * self._aspect / dpr)
        x = (self.width()  - img_w_log) / 2
        y = (self.height() - img_h_log) / 2
        return x, y, img_w_log, img_h_log

    # ── Paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 20))

        if self._pixmap is None:
            painter.setPen(QColor(90, 90, 90))
            painter.setFont(QFont("sans-serif", 11))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Select an OME-Zarr dataset to view",
            )
            return

        # Paint in device pixel space so the image is 1:1 with screen pixels,
        # unaffected by QT_SCALE_FACTOR.
        dpr = self.devicePixelRatioF()
        painter.scale(1.0 / dpr, 1.0 / dpr)

        img_w = self._pixmap.width()
        img_h = max(1, int(self._pixmap.height() * self._aspect))
        # Centre the image in device pixel space
        x_dev = int((self.width()  * dpr - img_w) / 2)
        y_dev = int((self.height() * dpr - img_h) / 2)
        painter.drawPixmap(x_dev, y_dev, img_w, img_h, self._pixmap)

        if abs(self._aspect - 1.0) > 0.01:
            painter.setPen(QColor(100, 200, 255))
            painter.drawText(x_dev + 6, y_dev + 32, f"AR {self._aspect:.3f}")

        self._draw_axis_lines(painter, x_dev, y_dev, img_w, img_h)

    # ── Axis overlay ──────────────────────────────────────────────────────────

    def _draw_axis_lines(self, painter: QPainter,
                         x0: int, y0: int, w: int, h: int):
        """Draw X/Y/Z axis arrows in the dark border outside the image frame.

        The shared origin sits just outside the bottom-left corner of the image.
        The horizontal arrow runs rightward below the image; the vertical arrow
        runs upward to the left of the image.  The through-axis label appears in
        the dark area above the top-right corner.
        """
        ARROW   = 32      # arrow shaft length in px
        TICK    = 5       # arrowhead arm half-length
        GAP     = 6       # gap between image edge and origin point
        FONT_SZ = 9

        font = QFont("sans-serif", FONT_SZ)
        font.setBold(True)
        painter.setFont(font)

        # Axis colours: X=red, Y=green, Z=blue
        _C = {"X": QColor(220, 60, 60), "Y": QColor(60, 200, 80), "Z": QColor(60, 140, 255)}

        # Shared origin: just outside the bottom-left corner of the image
        ox = x0 - GAP
        oy = y0 + h + GAP

        def draw_arrow_h(ax_name: str):
            """Rightward arrow below the image, starting from origin."""
            c = _C.get(ax_name, QColor(200, 200, 200))
            pen = QPen(c, 1.5)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            tip_x = ox + ARROW
            painter.drawLine(ox, oy, tip_x, oy)
            painter.drawLine(tip_x, oy, tip_x - TICK, oy - TICK)
            painter.drawLine(tip_x, oy, tip_x - TICK, oy + TICK)
            painter.setPen(c)
            painter.drawText(tip_x + 3, oy + 4, ax_name)

        def draw_arrow_v(ax_name: str):
            """Upward arrow to the left of the image, starting from origin."""
            c = _C.get(ax_name, QColor(200, 200, 200))
            pen = QPen(c, 1.5)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            tip_y = oy - ARROW
            painter.drawLine(ox, oy, ox, tip_y)
            painter.drawLine(ox, tip_y, ox - TICK, tip_y + TICK)
            painter.drawLine(ox, tip_y, ox + TICK, tip_y + TICK)
            painter.setPen(c)
            painter.drawText(ox + 4, tip_y - 2, ax_name)

        draw_arrow_h(self._ax_h)
        draw_arrow_v(self._ax_v)

        # Small origin dot at the corner
        painter.setPen(QColor(160, 160, 160))
        painter.drawPoint(ox, oy)

        # Through-axis label in the dark area above the top-right corner
        c_t = _C.get(self._ax_t, QColor(200, 200, 200))
        painter.setPen(c_t)
        painter.setFont(QFont("sans-serif", FONT_SZ))
        painter.drawText(x0 + w + 4, y0 - GAP // 2, f"⊙ {self._ax_t}")

    # ── Mouse events ──────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = (event.position().x(), event.position().y())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._drag_start is None or self._pixmap is None:
            return
        px, py = event.position().x(), event.position().y()
        dx = px - self._drag_start[0]
        dy = py - self._drag_start[1]
        self._drag_start = (px, py)

        _, _, dw, dh = self._display_rect()
        img_w = self._pixmap.width()
        img_h = max(1, int(self._pixmap.height() * self._aspect))
        scale_x = dw / max(img_w, 1)
        scale_y = dh / max(img_h, 1)

        delta_col = -dx / scale_x
        delta_row = -dy / scale_y / self._aspect
        self.pan_changed.emit(delta_row, delta_col)

    def mouseReleaseEvent(self, _event):
        self._drag_start = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.pan_released.emit()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta != 0:
            self.wheel_scrolled.emit(1 if delta < 0 else -1)
        event.accept()
