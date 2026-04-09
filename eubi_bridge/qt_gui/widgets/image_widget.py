"""
Image display widget — renders numpy RGB arrays with anisotropy correction,
pan support, and a timing overlay.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPixmap
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

        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)

    # ── Public API ────────────────────────────────────────────────────────────

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

    def _display_rect(self) -> tuple[int, int, int, int]:
        """(x, y, display_w, display_h) of the pixmap in widget coordinates.

        Pixels are shown 1:1 (no fit-to-window scaling).  The only transform
        applied is anisotropy correction: the y axis is stretched by self._aspect
        so that physical space proportions are preserved.  The image is centred
        in the widget; if it is smaller than the widget black borders appear.
        """
        if self._pixmap is None:
            return 0, 0, self.width(), self.height()
        img_w = self._pixmap.width()
        img_h = max(1, int(self._pixmap.height() * self._aspect))  # rows × physical-size-per-row
        x = (self.width()  - img_w) // 2
        y = (self.height() - img_h) // 2
        return x, y, img_w, img_h

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

        x, y, dw, dh = self._display_rect()
        painter.drawPixmap(x, y, dw, dh, self._pixmap)

        if abs(self._aspect - 1.0) > 0.01:
            painter.setPen(QColor(100, 200, 255))
            painter.drawText(x + 6, y + 32, f"AR {self._aspect:.3f}")

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
