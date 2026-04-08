#!/usr/bin/env python3
"""
Qt-based OME-Zarr viewer — performance prototype.

No HTTP layer: data flows directly  zarr/TensorStore → numpy → QImage → QWidget.
Compare against the web viewer to measure the HTTP-overhead and Canvas-decode cost.

Usage:
    python qt_gui/viewer.py
    python qt_gui/viewer.py /path/to/dataset.zarr
    python qt_gui/viewer.py s3://bucket/path/dataset.zarr

Requirements (already in pyproject.toml [gui-qt6]):
    PyQt6>=6.6  numpy  Pillow  eubi_bridge
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# ── Optional TensorStore ─────────────────────────────────────────────────────
try:
    import tensorstore as _ts_mod

    _TS_AVAILABLE = True
    try:
        _ts_context = _ts_mod.Context(
            {
                "cache_pool": {"total_bytes_limit": 512_000_000},
                "s3_request_concurrency": {"limit": 32},
                "gcs_request_concurrency": {"limit": 32},
                "file_io_concurrency": {"limit": 64},
            }
        )
    except Exception:
        _ts_context = None
except (ImportError, SyntaxError):
    _TS_AVAILABLE = False
    _ts_context = None

# ── eubi_bridge Pyramid ──────────────────────────────────────────────────────
try:
    from eubi_bridge.ngff.multiscales import Pyramid

    _PYRAMID_OK = True
except ImportError:
    _PYRAMID_OK = False
    Pyramid = None  # type: ignore

# ── PyQt6 ────────────────────────────────────────────────────────────────────
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Rendering helpers  (adapted from zarr_plane_server.py — no HTTP, no caching)
# ═══════════════════════════════════════════════════════════════════════════════

ORIENTATIONS = {
    "XY": {"h_axis": "x", "v_axis": "y"},
    "XZ": {"h_axis": "x", "v_axis": "z"},
    "YZ": {"h_axis": "z", "v_axis": "y"},
}


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        return (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        )
    return (255, 255, 255)


# ── Pyramid LRU cache ────────────────────────────────────────────────────────

class _PyramidCache:
    def __init__(self, max_size: int = 10):
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_size

    def get(self, path: str):
        with self._lock:
            if path in self._cache:
                self._cache.move_to_end(path)
                return self._cache[path]
        pyr = Pyramid(path)
        with self._lock:
            self._cache[path] = pyr
            self._cache.move_to_end(path)
            if len(self._cache) > self._max:
                self._cache.popitem(last=False)
        return pyr


_pyramid_cache = _PyramidCache()

# ── TensorStore handle cache ─────────────────────────────────────────────────

class _TSCache:
    def __init__(self, max_size: int = 50):
        self._cache: OrderedDict = OrderedDict()
        self._failed: set = set()
        self._lock = threading.Lock()
        self._max = max_size

    def get(self, zarr_path: str, level_path: str):
        if not _TS_AVAILABLE:
            return None
        key = (zarr_path, level_path)
        with self._lock:
            if key in self._failed:
                return None
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        try:
            spec = _build_ts_kvstore(zarr_path, level_path)
            store = _ts_mod.open(spec, read=True, context=_ts_context).result()
            with self._lock:
                self._cache[key] = store
                self._cache.move_to_end(key)
                if len(self._cache) > self._max:
                    self._cache.popitem(last=False)
            return store
        except Exception as e:
            with self._lock:
                self._failed.add(key)
            return None


def _build_ts_kvstore(zarr_path: str, level_path: str) -> dict:
    full_path = zarr_path.rstrip("/") + "/" + level_path if level_path else zarr_path
    if zarr_path.startswith("s3://"):
        without_scheme = zarr_path.removeprefix("s3://")
        bucket, _, prefix = without_scheme.partition("/")
        obj_key = "/".join(filter(None, [prefix.rstrip("/"), level_path]))
        return {
            "driver": "zarr",
            "kvstore": {"driver": "s3", "bucket": bucket, "path": obj_key + "/"},
        }
    if zarr_path.startswith("gs://"):
        without_scheme = zarr_path.removeprefix("gs://")
        bucket, _, prefix = without_scheme.partition("/")
        obj_key = "/".join(filter(None, [prefix.rstrip("/"), level_path]))
        return {
            "driver": "zarr",
            "kvstore": {"driver": "gcs", "bucket": bucket, "path": obj_key + "/"},
        }
    if zarr_path.startswith(("http://", "https://")):
        return {"driver": "zarr", "kvstore": {"driver": "http", "base_url": full_path + "/"}}
    return {"driver": "zarr", "kvstore": {"driver": "file", "path": full_path}}


_ts_cache = _TSCache()

# ── Orientation helpers ──────────────────────────────────────────────────────

_ori_cache: dict = {}
_ori_lock = threading.Lock()


def get_orientation_axes(pyr, level_path: str, orientation: str) -> dict:
    key = (id(pyr), level_path, orientation)
    with _ori_lock:
        if key in _ori_cache:
            return _ori_cache[key]

    ori = ORIENTATIONS.get(orientation, ORIENTATIONS["XY"])
    h_axis = ori["h_axis"]
    v_axis = ori["v_axis"]

    layer = pyr.layers[level_path]
    layer_shape = layer.shape
    axes = pyr.axes.lower()
    base_shape = pyr.shape

    h_idx = axes.find(h_axis)
    v_idx = axes.find(v_axis)
    if h_idx < 0:
        h_idx = len(axes) - 1
    if v_idx < 0:
        v_idx = len(axes) - 2 if len(axes) > 1 else len(axes) - 1

    def _scale(i):
        if i < len(base_shape) and i < len(layer_shape) and layer_shape[i] > 0:
            return base_shape[i] / layer_shape[i]
        return 1.0

    layer_height = layer_shape[v_idx] if v_idx < len(layer_shape) else layer_shape[-2]
    layer_width = layer_shape[h_idx] if h_idx < len(layer_shape) else layer_shape[-1]

    chunks = layer.chunks if hasattr(layer, "chunks") else layer_shape
    chunk_h = chunks[v_idx] if v_idx < len(chunks) else layer_height
    chunk_w = chunks[h_idx] if h_idx < len(chunks) else layer_width

    result = {
        "h_axis": h_axis, "v_axis": v_axis,
        "h_idx": h_idx, "v_idx": v_idx,
        "h_scale": _scale(h_idx), "v_scale": _scale(v_idx),
        "layer_height": layer_height, "layer_width": layer_width,
        "chunk_h": chunk_h, "chunk_w": chunk_w,
        "layer": layer, "axes": axes, "base_shape": base_shape, "layer_shape": layer_shape,
    }
    with _ori_lock:
        _ori_cache[key] = result
    return result


def compute_fov_region(info: dict, fov_center: tuple, fov_size: int):
    center_row = int(fov_center[0] / info["v_scale"])
    center_col = int(fov_center[1] / info["h_scale"])
    fov_h = min(fov_size, info["layer_height"])
    fov_w = min(fov_size, info["layer_width"])
    row_start = max(0, min(center_row - fov_h // 2, info["layer_height"] - fov_h))
    col_start = max(0, min(center_col - fov_w // 2, info["layer_width"] - fov_w))
    return row_start, row_start + fov_h, col_start, col_start + fov_w, fov_h, fov_w


def build_slices(info: dict, row_start, row_end, col_start, col_end, indices, channel_idx=None):
    axes = info["axes"]
    base_shape = info["base_shape"]
    layer_shape = info["layer_shape"]
    h_axis = info["h_axis"]
    v_axis = info["v_axis"]
    slices = []
    for i, ax in enumerate(axes):
        if ax == h_axis:
            slices.append(slice(col_start, col_end))
        elif ax == v_axis:
            slices.append(slice(row_start, row_end))
        elif ax == "c":
            slices.append(channel_idx if channel_idx is not None else slice(None))
        elif ax in indices:
            ratio = base_shape[i] / layer_shape[i] if layer_shape[i] > 0 else 1
            slices.append(max(0, min(int(indices[ax] / ratio), layer_shape[i] - 1)))
        else:
            slices.append(0)
    return slices


def fix_axis_order(data: np.ndarray, slices, info: dict) -> np.ndarray:
    axes = info["axes"]
    h_axis = info["h_axis"]
    v_axis = info["v_axis"]
    remaining = [ax for i, ax in enumerate(axes) if not isinstance(slices[i], int)]
    if data.ndim >= 2:
        h_pos = next((i for i, ax in enumerate(remaining) if ax == h_axis), -1)
        v_pos = next((i for i, ax in enumerate(remaining) if ax == v_axis), -1)
        if h_pos >= 0 and v_pos >= 0 and h_pos < v_pos:
            if data.ndim == 2:
                data = data.T
            elif data.ndim == 3:
                c_pos = remaining.index("c") if "c" in remaining else -1
                if c_pos >= 0:
                    data = np.moveaxis(data, c_pos, 0)
                    spatial = list(range(1, data.ndim))
                    spatial.reverse()
                    data = np.transpose(data, [0] + spatial)
    return data


def extract_region(pyr, level_path: str, orientation: str,
                   row_start, row_end, col_start, col_end,
                   indices: dict, channel_idx=None, zarr_path: str | None = None) -> np.ndarray:
    info = get_orientation_axes(pyr, level_path, orientation)
    slices = build_slices(info, row_start, row_end, col_start, col_end, indices, channel_idx)
    ts_store = _ts_cache.get(zarr_path, level_path) if zarr_path else None
    if ts_store is not None:
        data = np.asarray(ts_store[tuple(slices)].read().result())
    else:
        data = info["layer"][tuple(slices)]
        if hasattr(data, "compute"):
            data = data.compute()
        data = np.asarray(data)
    return fix_axis_order(data, slices, info)


def extract_plane(pyr, level_path: str, orientation: str, indices: dict,
                  fov_center: tuple, fov_size: int,
                  channel_idx=None, zarr_path: str | None = None):
    info = get_orientation_axes(pyr, level_path, orientation)
    row_start, row_end, col_start, col_end, _, _ = compute_fov_region(info, fov_center, fov_size)
    data = extract_region(pyr, level_path, orientation,
                          row_start, row_end, col_start, col_end,
                          indices, channel_idx, zarr_path=zarr_path)
    return data, {"fov": [row_start, row_end, col_start, col_end]}


def normalize_and_composite(plane_data: np.ndarray, channels_config: list) -> np.ndarray:
    def _norm_channel(ch_data, vmin, vmax):
        ch_data = ch_data.astype(np.float32)
        if vmin is None or vmax is None or vmax <= vmin:
            vmin = float(np.percentile(ch_data, 1))
            vmax = float(np.percentile(ch_data, 99))
        if vmax > vmin:
            return np.clip((ch_data - vmin) / (vmax - vmin), 0, 1)
        return np.zeros_like(ch_data)

    if plane_data.ndim == 2:
        cfg = channels_config[0] if channels_config else {"color": "#FFFFFF"}
        normed = _norm_channel(plane_data, cfg.get("intensityMin"), cfg.get("intensityMax"))
        r, g, b = hex_to_rgb(cfg.get("color", "#FFFFFF"))
        h, w = normed.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[..., 0] = (normed * r).astype(np.uint8)
        rgb[..., 1] = (normed * g).astype(np.uint8)
        rgb[..., 2] = (normed * b).astype(np.uint8)
        return rgb

    # ndim == 3: (channels, h, w)
    n_ch = plane_data.shape[0]
    h, w = plane_data.shape[1], plane_data.shape[2]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(min(n_ch, len(channels_config))):
        cfg = channels_config[i]
        if not cfg.get("visible", True):
            continue
        normed = _norm_channel(plane_data[i], cfg.get("intensityMin"), cfg.get("intensityMax"))
        r, g, b = hex_to_rgb(cfg.get("color", "#FFFFFF"))
        rgb[..., 0] += normed * (r / 255.0)
        rgb[..., 1] += normed * (g / 255.0)
        rgb[..., 2] += normed * (b / 255.0)
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# Background render thread
# ═══════════════════════════════════════════════════════════════════════════════

class RenderThread(QThread):
    frame_ready = pyqtSignal(object, float)   # (rgb ndarray, elapsed_ms)
    render_error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cond = threading.Condition()
        self._pending: dict | None = None
        self._stop = False

    def request(self, params: dict):
        with self._cond:
            self._pending = params
            self._cond.notify()

    def stop(self):
        with self._cond:
            self._stop = True
            self._cond.notify()

    def run(self):
        executor = ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 2) * 2))
        while True:
            with self._cond:
                while self._pending is None and not self._stop:
                    self._cond.wait(timeout=0.1)
                if self._stop:
                    break
                params = self._pending
                self._pending = None

            try:
                t0 = time.perf_counter()
                rgb = _do_render(params, executor)
                elapsed = (time.perf_counter() - t0) * 1000
                self.frame_ready.emit(rgb, elapsed)
            except Exception as exc:
                self.render_error.emit(str(exc))
        executor.shutdown(wait=False)


def _do_render(p: dict, executor: ThreadPoolExecutor) -> np.ndarray:
    pyr = _pyramid_cache.get(p["path"])
    level_path = pyr.meta.resolution_paths[p["level_idx"]]
    axes = pyr.axes.lower()
    has_c = "c" in axes
    channels_config = p["channels"]

    if has_c and len(channels_config) > 1:
        visible = [ch for ch in channels_config if ch.get("visible", True)]
        if not visible:
            visible = [channels_config[0]]
        results: list = [None] * len(visible)

        def fetch(i, vc):
            results[i] = extract_plane(
                pyr, level_path, p["orientation"], p["indices"],
                p["fov_center"], p["fov_size"],
                channel_idx=vc["index"], zarr_path=p["path"],
            )[0]

        futs = [executor.submit(fetch, i, vc) for i, vc in enumerate(visible)]
        for f in futs:
            f.result()
        combined = results[0] if len(results) == 1 else np.stack(results, axis=0)
        return normalize_and_composite(combined, visible)

    plane_data, _ = extract_plane(
        pyr, level_path, p["orientation"], p["indices"],
        p["fov_center"], p["fov_size"], zarr_path=p["path"],
    )
    return normalize_and_composite(plane_data, channels_config)


# ═══════════════════════════════════════════════════════════════════════════════
# Image display widget
# ═══════════════════════════════════════════════════════════════════════════════

class ImageWidget(QWidget):
    pan_changed = pyqtSignal(float, float)  # delta_row, delta_col (full-res coords)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._aspect = 1.0          # physical h/w ratio (>1 = tall pixels)
        self._elapsed = 0.0
        self._drag_start: tuple | None = None
        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    # called from main thread after render completes
    def set_frame(self, rgb: np.ndarray, aspect: float, elapsed: float):
        h, w = rgb.shape[:2]
        # QImage requires contiguous C array
        rgb_c = np.ascontiguousarray(rgb)
        qimg = QImage(rgb_c.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg.copy())
        self._aspect = max(0.01, aspect)
        self._elapsed = elapsed
        self.update()

    def _display_rect(self) -> tuple[int, int, int, int]:
        """Return (x, y, disp_w, disp_h) for the current pixmap in widget coords."""
        if self._pixmap is None:
            return 0, 0, self.width(), self.height()
        img_w = self._pixmap.width()
        img_h = int(img_w * self._aspect)
        scale = min(self.width() / max(img_w, 1), self.height() / max(img_h, 1))
        disp_w = max(1, int(img_w * scale))
        disp_h = max(1, int(img_h * scale))
        x = (self.width() - disp_w) // 2
        y = (self.height() - disp_h) // 2
        return x, y, disp_w, disp_h

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 20))
        if self._pixmap is None:
            painter.setPen(QColor(100, 100, 100))
            painter.setFont(QFont("sans-serif", 11))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Enter an OME-Zarr path and click Load")
            return

        x, y, dw, dh = self._display_rect()
        painter.drawPixmap(x, y, dw, dh, self._pixmap)

        # Timing overlay
        painter.setPen(QColor(255, 200, 50))
        painter.setFont(QFont("monospace", 9))
        painter.drawText(x + 6, y + 16, f"{self._elapsed:.0f} ms/frame")
        if abs(self._aspect - 1.0) > 0.01:
            painter.setPen(QColor(100, 200, 255))
            painter.drawText(x + 6, y + 32, f"AR {self._aspect:.3f}")

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
        img_h = int(img_w * self._aspect)
        # pixels per image-pixel on screen
        scale_x = dw / max(img_w, 1)
        scale_y = dh / max(img_h, 1)
        # panning: move center in opposite direction
        delta_col = -dx / scale_x
        delta_row = -dy / scale_y / self._aspect
        self.pan_changed.emit(delta_row, delta_col)

    def mouseReleaseEvent(self, _event):
        self._drag_start = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)


# ═══════════════════════════════════════════════════════════════════════════════
# Main window
# ═══════════════════════════════════════════════════════════════════════════════

_DEFAULT_COLORS = [
    "#FFFFFF", "#FF0000", "#00FF00", "#0000FF",
    "#FFFF00", "#FF00FF", "#00FFFF", "#FF8000",
]


class ViewerWindow(QMainWindow):
    def __init__(self, initial_path: str = ""):
        super().__init__()
        self.setWindowTitle("EuBI-Bridge Qt Viewer")
        self.resize(1280, 800)

        # ── State ────────────────────────────────────────────────────────────
        self._path = ""
        self._pyr = None
        self._axes = ""
        self._shape: list[int] = []
        self._level_paths: list[str] = []
        self._channels: list[dict] = []
        self._level_idx = 0
        self._orientation = "XY"
        self._t = 0
        self._z = 0
        self._fov_size = 512
        self._fov_center_y = 0
        self._fov_center_x = 0
        self._pixel_sizes: list[dict] = []   # [{"axis": "z", "value": 1.0}, ...]

        # ── Render thread ────────────────────────────────────────────────────
        self._render_thread = RenderThread(self)
        self._render_thread.frame_ready.connect(self._on_frame_ready)
        self._render_thread.render_error.connect(self._on_render_error)
        self._render_thread.start()

        # ── Debounce timer ───────────────────────────────────────────────────
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(40)
        self._debounce.timeout.connect(self._trigger_render)

        # ── Build UI ─────────────────────────────────────────────────────────
        self._build_ui()

        if initial_path:
            self._path_edit.setText(initial_path)
            self._load_dataset(initial_path)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # Left: controls panel
        controls = QWidget()
        controls.setFixedWidth(300)
        controls.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        ctrl_layout = QVBoxLayout(controls)
        ctrl_layout.setContentsMargins(6, 6, 6, 6)
        ctrl_layout.setSpacing(6)

        # Path input
        path_group = QGroupBox("Dataset")
        path_layout = QVBoxLayout(path_group)
        row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("OME-Zarr path or s3://...")
        self._path_edit.returnPressed.connect(self._on_load_clicked)
        row.addWidget(self._path_edit)
        browse_btn = QPushButton("…")
        browse_btn.setFixedWidth(28)
        browse_btn.setToolTip("Browse for local .zarr folder")
        browse_btn.clicked.connect(self._on_browse_clicked)
        row.addWidget(browse_btn)
        path_layout.addLayout(row)
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._on_load_clicked)
        path_layout.addWidget(load_btn)
        self._status_label = QLabel("No dataset loaded")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: gray; font-size: 10px;")
        path_layout.addWidget(self._status_label)
        ctrl_layout.addWidget(path_group)

        # View controls
        view_group = QGroupBox("View")
        view_layout = QVBoxLayout(view_group)

        # Orientation
        ori_row = QHBoxLayout()
        ori_row.addWidget(QLabel("Orientation:"))
        self._ori_combo = QComboBox()
        self._ori_combo.addItems(["XY", "XZ", "YZ"])
        self._ori_combo.currentTextChanged.connect(self._on_orientation_changed)
        ori_row.addWidget(self._ori_combo)
        view_layout.addLayout(ori_row)

        # Zoom level
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("Zoom level:"))
        self._zoom_label = QLabel("0")
        self._zoom_label.setFixedWidth(20)
        zoom_row.addWidget(self._zoom_label)
        view_layout.addLayout(zoom_row)
        self._zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self._zoom_slider.setMinimum(0)
        self._zoom_slider.setMaximum(0)
        self._zoom_slider.valueChanged.connect(self._on_zoom_changed)
        view_layout.addWidget(self._zoom_slider)

        # FOV size
        fov_row = QHBoxLayout()
        fov_row.addWidget(QLabel("FOV size:"))
        self._fov_combo = QComboBox()
        for s in [128, 256, 512, 1024]:
            self._fov_combo.addItem(str(s), s)
        self._fov_combo.setCurrentIndex(2)  # 512
        self._fov_combo.currentIndexChanged.connect(self._on_fov_changed)
        fov_row.addWidget(self._fov_combo)
        view_layout.addLayout(fov_row)

        ctrl_layout.addWidget(view_group)

        # Sliders
        slice_group = QGroupBox("Slices")
        slice_layout = QVBoxLayout(slice_group)

        t_row = QHBoxLayout()
        t_row.addWidget(QLabel("T:"))
        self._t_label = QLabel("0")
        self._t_label.setFixedWidth(35)
        t_row.addWidget(self._t_label)
        slice_layout.addLayout(t_row)
        self._t_slider = QSlider(Qt.Orientation.Horizontal)
        self._t_slider.setMinimum(0)
        self._t_slider.setMaximum(0)
        self._t_slider.valueChanged.connect(self._on_t_changed)
        slice_layout.addWidget(self._t_slider)

        z_row = QHBoxLayout()
        z_row.addWidget(QLabel("Z:"))
        self._z_label = QLabel("0")
        self._z_label.setFixedWidth(35)
        z_row.addWidget(self._z_label)
        slice_layout.addLayout(z_row)
        self._z_slider = QSlider(Qt.Orientation.Horizontal)
        self._z_slider.setMinimum(0)
        self._z_slider.setMaximum(0)
        self._z_slider.valueChanged.connect(self._on_z_changed)
        slice_layout.addWidget(self._z_slider)

        ctrl_layout.addWidget(slice_group)

        # Channels
        ch_group = QGroupBox("Channels")
        ch_inner = QWidget()
        self._ch_layout = QVBoxLayout(ch_inner)
        self._ch_layout.setContentsMargins(2, 2, 2, 2)
        self._ch_layout.setSpacing(2)
        ch_scroll = QScrollArea()
        ch_scroll.setWidget(ch_inner)
        ch_scroll.setWidgetResizable(True)
        ch_scroll.setMaximumHeight(220)
        ch_group_layout = QVBoxLayout(ch_group)
        ch_group_layout.addWidget(ch_scroll)
        ctrl_layout.addWidget(ch_group)

        ctrl_layout.addStretch()

        # Right: image
        self._image_widget = ImageWidget()
        self._image_widget.pan_changed.connect(self._on_pan)

        splitter.addWidget(controls)
        splitter.addWidget(self._image_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    # ── Dataset loading ───────────────────────────────────────────────────────

    def _on_browse_clicked(self):
        folder = QFileDialog.getExistingDirectory(self, "Select OME-Zarr folder")
        if folder:
            self._path_edit.setText(folder)
            self._load_dataset(folder)

    def _on_load_clicked(self):
        path = self._path_edit.text().strip()
        if path:
            self._load_dataset(path)

    def _load_dataset(self, path: str):
        self._status_label.setText("Loading…")
        QApplication.processEvents()
        try:
            pyr = _pyramid_cache.get(path)
        except Exception as exc:
            self._status_label.setText(f"Error: {exc}")
            return

        self._path = path
        self._pyr = pyr
        self._axes = pyr.axes.lower()
        self._shape = list(pyr.shape)
        self._level_paths = list(pyr.meta.resolution_paths)

        # Pixel sizes from metadata
        self._pixel_sizes = self._read_pixel_sizes(path)

        # Channel info
        self._channels = self._read_channels(path, pyr)

        # Sliders
        def _dim(ax):
            idx = self._axes.find(ax)
            return self._shape[idx] if idx >= 0 else 1

        t_max = max(0, _dim("t") - 1)
        z_max = max(0, _dim("z") - 1)
        self._t_slider.setMaximum(t_max)
        self._t_slider.setValue(t_max // 2)
        self._z_slider.setMaximum(z_max)
        self._z_slider.setValue(z_max // 2)

        n_levels = len(self._level_paths)
        self._zoom_slider.setMaximum(max(0, n_levels - 1))
        self._zoom_slider.setValue(0)

        # FOV center
        y_max = _dim("y")
        x_max = _dim("x")
        self._fov_center_y = y_max // 2
        self._fov_center_x = x_max // 2

        # Rebuild channel widgets
        self._rebuild_channel_widgets()

        axes_str = self._axes.upper()
        shape_str = "×".join(str(s) for s in self._shape)
        self._status_label.setText(f"Axes: {axes_str}\nShape: {shape_str}\nLevels: {n_levels}")

        self._schedule_render()

    def _read_pixel_sizes(self, path: str) -> list[dict]:
        """Try to read pixel sizes from .zattrs / zarr.json."""
        try:
            for fname in [".zattrs", "zarr.json"]:
                fpath = os.path.join(path, fname)
                if not os.path.exists(fpath):
                    continue
                with open(fpath) as f:
                    data = json.load(f)
                # Unwrap zarr v3 attributes nesting
                attrs = data.get("attributes", data)
                # Try ome-zarr 0.5+
                ome = attrs.get("ome", attrs)
                ms_list = ome.get("multiscales", attrs.get("multiscales", []))
                if not ms_list:
                    continue
                ms = ms_list[0]
                axes = ms.get("axes", [])
                datasets = ms.get("datasets", [])
                if not datasets:
                    continue
                # Use finest level (index 0)
                ct = datasets[0].get("coordinateTransformations", [])
                for xform in ct:
                    if xform.get("type") == "scale":
                        scales = xform.get("scale", [])
                        return [
                            {"axis": ax.get("name", "?"), "value": float(s)}
                            for ax, s in zip(axes, scales)
                        ]
        except Exception:
            pass
        return []

    def _read_channels(self, path: str, pyr) -> list[dict]:
        """Extract channel info from omero metadata, or create defaults."""
        axes = pyr.axes.lower()
        c_idx = axes.find("c")
        n_ch = pyr.shape[c_idx] if c_idx >= 0 else 1
        defaults = [
            {"index": i, "label": f"Ch {i}", "color": _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)],
             "visible": True, "intensityMin": None, "intensityMax": None}
            for i in range(n_ch)
        ]
        try:
            for fname in [".zattrs", "zarr.json"]:
                fpath = os.path.join(path, fname)
                if not os.path.exists(fpath):
                    continue
                with open(fpath) as f:
                    data = json.load(f)
                attrs = data.get("attributes", data)
                omero = attrs.get("omero", None)
                if omero is None:
                    continue
                ch_list = omero.get("channels", [])
                for i, ch in enumerate(ch_list):
                    if i >= len(defaults):
                        break
                    win = ch.get("window", {})
                    defaults[i]["label"] = ch.get("label", defaults[i]["label"])
                    color = ch.get("color", "FFFFFF")
                    defaults[i]["color"] = f"#{color}" if not color.startswith("#") else color
                    defaults[i]["intensityMin"] = win.get("start", None)
                    defaults[i]["intensityMax"] = win.get("end", None)
                    defaults[i]["visible"] = ch.get("active", True)
                break
        except Exception:
            pass
        return defaults

    def _rebuild_channel_widgets(self):
        # Clear existing widgets
        while self._ch_layout.count():
            item = self._ch_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for ch in self._channels:
            row_widget = QWidget()
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(2, 1, 2, 1)

            vis_cb = QCheckBox()
            vis_cb.setChecked(ch["visible"])
            vis_cb.stateChanged.connect(
                lambda state, c=ch: self._on_channel_visibility(c, state)
            )
            row.addWidget(vis_cb)

            color_btn = QPushButton()
            color_btn.setFixedSize(16, 16)
            color_btn.setStyleSheet(
                f"background-color: {ch['color']}; border: 1px solid #555; border-radius: 3px;"
            )
            row.addWidget(color_btn)

            lbl = QLabel(ch["label"])
            lbl.setStyleSheet("font-size: 10px;")
            row.addWidget(lbl)
            row.addStretch()

            # Min/max spin boxes
            for key, placeholder in [("intensityMin", "min"), ("intensityMax", "max")]:
                spin = QDoubleSpinBox()
                spin.setRange(-1e9, 1e9)
                spin.setDecimals(1)
                spin.setFixedWidth(72)
                spin.setSpecialValueText("auto")
                spin.setValue(spin.minimum())  # "auto"
                if ch[key] is not None:
                    spin.setValue(float(ch[key]))
                spin.setToolTip(placeholder)
                spin.valueChanged.connect(
                    lambda v, c=ch, k=key, s=spin: self._on_channel_range(c, k, v, s)
                )
                row.addWidget(spin)

            self._ch_layout.addWidget(row_widget)

    # ── Control callbacks ─────────────────────────────────────────────────────

    def _on_orientation_changed(self, text: str):
        self._orientation = text
        self._schedule_render()

    def _on_zoom_changed(self, value: int):
        self._level_idx = value
        self._zoom_label.setText(str(value))
        self._schedule_render()

    def _on_fov_changed(self, _idx: int):
        self._fov_size = self._fov_combo.currentData()
        self._schedule_render()

    def _on_t_changed(self, value: int):
        self._t = value
        self._t_label.setText(str(value))
        self._schedule_render()

    def _on_z_changed(self, value: int):
        self._z = value
        self._z_label.setText(str(value))
        self._schedule_render()

    def _on_channel_visibility(self, ch: dict, state: int):
        ch["visible"] = state == Qt.CheckState.Checked.value
        self._schedule_render()

    def _on_channel_range(self, ch: dict, key: str, value: float, spin: QDoubleSpinBox):
        if value <= spin.minimum() + 1e-6:
            ch[key] = None
        else:
            ch[key] = value
        self._schedule_render()

    def _on_pan(self, delta_row: float, delta_col: float):
        self._fov_center_y = max(0, self._fov_center_y + int(delta_row))
        self._fov_center_x = max(0, self._fov_center_x + int(delta_col))
        self._schedule_render()

    # ── Render scheduling ─────────────────────────────────────────────────────

    def _schedule_render(self):
        self._debounce.start()  # restart 40 ms timer

    def _trigger_render(self):
        if not self._path or self._pyr is None:
            return
        axes = self._axes
        indices = {}
        if "t" in axes:
            indices["t"] = self._t
        if "z" in axes and self._orientation == "XY":
            indices["z"] = self._z
        if self._orientation == "XZ" and "y" in axes:
            indices["y"] = self._z
        if self._orientation == "YZ" and "x" in axes:
            indices["x"] = self._z

        level_idx = max(0, min(self._level_idx, len(self._level_paths) - 1))
        params = {
            "path": self._path,
            "level_idx": level_idx,
            "orientation": self._orientation,
            "indices": indices,
            "fov_center": (self._fov_center_y, self._fov_center_x),
            "fov_size": self._fov_size,
            "channels": [dict(ch) for ch in self._channels],
        }
        self._render_thread.request(params)

    # ── Render result handling ────────────────────────────────────────────────

    def _on_frame_ready(self, rgb: np.ndarray, elapsed_ms: float):
        aspect = self._compute_pixel_aspect()
        self._image_widget.set_frame(rgb, aspect, elapsed_ms)

    def _on_render_error(self, msg: str):
        self._status_label.setText(f"Render error:\n{msg[:200]}")

    def _compute_pixel_aspect(self) -> float:
        """Physical height/width ratio for the current orientation and zoom level."""
        if not self._pixel_sizes:
            return 1.0
        size_map = {s["axis"]: s["value"] for s in self._pixel_sizes}
        ori = self._orientation
        row_ax = "y" if ori == "XY" else "z"
        col_ax = "x" if ori in ("XY", "XZ") else "y"
        row_sz = size_map.get(row_ax, 1.0)
        col_sz = size_map.get(col_ax, 1.0)
        if col_sz <= 0:
            return 1.0
        ratio = row_sz / col_sz
        return ratio if 0.01 < ratio < 100 else 1.0

    def closeEvent(self, _event):
        self._render_thread.stop()
        self._render_thread.wait(2000)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if not _PYRAMID_OK:
        print("ERROR: eubi_bridge not found. Install the package first.", file=sys.stderr)
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # dark palette
    from PyQt6.QtGui import QPalette
    palette = QPalette()
    dark = QColor(45, 45, 48)
    darker = QColor(30, 30, 30)
    palette.setColor(QPalette.ColorRole.Window, dark)
    palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Base, darker)
    palette.setColor(QPalette.ColorRole.AlternateBase, dark)
    palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Button, dark)
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 120, 215))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    initial_path = sys.argv[1] if len(sys.argv) > 1 else ""
    window = ViewerWindow(initial_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
