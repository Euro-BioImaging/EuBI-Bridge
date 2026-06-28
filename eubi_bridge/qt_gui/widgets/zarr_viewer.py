"""
ZarrViewer — reusable OME-Zarr navigation + rendering widget.

Encapsulates everything that InspectPage and ProcessPage share:
  - pyramid loading + channel reading
  - navigation UI (zoom slider, orientation buttons, fit, T/Z sliders, FOV combo)
  - ChannelPanel integration
  - RenderWorker + debounce / pan-rate-limit timers
  - Pan coarse-preview escalation (InspectPage's "better behaviour")
  - MinMaxWorker integration (opt-in, enabled=True by default)

Pages that embed ZarrViewer connect to:
    frame_ready(rgb_array, elapsed_ms, generation)   — handle compositing / status themselves
    dataset_loaded(pyramid)                           — populate page-specific metadata
    status_changed(str)                              — forwarded to main-window status bar

Public accessors let ProcessPage's overlay compositing read viewer state without
reaching into private attributes.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from eubi_bridge.qt_gui.widgets.channel_panel import ChannelPanel
from eubi_bridge.qt_gui.widgets.image_widget import ImageWidget
from eubi_bridge.qt_gui.workers.minmax_worker import MinMaxWorker
from eubi_bridge.qt_gui.workers.render_worker import RenderWorker

_SERVER_DIR = os.path.join(os.path.dirname(__file__), "..", "server")
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

from zarr_plane_server import pyramid_cache  # type: ignore

# ── Shared constants ──────────────────────────────────────────────────────────

_ORI = {
    "XY": ("y", "x", "z", "Z"),
    "XZ": ("z", "x", "y", "Y"),
    "YZ": ("y", "z", "x", "X"),
}

FOV_SIZES = [128, 256, 512, 1024]

_DEFAULT_COLORS = [
    "#FFFFFF", "#FF0000", "#00FF00", "#0000FF",
    "#FFFF00", "#FF00FF", "#00FFFF", "#FF8000",
]


# ── ZarrViewer ────────────────────────────────────────────────────────────────

class ZarrViewer(QWidget):
    """Self-contained OME-Zarr viewer widget.

    Layout (horizontal splitter):
      Left  : scrollable navigation panel + ChannelPanel
      Right : ImageWidget

    Constructor args:
        enable_minmax (bool): wire up MinMaxWorker for Auto histogram buttons.
                              Default True.  Set False if the host page does not
                              need per-channel auto-ranging.
        show_fov_combo (bool): show the FOV-size combo box.  Default True.
        show_reload_btn (bool): show the Reload button.  Default True.
    """

    dataset_loaded = pyqtSignal(object)          # emits the Pyramid object
    frame_ready    = pyqtSignal(object, float, int)  # (rgb_ndarray, elapsed_ms, generation)
    status_changed = pyqtSignal(str)

    def __init__(
        self,
        parent=None,
        *,
        enable_minmax: bool = True,
        show_fov_combo: bool = True,
        show_reload_btn: bool = True,
    ):
        super().__init__(parent)

        self._enable_minmax   = enable_minmax
        self._show_fov_combo  = show_fov_combo
        self._show_reload_btn = show_reload_btn

        # ── Dataset state ─────────────────────────────────────────────────────
        self._path          = ""
        self._pyr           = None
        self._axes          = ""
        self._shape: list   = []
        self._level_paths: list[str] = []
        self._channels: list[dict]   = []

        # ── Navigation state ──────────────────────────────────────────────────
        self._level_idx    = 0
        self._orientation  = "XY"
        self._t            = 0
        self._z            = 0
        self._fov_size     = 512
        self._fov_center_y = 0.0
        self._fov_center_x = 0.0

        # ── Render worker ─────────────────────────────────────────────────────
        self._render_gen = 0
        self._render_worker = RenderWorker(self)
        self._render_worker.frame_ready.connect(self._on_frame_ready_internal)
        self._render_worker.render_error.connect(self._on_render_error)
        self._render_worker.start()

        # Debounce: coalesces rapid control changes (e.g. slider drags) into a
        # single render at most once every 40 ms.
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(40)
        self._debounce.timeout.connect(self._trigger_render)

        # Rate-limiter for pan: fires once per event-loop cycle so _render_gen
        # never races ahead of actual renders (critical for multi-channel images).
        self._pan_render_timer = QTimer(self)
        self._pan_render_timer.setSingleShot(True)
        self._pan_render_timer.setInterval(0)
        self._pan_render_timer.timeout.connect(self._trigger_render)

        self._is_panning = False

        # ── Pan coarse-preview cache ──────────────────────────────────────────
        # Key: (level_idx, fov_size, orientation) → coarse_level_idx to use.
        self._pan_cache: dict[tuple, int] = {}
        self._pan_coarse_idx: int | None = None
        self._pan_slow_streak: int = 0
        self._pan_start_gen: int   = -1

        # ── Last render params (for overlay compositing in ProcessPage) ───────
        self._last_render_params: dict = {}

        # ── MinMax workers ────────────────────────────────────────────────────
        self._minmax_workers: dict[int, MinMaxWorker] = {}

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(splitter)

        # Left: scrollable navigation + channel panel
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(220)
        scroll.setMaximumWidth(340)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(6)

        # Navigation group
        nav_group = QGroupBox("Navigation")
        nav_layout = QVBoxLayout(nav_group)

        # Zoom level
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("Zoom level:"))
        self._zoom_label = QLabel("0 / 0")
        self._zoom_label.setStyleSheet("font-size: 10px;")
        zoom_row.addWidget(self._zoom_label)
        nav_layout.addLayout(zoom_row)
        self._zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self._zoom_slider.setMinimum(0)
        self._zoom_slider.setMaximum(0)
        self._zoom_slider.valueChanged.connect(self._on_zoom_changed)
        nav_layout.addWidget(self._zoom_slider)

        # FOV size (optional)
        if self._show_fov_combo:
            fov_row = QHBoxLayout()
            fov_row.addWidget(QLabel("FOV size:"))
            self._fov_combo = QComboBox()
            for s in FOV_SIZES:
                self._fov_combo.addItem(str(s), s)
            self._fov_combo.setCurrentIndex(2)  # 512
            self._fov_combo.currentIndexChanged.connect(self._on_fov_changed)
            fov_row.addWidget(self._fov_combo)
            nav_layout.addLayout(fov_row)
        else:
            self._fov_combo = None  # type: ignore[assignment]

        # Orientation buttons
        ori_row = QHBoxLayout()
        for ori in ("XY", "XZ", "YZ"):
            btn = QPushButton(ori)
            btn.setCheckable(True)
            btn.setFixedWidth(40)
            btn.clicked.connect(lambda checked, o=ori: self._on_orientation(o))
            ori_row.addWidget(btn)
            setattr(self, f"_ori_btn_{ori}", btn)
        self._ori_btn_XY.setChecked(True)
        ori_row.addStretch()
        nav_layout.addLayout(ori_row)

        # Fit to view
        fit_btn = QPushButton("Fit to view")
        fit_btn.clicked.connect(self._on_fit)
        nav_layout.addWidget(fit_btn)

        # Reload (optional)
        if self._show_reload_btn:
            self._reload_btn = QPushButton("Reload")
            self._reload_btn.setToolTip("Reload dataset and reset pan-preview cache")
            self._reload_btn.setEnabled(False)
            self._reload_btn.clicked.connect(self._on_reload)
            nav_layout.addWidget(self._reload_btn)
        else:
            self._reload_btn = None  # type: ignore[assignment]

        # T slider
        t_row = QHBoxLayout()
        self._t_axis_label = QLabel("T:")
        t_row.addWidget(self._t_axis_label)
        self._t_label = QLabel("0")
        self._t_label.setFixedWidth(35)
        t_row.addWidget(self._t_label)
        nav_layout.addLayout(t_row)
        self._t_slider = QSlider(Qt.Orientation.Horizontal)
        self._t_slider.setMinimum(0)
        self._t_slider.setMaximum(0)
        self._t_slider.valueChanged.connect(self._on_t_changed)
        nav_layout.addWidget(self._t_slider)

        # Z / depth slider
        z_row = QHBoxLayout()
        self._z_axis_label = QLabel("Z:")
        z_row.addWidget(self._z_axis_label)
        self._z_label = QLabel("0")
        self._z_label.setFixedWidth(35)
        z_row.addWidget(self._z_label)
        nav_layout.addLayout(z_row)
        self._z_slider = QSlider(Qt.Orientation.Horizontal)
        self._z_slider.setMinimum(0)
        self._z_slider.setMaximum(0)
        self._z_slider.valueChanged.connect(self._on_z_changed)
        nav_layout.addWidget(self._z_slider)

        content_layout.addWidget(nav_group)

        # Channel panel
        self._channel_panel = ChannelPanel()
        self._channel_panel.channels_changed.connect(self._on_channels_changed)
        if self._enable_minmax:
            self._channel_panel.auto_requested.connect(self._on_auto_requested)
        self._channel_panel.setMinimumHeight(120)
        content_layout.addWidget(self._channel_panel)

        content_layout.addStretch()
        scroll.setWidget(content)
        splitter.addWidget(scroll)

        # Right: ImageWidget
        self._image_widget = ImageWidget()
        self._image_widget.pan_changed.connect(self._on_pan)
        self._image_widget.pan_released.connect(self._on_pan_released)
        self._image_widget.wheel_scrolled.connect(self._on_wheel)
        splitter.addWidget(self._image_widget)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    # ── Public accessors ──────────────────────────────────────────────────────

    @property
    def image_widget(self) -> ImageWidget:
        return self._image_widget

    @property
    def channel_panel(self) -> ChannelPanel:
        return self._channel_panel

    @property
    def axes(self) -> str:
        return self._axes

    @property
    def shape(self) -> list:
        return self._shape

    @property
    def level_paths(self) -> list:
        return self._level_paths

    @property
    def channels(self) -> list:
        return self._channels

    @property
    def level_idx(self) -> int:
        return self._level_idx

    @property
    def orientation(self) -> str:
        return self._orientation

    @property
    def t(self) -> int:
        return self._t

    @property
    def z(self) -> int:
        return self._z

    @property
    def fov_size(self) -> int:
        return self._fov_size

    @property
    def fov_center(self) -> tuple:
        return (self._fov_center_y, self._fov_center_x)

    @property
    def pyr(self):
        return self._pyr

    @property
    def path(self) -> str:
        return self._path

    def current_level_scales(self) -> tuple[float, float]:
        """(v_scale, h_scale): world-pixels per current-level-pixel for each spatial axis."""
        return self._current_level_scales()

    def get_last_render_params(self) -> dict:
        """Return the parameter dict last passed to RenderWorker."""
        return dict(self._last_render_params)

    # ── Dataset loading ───────────────────────────────────────────────────────

    def load_dataset(self, zarr_path: str):
        """Load an OME-Zarr pyramid, initialise sliders, emit dataset_loaded."""
        try:
            pyr = pyramid_cache.get(zarr_path)
        except Exception as exc:
            self.status_changed.emit(f"Error loading: {exc}")
            return

        self._path        = zarr_path
        self._pyr         = pyr
        self._axes        = pyr.axes.lower()
        self._shape       = list(pyr.shape)
        self._level_paths = list(pyr.meta.resolution_paths)
        self._channels    = self._read_channels(pyr)

        # New dataset — clear pan-preview cache so performance is re-profiled
        # for this dataset's chunk layout and channel count.
        self._pan_cache.clear()

        self._init_sliders()
        self._channel_panel.set_channels(self._channels)

        if self._reload_btn is not None:
            self._reload_btn.setEnabled(True)

        self.status_changed.emit(zarr_path)
        self.dataset_loaded.emit(pyr)
        self._schedule_render()

    def _dim(self, ax: str) -> int:
        idx = self._axes.find(ax)
        return self._shape[idx] if idx >= 0 else 1

    def _read_channels(self, pyr) -> list[dict]:
        axes  = pyr.axes.lower()
        c_idx = axes.find("c")
        n_ch  = pyr.shape[c_idx] if c_idx >= 0 else 1
        defaults = [
            {
                "index":        i,
                "label":        f"Ch {i}",
                "color":        _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)],
                "visible":      True,
                "intensityMin": None,
                "intensityMax": None,
            }
            for i in range(n_ch)
        ]
        try:
            omero = pyr.meta.metadata.get("omero")
            if omero:
                for i, ch in enumerate(omero.get("channels", [])):
                    if i >= len(defaults):
                        break
                    win   = ch.get("window", {})
                    color = ch.get("color", "FFFFFF")
                    defaults[i].update({
                        "label":        ch.get("label", defaults[i]["label"]),
                        "color":        f"#{color}" if not color.startswith("#") else color,
                        "visible":      ch.get("active", True),
                        "intensityMin": win.get("start"),
                        "intensityMax": win.get("end"),
                    })
        except Exception:
            pass
        return defaults

    def _init_sliders(self):
        """Reset all navigation controls to the start position for a new dataset."""
        # Always reset to XY on load
        self._orientation = "XY"
        for o in ("XY", "XZ", "YZ"):
            getattr(self, f"_ori_btn_{o}").setChecked(o == "XY")
        self._z_axis_label.setText("Z:")
        v_ax, h_ax, through_ax, _ = _ORI["XY"]
        self._image_widget.set_axes(h_ax, v_ax, through_ax)

        n_levels = len(self._level_paths)
        self._zoom_slider.blockSignals(True)
        self._zoom_slider.setMaximum(max(0, n_levels - 1))
        self._zoom_slider.setValue(0)
        self._zoom_slider.blockSignals(False)
        self._zoom_label.setText(f"0 / {max(0, n_levels - 1)}")
        self._level_idx = 0

        t_max = max(0, self._dim("t") - 1)
        self._t_slider.blockSignals(True)
        self._t_slider.setMaximum(t_max)
        self._t_slider.setValue(0)
        self._t_slider.blockSignals(False)
        self._t_label.setText("0")
        self._t = 0
        self._t_axis_label.setVisible(t_max > 0)
        self._t_label.setVisible(t_max > 0)
        self._t_slider.setVisible(t_max > 0)

        depth_max = max(0, self._dim(through_ax) - 1)
        self._z_slider.blockSignals(True)
        self._z_slider.setMaximum(depth_max)
        mid = depth_max // 2
        self._z_slider.setValue(mid)
        self._z_slider.blockSignals(False)
        self._z_label.setText(str(mid))
        self._z = mid
        self._z_axis_label.setVisible(depth_max > 0)
        self._z_label.setVisible(depth_max > 0)
        self._z_slider.setVisible(depth_max > 0)

        self._fov_center_y = self._dim(v_ax) / 2.0
        self._fov_center_x = self._dim(h_ax) / 2.0

    # ── Navigation callbacks ──────────────────────────────────────────────────

    def _on_zoom_changed(self, value: int):
        self._level_idx = value
        n = len(self._level_paths)
        self._zoom_label.setText(f"{value} / {max(0, n - 1)}")
        self._schedule_render()

    def _on_fov_changed(self, _idx: int):
        if self._fov_combo is not None:
            self._fov_size = self._fov_combo.currentData()
        self._schedule_render()

    def _on_orientation(self, ori: str):
        self._orientation = ori
        for o in ("XY", "XZ", "YZ"):
            getattr(self, f"_ori_btn_{o}").setChecked(o == ori)

        v_ax, h_ax, through_ax, through_lbl = _ORI[ori]
        self._z_axis_label.setText(f"{through_lbl}:")
        self._image_widget.set_axes(h_ax, v_ax, through_ax)

        ax_max = max(0, self._dim(through_ax) - 1)
        self._z_slider.blockSignals(True)
        self._z_slider.setMaximum(ax_max)
        mid = ax_max // 2
        self._z_slider.setValue(mid)
        self._z_slider.blockSignals(False)
        self._z = mid
        self._z_label.setText(str(mid))

        self._fov_center_y = self._dim(v_ax) / 2.0
        self._fov_center_x = self._dim(h_ax) / 2.0
        self._schedule_render()

    def _on_fit(self):
        v_ax, h_ax, _, _ = _ORI[self._orientation]
        self._fov_center_y = self._dim(v_ax) / 2.0
        self._fov_center_x = self._dim(h_ax) / 2.0
        self._schedule_render()

    def _on_reload(self):
        """Evict pyramid cache and reload the current dataset."""
        if self._path:
            self._pan_cache.clear()
            with pyramid_cache._lock:
                pyramid_cache._cache.pop(self._path, None)
            self.load_dataset(self._path)

    def _on_t_changed(self, value: int):
        self._t = value
        self._t_label.setText(str(value))
        self._schedule_render()

    def _on_z_changed(self, value: int):
        self._z = value
        self._z_label.setText(str(value))
        self._schedule_render()

    def _on_channels_changed(self, channels: list[dict]):
        self._channels = channels
        self._schedule_render()

    # ── MinMax worker ─────────────────────────────────────────────────────────

    def _on_auto_requested(self, channel_idx: int):
        if not self._path or not self._level_paths:
            return
        indices: dict = {}
        if "t" in self._axes:
            indices["t"] = self._t
        if self._orientation == "XY" and "z" in self._axes:
            indices["z"] = self._z
        elif self._orientation == "XZ" and "y" in self._axes:
            indices["y"] = self._z
        elif self._orientation == "YZ" and "x" in self._axes:
            indices["x"] = self._z
        level_idx  = max(0, min(self._level_idx, len(self._level_paths) - 1))
        level_path = self._level_paths[level_idx]
        fov_center = (self._fov_center_y, self._fov_center_x)

        if channel_idx in self._minmax_workers:
            self._minmax_workers[channel_idx].quit()
        worker = MinMaxWorker(
            self._path, channel_idx,
            level_path, self._orientation, indices,
            fov_center, self._fov_size,
            self,
        )
        worker.result.connect(self._on_minmax_result)
        worker.failed.connect(
            lambda idx, msg: self.status_changed.emit(f"Auto ch{idx}: {msg}")
        )
        self._minmax_workers[channel_idx] = worker
        worker.start()

    def _on_minmax_result(self, channel_idx: int, vmin: float, vmax: float):
        self._channel_panel.set_channel_range(channel_idx, vmin, vmax)
        self._minmax_workers.pop(channel_idx, None)

    # ── Pan handling ──────────────────────────────────────────────────────────

    def _on_pan(self, delta_row: float, delta_col: float):
        if not self._is_panning:
            self._pan_start_gen   = self._render_gen
            self._pan_slow_streak = 0
            key = (self._level_idx, self._fov_size, self._orientation)
            self._pan_coarse_idx  = self._pan_cache.get(key, None)
        self._is_panning = True
        v_ax, h_ax, _, _ = _ORI[self._orientation]
        v_scale, h_scale = self._current_level_scales()
        step_y = delta_row * v_scale
        step_x = delta_col * h_scale
        half_y = (self._fov_size / 2.0) * v_scale
        half_x = (self._fov_size / 2.0) * h_scale
        y_lo, y_hi = half_y, max(half_y, float(self._dim(v_ax)) - half_y)
        x_lo, x_hi = half_x, max(half_x, float(self._dim(h_ax)) - half_x)
        self._fov_center_y = max(y_lo, min(y_hi, self._fov_center_y + step_y))
        self._fov_center_x = max(x_lo, min(x_hi, self._fov_center_x + step_x))
        if not self._pan_render_timer.isActive():
            self._pan_render_timer.start(0)

    def _on_pan_released(self):
        self._pan_render_timer.stop()
        self._is_panning = False
        self._trigger_render()

    def _on_wheel(self, delta: int):
        new_z = max(0, min(self._z_slider.maximum(), self._z + delta))
        if new_z != self._z:
            self._z_slider.setValue(new_z)

    # ── Render scheduling ─────────────────────────────────────────────────────

    def _schedule_render(self):
        self._debounce.start()

    def _trigger_render(self):
        if not self._path or not self._level_paths:
            return
        indices: dict = {}
        if "t" in self._axes:
            indices["t"] = self._t
        if self._orientation == "XY" and "z" in self._axes:
            indices["z"] = self._z
        elif self._orientation == "XZ" and "y" in self._axes:
            indices["y"] = self._z
        elif self._orientation == "YZ" and "x" in self._axes:
            indices["x"] = self._z

        level_idx       = max(0, min(self._level_idx, len(self._level_paths) - 1))
        fov_size        = self._fov_size
        target_fov_size = 0  # 0 = no upsampling needed

        # ── Coarse-level pan preview ──────────────────────────────────────────
        if self._is_panning and self._pan_coarse_idx is not None:
            coarse = self._pan_coarse_idx
            try:
                from zarr_plane_server import get_orientation_axes
                cur = get_orientation_axes(
                    self._pyr, self._level_paths[level_idx], self._orientation)
                prv = get_orientation_axes(
                    self._pyr, self._level_paths[coarse], self._orientation)
                ratio           = cur['v_scale'] / prv['v_scale']
                coarse_fov_exact = int(fov_size * ratio)
                coarse_fov       = max(64, coarse_fov_exact)
                # Safety: if coarse_fov would cover the entire layer or the
                # exact ratio is too small, fall back to the selected level.
                if (coarse_fov >= prv['layer_height'] or coarse_fov >= prv['layer_width']
                        or coarse_fov_exact < 64):
                    key = (self._level_idx, self._fov_size, self._orientation)
                    self._pan_cache.pop(key, None)
                    self._pan_coarse_idx = None
                else:
                    target_fov_size = fov_size
                    fov_size        = coarse_fov
                    level_idx       = coarse
            except Exception:
                target_fov_size = 0

        v_ax, h_ax, _, _ = _ORI[self._orientation]
        v_scale, h_scale = self._current_level_scales()
        half_y   = int(self._fov_size / 2 * v_scale)
        half_x   = int(self._fov_size / 2 * h_scale)
        center_y = max(half_y, min(self._dim(v_ax) - half_y, round(self._fov_center_y)))
        center_x = max(half_x, min(self._dim(h_ax) - half_x, round(self._fov_center_x)))

        params = {
            "path":            self._path,
            "level_idx":       level_idx,
            "orientation":     self._orientation,
            "indices":         indices,
            "fov_center":      (center_y, center_x),
            "fov_size":        fov_size,
            "target_fov_size": target_fov_size,
            "channels":        [dict(ch) for ch in self._channels],
        }
        self._last_render_params = params
        self._render_worker.request(params)
        self._render_gen = self._render_worker._generation

    # ── Internal frame-ready handler (pan escalation logic) ───────────────────

    def _on_frame_ready_internal(self, rgb: np.ndarray, elapsed_ms: float, generation: int):
        """Runs the pan coarse-preview escalation logic then re-emits frame_ready."""
        # Discard superseded frames
        if generation < self._render_gen:
            return
        # During a drag, discard frames that predate this drag
        if self._is_panning and generation <= self._pan_start_gen:
            return

        if self._is_panning:
            key     = (self._level_idx, self._fov_size, self._orientation)
            max_lvl = len(self._level_paths) - 1
            sel     = max(0, min(self._level_idx, max_lvl))

            if elapsed_ms > 8.0:
                self._pan_slow_streak += 1
                if self._pan_slow_streak >= 2:
                    self._pan_slow_streak = 0
                    current  = self._pan_coarse_idx if self._pan_coarse_idx is not None else sel
                    proposed = current + 1
                    if proposed <= max_lvl and self._pyr:
                        should_escalate = True
                        try:
                            from zarr_plane_server import get_orientation_axes
                            sel_info  = get_orientation_axes(
                                self._pyr, self._level_paths[sel], self._orientation)
                            prop_info = get_orientation_axes(
                                self._pyr, self._level_paths[proposed], self._orientation)
                            ratio            = sel_info['v_scale'] / prop_info['v_scale']
                            coarse_fov_exact = int(self._fov_size * ratio)
                            coarse_fov       = max(64, coarse_fov_exact)
                            if (prop_info['layer_height'] <= coarse_fov
                                    or prop_info['layer_width'] <= coarse_fov
                                    or coarse_fov_exact < 64):
                                should_escalate = False
                        except Exception:
                            pass
                        if should_escalate:
                            self._pan_coarse_idx = proposed
                            self._pan_cache[key] = proposed
                            if not self._pan_render_timer.isActive():
                                self._pan_render_timer.start(0)
            else:
                self._pan_slow_streak = 0

        # Settle-render: step cache one level finer if it completed fast
        if not self._is_panning and elapsed_ms < 8.0:
            key    = (self._level_idx, self._fov_size, self._orientation)
            cached = self._pan_cache.get(key)
            if cached is not None and cached > self._level_idx:
                self._pan_cache[key] = cached - 1
                if self._pan_cache[key] <= self._level_idx:
                    self._pan_cache.pop(key, None)

        self.frame_ready.emit(rgb, elapsed_ms, generation)

    def _on_render_error(self, msg: str):
        self.status_changed.emit(f"Render error: {msg[:120]}")

    # ── Scale helper ──────────────────────────────────────────────────────────

    def _current_level_scales(self) -> tuple[float, float]:
        """(v_scale, h_scale) for the currently selected pyramid level."""
        if not self._pyr or not self._level_paths:
            return 1.0, 1.0
        level_path = self._level_paths[min(self._level_idx, len(self._level_paths) - 1)]
        try:
            from zarr_plane_server import get_orientation_axes
            info = get_orientation_axes(self._pyr, level_path, self._orientation)
            return info['v_scale'], info['h_scale']
        except Exception:
            return 1.0, 1.0

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, _event):
        self._render_worker.stop()
        self._render_worker.wait(2000)
