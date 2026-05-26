"""
Process page — Crop and Annotate sub-tabs sharing a single viewer.

Layout:
  Left  : SidebarBrowser (zarr mode)
  Right : QVBoxLayout
            Viewer panel (shared, always visible)
            QTabWidget
              "Crop"     tab — rubber-band ROI → crop_ome_zarr
              "Annotate" tab — brush painting + RF classifier → NGFF labels
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import Dict, List, Optional, Tuple

import dask.array as da
import numpy as np
import zarr
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from eubi_annotate.core.annotation import AnnotationStore, train_and_classify
from eubi_annotate.core.label_writer import write_ngff_labels
from eubi_bridge.qt_gui.widgets.channel_panel import ChannelPanel
from eubi_bridge.qt_gui.widgets.image_widget import ImageWidget
from eubi_bridge.qt_gui.widgets.log_widget import LogWidget
from eubi_bridge.qt_gui.widgets.sidebar_browser import SidebarBrowser
from eubi_annotate.qt_gui.workers.classifier_worker import ClassifierWorker
from eubi_annotate.qt_gui.workers.crop_worker import CropWorker
from eubi_bridge.qt_gui.workers.render_worker import RenderWorker

_SERVER_DIR = os.path.join(os.path.dirname(__file__), "..", "server")
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

from zarr_plane_server import pyramid_cache  # type: ignore

# ── Optional PyTorch / DINOv2 availability ────────────────────────────────────
try:
    import torch as _torch
    _TORCH_AVAILABLE = True
    _CUDA_AVAILABLE  = _torch.cuda.is_available()
except ImportError:
    _TORCH_AVAILABLE = False
    _CUDA_AVAILABLE  = False

# ── Constants ─────────────────────────────────────────────────────────────────

_ORI = {
    "XY": ("y", "x", "z", "Z"),
    "XZ": ("z", "x", "y", "Y"),
    "YZ": ("y", "z", "x", "X"),
}

_DEFAULT_COLORS = [
    "#FFFFFF", "#FF0000", "#00FF00", "#0000FF",
    "#FFFF00", "#FF00FF", "#00FFFF", "#FF8000",
]

# Class annotation palette (r, g, b)
_CLASS_PALETTE: List[Tuple[int, int, int]] = [
    (220, 60, 60),
    (60, 200, 60),
    (60, 100, 220),
    (220, 200, 50),
    (180, 60, 220),
    (50, 200, 220),
    (220, 130, 50),
    (160, 160, 160),
]

FOV_SIZES = [128, 256, 512, 1024]


def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))  # type: ignore


# ── ProcessPage ───────────────────────────────────────────────────────────────

class ProcessPage(QWidget):
    """Process tab: Crop and Annotate sub-tabs with a shared viewer."""

    status_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        # ── Shared viewer state ───────────────────────────────────────────────
        self._path         = ""
        self._pyr          = None
        self._axes         = ""
        self._shape: list  = []
        self._level_paths: List[str] = []
        self._channels: List[dict]   = []

        self._level_idx    = 0
        self._orientation  = "XY"
        self._t            = 0
        self._z            = 0
        self._fov_size     = 512
        self._fov_center_y = 0.0
        self._fov_center_x = 0.0

        self._last_render_params: dict = {}
        self._is_panning = False
        self._render_gen = 0

        # ── Annotation state ──────────────────────────────────────────────────
        # _label_store is the single merged ground truth: hand strokes + classifier output
        self._label_store: Optional[AnnotationStore] = None
        # annotation box in base-array coords {v_ax, h_ax, v_start, v_end, h_start, h_end}
        self._annotation_box: Optional[dict] = None
        self._classes: List[dict] = []   # {name, color:(r,g,b)}
        self._overlay_opacity = 0.5
        self._brush_size = 5
        self._classifier_worker: Optional[ClassifierWorker] = None
        self._clf_running = False
        self._trained_clf = None          # cached fitted sklearn Pipeline
        self._annotations_dirty = True   # True → must retrain before predicting
        self._feature_config: Optional[dict] = None  # feature params used when training
        self._save_worker: Optional[QTimer] = None  # reused as thread sentinel

        # ── Workers ───────────────────────────────────────────────────────────
        self._render_worker = RenderWorker(self)
        self._render_worker.frame_ready.connect(self._on_frame_ready)
        self._render_worker.start()

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(40)
        self._debounce.timeout.connect(self._trigger_render)

        self._pan_render_timer = QTimer(self)
        self._pan_render_timer.setSingleShot(True)
        self._pan_render_timer.setInterval(0)
        self._pan_render_timer.timeout.connect(self._trigger_render)

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        outer_splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(outer_splitter)

        # Left: file browser
        self._browser = SidebarBrowser(mode="zarr")
        self._browser.zarr_selected.connect(self._load_zarr)
        self._browser.setMinimumWidth(180)
        self._browser.setMaximumWidth(300)
        outer_splitter.addWidget(self._browser)

        # Right: viewer + sub-tabs in a vertical splitter
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        outer_splitter.addWidget(right_splitter)
        outer_splitter.setStretchFactor(0, 0)
        outer_splitter.setStretchFactor(1, 1)

        # Viewer panel (top)
        viewer_widget = self._build_viewer_panel()
        right_splitter.addWidget(viewer_widget)

        # Sub-tabs (bottom)
        self._sub_tabs = QTabWidget()
        self._sub_tabs.addTab(self._build_crop_tab(),    "Crop")
        self._sub_tabs.addTab(self._build_annotate_tab(), "Annotate")
        self._sub_tabs.currentChanged.connect(self._on_subtab_changed)
        right_splitter.addWidget(self._sub_tabs)

        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 1)

    # ── Viewer panel ──────────────────────────────────────────────────────────

    def _build_viewer_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 2)
        layout.setSpacing(4)

        # Controls row 1: orientation + zoom + fit
        row1 = QHBoxLayout()

        ori_group = QButtonGroup(self)
        ori_group.setExclusive(True)
        for ori in ("XY", "XZ", "YZ"):
            btn = QPushButton(ori)
            btn.setCheckable(True)
            btn.setFixedWidth(40)
            btn.clicked.connect(lambda _, o=ori: self._on_orientation(o))
            ori_group.addButton(btn)
            row1.addWidget(btn)
            setattr(self, f"_ori_btn_{ori}", btn)
        self._ori_btn_XY.setChecked(True)

        row1.addWidget(QLabel("  Zoom:"))
        self._zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self._zoom_slider.setMinimum(0)
        self._zoom_slider.setMaximum(0)
        self._zoom_slider.setFixedWidth(80)
        self._zoom_slider.valueChanged.connect(self._on_zoom_changed)
        row1.addWidget(self._zoom_slider)
        self._zoom_label = QLabel("0/0")
        self._zoom_label.setStyleSheet("font-size: 10px;")
        row1.addWidget(self._zoom_label)

        fit_btn = QPushButton("Fit")
        fit_btn.setFixedWidth(36)
        fit_btn.clicked.connect(self._on_fit)
        row1.addWidget(fit_btn)
        row1.addStretch()
        layout.addLayout(row1)

        # Controls row 2: T and Z sliders
        row2 = QHBoxLayout()
        self._t_axis_label = QLabel("T:")
        row2.addWidget(self._t_axis_label)
        self._t_label = QLabel("0")
        self._t_label.setFixedWidth(30)
        row2.addWidget(self._t_label)
        self._t_slider = QSlider(Qt.Orientation.Horizontal)
        self._t_slider.setMinimum(0)
        self._t_slider.setMaximum(0)
        self._t_slider.valueChanged.connect(self._on_t_changed)
        row2.addWidget(self._t_slider)

        row2.addSpacing(12)
        self._z_axis_label = QLabel("Z:")
        row2.addWidget(self._z_axis_label)
        self._z_label = QLabel("0")
        self._z_label.setFixedWidth(30)
        row2.addWidget(self._z_label)
        self._z_slider = QSlider(Qt.Orientation.Horizontal)
        self._z_slider.setMinimum(0)
        self._z_slider.setMaximum(0)
        self._z_slider.valueChanged.connect(self._on_z_changed)
        row2.addWidget(self._z_slider)
        layout.addLayout(row2)

        # Controls row 3: annotation overlay toggles
        row3 = QHBoxLayout()
        self._show_ann_cb = QCheckBox("Show annotations")
        self._show_ann_cb.setChecked(True)
        self._show_ann_cb.toggled.connect(self._on_overlay_toggle)
        self._show_clf_cb = QCheckBox("Show classification")
        self._show_clf_cb.setChecked(True)
        self._show_clf_cb.toggled.connect(self._on_overlay_toggle)
        row3.addWidget(self._show_ann_cb)
        row3.addWidget(self._show_clf_cb)
        row3.addStretch()
        layout.addLayout(row3)

        # Viewer + channel panel in a horizontal splitter
        viewer_hsplit = QSplitter(Qt.Orientation.Horizontal)

        self._image_widget = ImageWidget()
        self._image_widget.pan_changed.connect(self._on_pan)
        self._image_widget.pan_released.connect(self._on_pan_released)
        self._image_widget.wheel_scrolled.connect(self._on_wheel)
        self._image_widget.roi_selected.connect(self._on_roi_selected)
        self._image_widget.box_selected.connect(self._on_box_selected)
        self._image_widget.annotate_at.connect(self._on_annotate_at)
        self._image_widget.erase_at.connect(self._on_erase_at)
        viewer_hsplit.addWidget(self._image_widget)

        self._channel_panel = ChannelPanel()
        self._channel_panel.channels_changed.connect(self._on_channels_changed)
        self._channel_panel.setMinimumWidth(160)
        self._channel_panel.setMaximumWidth(240)
        viewer_hsplit.addWidget(self._channel_panel)

        viewer_hsplit.setStretchFactor(0, 1)
        viewer_hsplit.setStretchFactor(1, 0)
        layout.addWidget(viewer_hsplit)

        self._viewer_status = QLabel("")
        self._viewer_status.setStyleSheet("font-size: 10px; color: #aaa;")
        layout.addWidget(self._viewer_status)

        return panel

    # ── Crop sub-tab ──────────────────────────────────────────────────────────

    def _build_crop_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Mode row
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._crop_pan_btn = QPushButton("Pan")
        self._crop_pan_btn.setCheckable(True)
        self._crop_pan_btn.setChecked(True)
        self._crop_draw_btn = QPushButton("Draw Crop Box")
        self._crop_draw_btn.setCheckable(True)
        grp = QButtonGroup(self)
        grp.setExclusive(True)
        grp.addButton(self._crop_pan_btn)
        grp.addButton(self._crop_draw_btn)
        self._crop_pan_btn.clicked.connect(lambda: self._image_widget.set_mode('pan'))
        self._crop_draw_btn.clicked.connect(lambda: self._image_widget.set_mode('crop'))
        mode_row.addWidget(self._crop_pan_btn)
        mode_row.addWidget(self._crop_draw_btn)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # Scrollable params
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setSpacing(6)

        # Crop region spin boxes
        region_group = QGroupBox("Crop Region")
        rl = QVBoxLayout(region_group)
        self._crop_spinboxes: Dict[str, QSpinBox] = {}
        self._crop_axis_rows: Dict[str, QWidget] = {}
        for ax in ("y", "x", "z", "t", "c"):
            row_w = QWidget()
            row = QHBoxLayout(row_w)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(QLabel(f"{ax.upper()}:"))
            sb_start = QSpinBox()
            sb_start.setMinimum(0)
            sb_start.setMaximum(99999)
            sb_stop = QSpinBox()
            sb_stop.setMinimum(0)
            sb_stop.setMaximum(99999)
            row.addWidget(QLabel("start"))
            row.addWidget(sb_start)
            row.addWidget(QLabel("stop"))
            row.addWidget(sb_stop)
            row.addStretch()
            self._crop_spinboxes[f"{ax}_start"] = sb_start
            self._crop_spinboxes[f"{ax}_stop"]  = sb_stop
            self._crop_axis_rows[ax] = row_w
            rl.addWidget(row_w)
        cl.addWidget(region_group)

        # Output settings
        out_group = QGroupBox("Output")
        ol = QVBoxLayout(out_group)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Path:"))
        self._crop_output_path = QLineEdit()
        self._crop_output_path.setPlaceholderText("output.zarr")
        path_row.addWidget(self._crop_output_path)
        browse_btn = QPushButton("…")
        browse_btn.setFixedWidth(28)
        browse_btn.clicked.connect(self._on_crop_browse)
        path_row.addWidget(browse_btn)
        ol.addLayout(path_row)

        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Zarr format:"))
        self._crop_fmt = QComboBox()
        self._crop_fmt.addItem("v2", 2)
        self._crop_fmt.addItem("v3", 3)
        fmt_row.addWidget(self._crop_fmt)
        fmt_row.addStretch()
        ol.addLayout(fmt_row)

        sf_row = QHBoxLayout()
        sf_row.addWidget(QLabel("Scale factors:"))
        self._crop_scale_factors = QLineEdit("1,1,2,2")
        self._crop_scale_factors.setToolTip(
            "Downsample factors per axis (e.g. 1,1,2,2 for CZYX or 1,2,2 for ZYX)"
        )
        sf_row.addWidget(self._crop_scale_factors)
        ol.addLayout(sf_row)

        lev_row = QHBoxLayout()
        lev_row.addWidget(QLabel("Pyramid levels:"))
        self._crop_n_levels = QSpinBox()
        self._crop_n_levels.setMinimum(1)
        self._crop_n_levels.setMaximum(16)
        self._crop_n_levels.setValue(4)
        lev_row.addWidget(self._crop_n_levels)
        lev_row.addStretch()
        ol.addLayout(lev_row)
        cl.addWidget(out_group)

        scroll.setWidget(content)
        layout.addWidget(scroll)

        # Run button + log
        run_row = QHBoxLayout()
        self._crop_run_btn = QPushButton("Run Crop")
        self._crop_run_btn.setEnabled(False)
        self._crop_run_btn.clicked.connect(self._on_run_crop)
        run_row.addWidget(self._crop_run_btn)
        run_row.addStretch()
        layout.addLayout(run_row)

        self._crop_log = LogWidget()
        self._crop_log.setMaximumHeight(100)
        layout.addWidget(self._crop_log)

        return tab

    # ── Annotate sub-tab ──────────────────────────────────────────────────────

    def _build_annotate_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Mode row
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._ann_pan_btn  = QPushButton("Pan")
        self._ann_pan_btn.setCheckable(True)
        self._ann_pan_btn.setChecked(True)
        self._ann_box_btn  = QPushButton("Draw Box")
        self._ann_box_btn.setCheckable(True)
        self._ann_box_btn.setToolTip("Draw the region to classify (orange box)")
        self._ann_draw_btn = QPushButton("Annotate")
        self._ann_draw_btn.setCheckable(True)
        grp = QButtonGroup(self)
        grp.setExclusive(True)
        grp.addButton(self._ann_pan_btn)
        grp.addButton(self._ann_box_btn)
        grp.addButton(self._ann_draw_btn)
        self._ann_pan_btn.clicked.connect(lambda: self._image_widget.set_mode('pan'))
        self._ann_box_btn.clicked.connect(lambda: self._image_widget.set_mode('box'))
        self._ann_draw_btn.clicked.connect(lambda: self._image_widget.set_mode('annotate'))
        mode_row.addWidget(self._ann_pan_btn)
        mode_row.addWidget(self._ann_box_btn)
        mode_row.addWidget(self._ann_draw_btn)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setSpacing(6)

        # Classes
        cls_group = QGroupBox("Classes")
        cg = QVBoxLayout(cls_group)
        self._class_list = QListWidget()
        self._class_list.setMaximumHeight(120)
        cg.addWidget(self._class_list)
        cls_btn_row = QHBoxLayout()
        add_cls_btn = QPushButton("+ Add Class")
        add_cls_btn.clicked.connect(self._add_class)
        rem_cls_btn = QPushButton("- Remove")
        rem_cls_btn.clicked.connect(self._remove_class)
        cls_btn_row.addWidget(add_cls_btn)
        cls_btn_row.addWidget(rem_cls_btn)
        cls_btn_row.addStretch()
        cg.addLayout(cls_btn_row)
        cl.addWidget(cls_group)

        # Erase row — right-click is the shortcut; button here for discoverability
        erase_row = QHBoxLayout()
        self._erase_btn = QPushButton("Erase  (right-click)")
        self._erase_btn.setToolTip(
            "Right-click and drag in Annotate mode to erase existing annotations"
        )
        self._erase_btn.setCheckable(True)
        self._erase_btn.clicked.connect(self._on_erase_btn_toggled)
        erase_row.addWidget(self._erase_btn)
        erase_row.addStretch()
        cl.addLayout(erase_row)

        # Brush
        brush_group = QGroupBox("Brush")
        bg = QHBoxLayout(brush_group)
        bg.addWidget(QLabel("Size (px):"))
        self._brush_slider = QSlider(Qt.Orientation.Horizontal)
        self._brush_slider.setMinimum(1)
        self._brush_slider.setMaximum(50)
        self._brush_slider.setValue(5)
        self._brush_slider.valueChanged.connect(lambda v: setattr(self, '_brush_size', v))
        bg.addWidget(self._brush_slider)
        self._brush_label = QLabel("5")
        self._brush_label.setFixedWidth(24)
        self._brush_slider.valueChanged.connect(lambda v: self._brush_label.setText(str(v)))
        bg.addWidget(self._brush_label)
        cl.addWidget(brush_group)

        # Clear buttons
        clr_row = QHBoxLayout()
        clr_slice_btn = QPushButton("Clear This Slice")
        clr_slice_btn.clicked.connect(self._on_clear_slice)
        clr_all_btn = QPushButton("Clear All")
        clr_all_btn.clicked.connect(self._on_clear_all)
        clr_row.addWidget(clr_slice_btn)
        clr_row.addWidget(clr_all_btn)
        clr_row.addStretch()
        cl.addLayout(clr_row)

        # Classifier
        clf_group = QGroupBox("Classifier")
        cfg = QVBoxLayout(clf_group)
        cfg.setSpacing(4)

        # ── Gaussian features ─────────────────────────────────────────────────
        cfg.addWidget(QLabel("Gaussian features:"))

        g_type_row = QHBoxLayout()
        g_type_row.addWidget(QLabel("  Types:"))
        self._gauss_type_checks: Dict[str, QCheckBox] = {}
        for name in ('Smooth', 'Gradient', 'Laplacian'):
            cb = QCheckBox(name)
            cb.setChecked(True)
            g_type_row.addWidget(cb)
            self._gauss_type_checks[name.lower()] = cb
        g_type_row.addStretch()
        cfg.addLayout(g_type_row)

        g_sigma_row = QHBoxLayout()
        g_sigma_row.addWidget(QLabel("  Scales (σ):"))
        self._sigma_checks: Dict[float, QCheckBox] = {}
        for s in (1, 1.5, 2, 3, 4):
            cb = QCheckBox(str(s))
            cb.setChecked(True)
            g_sigma_row.addWidget(cb)
            self._sigma_checks[s] = cb
        g_sigma_row.addStretch()
        cfg.addLayout(g_sigma_row)

        # ── Rank / morphological features ─────────────────────────────────────
        cfg.addWidget(QLabel("Rank / morphological features:"))

        w_type_row = QHBoxLayout()
        w_type_row.addWidget(QLabel("  Types:"))
        self._window_type_checks: Dict[str, QCheckBox] = {}
        for name in ('Mean', 'Median', 'Min', 'Max'):
            cb = QCheckBox(name)
            cb.setChecked(True)
            w_type_row.addWidget(cb)
            self._window_type_checks[name.lower()] = cb
        w_type_row.addStretch()
        cfg.addLayout(w_type_row)

        w_size_row = QHBoxLayout()
        w_size_row.addWidget(QLabel("  Window sizes:"))
        self._window_size_checks: Dict[int, QCheckBox] = {}
        for sz in (3, 5, 7, 11, 15):
            cb = QCheckBox(str(sz))
            cb.setChecked(sz <= 7)   # default: 3, 5, 7
            w_size_row.addWidget(cb)
            self._window_size_checks[sz] = cb
        w_size_row.addStretch()
        cfg.addLayout(w_size_row)

        # ── Raw intensity ──────────────────────────────────────────────────────
        self._include_raw_cb = QCheckBox("Raw (unfiltered) intensity")
        self._include_raw_cb.setChecked(True)
        cfg.addWidget(self._include_raw_cb)

        # ── DINOv2 features ────────────────────────────────────────────────────
        cfg.addWidget(QLabel("DINOv2 features (requires PyTorch):"))
        dino_row = QHBoxLayout()
        self._dinov2_cb = QCheckBox("Enable")
        self._dinov2_cb.setChecked(False)
        self._dinov2_cb.setEnabled(_TORCH_AVAILABLE)
        if not _TORCH_AVAILABLE:
            self._dinov2_cb.setToolTip(
                "PyTorch is not installed.\n"
                "CPU-only:   pip install torch\n"
                "CUDA 12.x:  pip install torch --index-url "
                "https://download.pytorch.org/whl/cu121"
            )
        dino_row.addWidget(self._dinov2_cb)

        dino_row.addWidget(QLabel("Model:"))
        self._dinov2_model_combo = QComboBox()
        self._dinov2_model_combo.addItem("ViT-S/14  (384-d, ~86 MB)",  "vits14")
        self._dinov2_model_combo.addItem("ViT-B/14  (768-d, ~330 MB)", "vitb14")
        self._dinov2_model_combo.setEnabled(_TORCH_AVAILABLE)
        dino_row.addWidget(self._dinov2_model_combo)

        dino_row.addWidget(QLabel("Device:"))
        self._dinov2_device_combo = QComboBox()
        self._dinov2_device_combo.addItem("CPU", "cpu")
        if _CUDA_AVAILABLE:
            self._dinov2_device_combo.addItem("CUDA", "cuda")
        self._dinov2_device_combo.setEnabled(_TORCH_AVAILABLE)
        dino_row.addWidget(self._dinov2_device_combo)
        dino_row.addStretch()
        cfg.addLayout(dino_row)

        # ── Classifier type ────────────────────────────────────────────────────
        clf_type_row = QHBoxLayout()
        clf_type_row.addWidget(QLabel("Classifier:"))
        self._clf_type_combo = QComboBox()
        self._clf_type_combo.addItem("Random Forest", "rf")
        self._clf_type_combo.addItem("XGBoost", "xgb")
        clf_type_row.addWidget(self._clf_type_combo)
        clf_type_row.addStretch()
        cfg.addLayout(clf_type_row)

        # ── Run / status / opacity ─────────────────────────────────────────────
        self._clf_run_btn = QPushButton("Run Classifier")
        self._clf_run_btn.setEnabled(False)
        self._clf_run_btn.clicked.connect(self._on_run_classifier)
        cfg.addWidget(self._clf_run_btn)

        self._clf_status = QLabel("")
        self._clf_status.setStyleSheet("font-size: 10px; color: #aaa;")
        cfg.addWidget(self._clf_status)

        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Overlay opacity:"))
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setMinimum(0)
        self._opacity_slider.setMaximum(100)
        self._opacity_slider.setValue(50)
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_row.addWidget(self._opacity_slider)
        cfg.addLayout(opacity_row)
        cl.addWidget(clf_group)

        cl.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

        # Save section — outside scroll area so it's always visible
        save_group = QGroupBox("Save")
        sg = QVBoxLayout(save_group)
        lbl_row = QHBoxLayout()
        lbl_row.addWidget(QLabel("Label name:"))
        self._label_name_edit = QLineEdit("annotations")
        lbl_row.addWidget(self._label_name_edit)
        sg.addLayout(lbl_row)
        save_btn_row = QHBoxLayout()
        self._save_btn = QPushButton("Save to OME-Zarr")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save_annotations)
        save_btn_row.addWidget(self._save_btn)
        save_btn_row.addStretch()
        sg.addLayout(save_btn_row)
        self._save_status = QLabel("")
        self._save_status.setStyleSheet("font-size: 10px; color: #aaa;")
        sg.addWidget(self._save_status)
        layout.addWidget(save_group)

        return tab

    # ── Dataset loading ───────────────────────────────────────────────────────

    def _load_zarr(self, path: str):
        try:
            pyr = pyramid_cache.get(path)
        except Exception as exc:
            self._viewer_status.setText(f"Error: {exc}")
            return

        self._path        = path
        self._pyr         = pyr
        self._axes        = pyr.axes.lower()
        self._shape       = list(pyr.shape)
        self._level_paths = list(pyr.meta.resolution_paths)
        self._channels    = self._read_channels(pyr)

        # Reset annotation state for new dataset
        self._label_store      = AnnotationStore(pyr.shape, pyr.axes.lower())
        self._annotation_box   = None
        self._trained_clf      = None
        self._feature_config   = None
        self._annotations_dirty = True
        self._image_widget.clear_annotation_box()

        self._init_sliders()
        self._init_crop_spinboxes()
        self._channel_panel.set_channels(self._channels)
        self._crop_run_btn.setEnabled(True)
        self._clf_run_btn.setEnabled(True)
        self._save_btn.setEnabled(True)

        self.status_changed.emit(path)
        self._schedule_render()

    def _dim(self, ax: str) -> int:
        idx = self._axes.find(ax)
        return self._shape[idx] if idx >= 0 else 1

    def _read_channels(self, pyr) -> List[dict]:
        axes = pyr.axes.lower()
        c_idx = axes.find("c")
        n_ch  = pyr.shape[c_idx] if c_idx >= 0 else 1
        defaults = [
            {"index": i, "label": f"Ch {i}", "color": _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)],
             "visible": True, "intensityMin": None, "intensityMax": None}
            for i in range(n_ch)
        ]
        try:
            omero = pyr.meta.metadata.get("omero")
            if omero:
                for i, ch in enumerate(omero.get("channels", [])):
                    if i >= len(defaults):
                        break
                    win = ch.get("window", {})
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
        self._orientation = "XY"
        for o in ("XY", "XZ", "YZ"):
            getattr(self, f"_ori_btn_{o}").setChecked(o == "XY")

        v_ax, h_ax, through_ax, _ = _ORI["XY"]
        self._image_widget.set_axes(h_ax, v_ax, through_ax)
        self._z_axis_label.setText("Z:")

        n_levels = len(self._level_paths)
        self._zoom_slider.blockSignals(True)
        self._zoom_slider.setMaximum(max(0, n_levels - 1))
        self._zoom_slider.setValue(0)
        self._zoom_slider.blockSignals(False)
        self._zoom_label.setText(f"0/{max(0, n_levels - 1)}")
        self._level_idx = 0

        t_max = max(0, self._dim("t") - 1)
        self._t_slider.blockSignals(True)
        self._t_slider.setMaximum(t_max)
        self._t_slider.setValue(0)
        self._t_slider.blockSignals(False)
        self._t_label.setText("0")
        self._t = 0
        for w in (self._t_axis_label, self._t_label, self._t_slider):
            w.setVisible(t_max > 0)

        depth_max = max(0, self._dim(through_ax) - 1)
        self._z_slider.blockSignals(True)
        self._z_slider.setMaximum(depth_max)
        mid = depth_max // 2
        self._z_slider.setValue(mid)
        self._z_slider.blockSignals(False)
        self._z_label.setText(str(mid))
        self._z = mid
        for w in (self._z_axis_label, self._z_label, self._z_slider):
            w.setVisible(depth_max > 0)

        self._fov_center_y = self._dim(v_ax) / 2.0
        self._fov_center_x = self._dim(h_ax) / 2.0

    def _init_crop_spinboxes(self):
        axes = self._axes
        for ax, row_w in self._crop_axis_rows.items():
            present = ax in axes
            row_w.setVisible(present)
            if present:
                dim = self._dim(ax)
                self._crop_spinboxes[f"{ax}_start"].setMaximum(max(0, dim - 1))
                self._crop_spinboxes[f"{ax}_start"].setValue(0)
                self._crop_spinboxes[f"{ax}_stop"].setMaximum(dim)
                self._crop_spinboxes[f"{ax}_stop"].setValue(dim)

    # ── Viewer callbacks ──────────────────────────────────────────────────────

    def _on_zoom_changed(self, value: int):
        self._level_idx = value
        n = len(self._level_paths)
        self._zoom_label.setText(f"{value}/{max(0, n-1)}")
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
        self._annotation_box = None
        self._image_widget.clear_annotation_box()
        self._schedule_render()

    def _on_fit(self):
        v_ax, h_ax, _, _ = _ORI[self._orientation]
        self._fov_center_y = self._dim(v_ax) / 2.0
        self._fov_center_x = self._dim(h_ax) / 2.0
        self._schedule_render()

    def _on_t_changed(self, value: int):
        self._t = value
        self._t_label.setText(str(value))
        self._schedule_render()

    def _on_z_changed(self, value: int):
        self._z = value
        self._z_label.setText(str(value))
        self._schedule_render()

    def _stored_classification(self) -> Optional[np.ndarray]:
        """Unused — kept as no-op; overlay comes from _label_store directly."""
        return None

    def _on_pan(self, delta_row: float, delta_col: float):
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

    def _on_channels_changed(self, channels: list):
        self._channels = channels
        self._schedule_render()

    def _on_overlay_toggle(self):
        self._schedule_render()

    def _current_level_scales(self) -> Tuple[float, float]:
        if not self._pyr or not self._level_paths:
            return 1.0, 1.0
        level_path = self._level_paths[min(self._level_idx, len(self._level_paths) - 1)]
        try:
            from zarr_plane_server import get_orientation_axes
            info = get_orientation_axes(self._pyr, level_path, self._orientation)
            return info['v_scale'], info['h_scale']
        except Exception:
            return 1.0, 1.0

    def _on_subtab_changed(self, idx: int):
        # Reset image widget to pan mode when switching tabs
        self._image_widget.set_mode('pan')
        if idx == 0:
            self._crop_pan_btn.setChecked(True)
        else:
            self._ann_pan_btn.setChecked(True)

    # ── Render ────────────────────────────────────────────────────────────────

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

        level_idx  = max(0, min(self._level_idx, len(self._level_paths) - 1))
        v_ax, h_ax, _, _ = _ORI[self._orientation]
        v_scale, h_scale = self._current_level_scales()
        half_y = int(self._fov_size / 2 * v_scale)
        half_x = int(self._fov_size / 2 * h_scale)
        center_y = max(half_y, min(self._dim(v_ax) - half_y, round(self._fov_center_y)))
        center_x = max(half_x, min(self._dim(h_ax) - half_x, round(self._fov_center_x)))

        params = {
            "path":            self._path,
            "level_idx":       level_idx,
            "orientation":     self._orientation,
            "indices":         indices,
            "fov_center":      (center_y, center_x),
            "fov_size":        self._fov_size,
            "target_fov_size": 0,
            "channels":        [dict(ch) for ch in self._channels],
        }
        self._last_render_params = params
        self._render_worker.request(params)
        self._render_gen = self._render_worker._generation

    def _on_frame_ready(self, rgb: np.ndarray, elapsed_ms: float, generation: int):
        if generation < self._render_gen:
            return
        rgb = self._composite_overlays(rgb)
        self._image_widget.set_frame(rgb, 1.0, elapsed_ms)
        self._viewer_status.setText(f"{elapsed_ms:.0f} ms")

    # ── Overlay compositing ───────────────────────────────────────────────────

    def _composite_overlays(self, rgb: np.ndarray) -> np.ndarray:
        """Alpha-composite annotation and classifier masks onto rendered rgb."""
        if self._label_store is None:
            return rgb
        if not self._last_render_params:
            return rgb

        try:
            from zarr_plane_server import compute_fov_region, get_orientation_axes
            from PIL import Image as PIL_Image

            lp = self._level_paths[
                min(self._last_render_params.get("level_idx", 0),
                    len(self._level_paths) - 1)
            ]
            ori = self._last_render_params.get("orientation", "XY")
            info = get_orientation_axes(self._pyr, lp, ori)
            fov_center = self._last_render_params.get("fov_center", (0, 0))
            fov_size   = self._last_render_params.get("fov_size", self._fov_size)
            row_start, row_end, col_start, col_end, fov_h, fov_w = compute_fov_region(
                info, fov_center, fov_size
            )

            H_render, W_render = rgb.shape[:2]
            result = rgb.copy()
            alpha = self._overlay_opacity

            # Convert level-space region → base-array bounds
            v_scale = info['v_scale']
            h_scale = info['h_scale']
            base_r0 = int(row_start * v_scale)
            base_r1 = int(row_end   * v_scale)
            base_c0 = int(col_start * h_scale)
            base_c1 = int(col_end   * h_scale)

            def resize_mask(mask_full: np.ndarray) -> np.ndarray:
                """Crop and resize annotation/prediction mask to rendered tile."""
                H_base, W_base = mask_full.shape
                r0 = max(0, min(base_r0, H_base - 1))
                r1 = max(r0 + 1, min(base_r1, H_base))
                c0 = max(0, min(base_c0, W_base - 1))
                c1 = max(c0 + 1, min(base_c1, W_base))
                crop = mask_full[r0:r1, c0:c1]
                pil  = PIL_Image.fromarray(crop)
                return np.array(pil.resize((W_render, H_render), PIL_Image.NEAREST))

            def apply_mask(result_arr, mask_resized, classes_src):
                for cls_idx, cls_info in enumerate(classes_src, start=1):
                    where = mask_resized == cls_idx
                    if not where.any():
                        continue
                    r, g, b = cls_info['color']
                    result_arr[where, 0] = np.clip(
                        result_arr[where, 0] * (1 - alpha) + r * alpha, 0, 255
                    ).astype(np.uint8)
                    result_arr[where, 1] = np.clip(
                        result_arr[where, 1] * (1 - alpha) + g * alpha, 0, 255
                    ).astype(np.uint8)
                    result_arr[where, 2] = np.clip(
                        result_arr[where, 2] * (1 - alpha) + b * alpha, 0, 255
                    ).astype(np.uint8)
                return result_arr

            # Merged label overlay (annotation strokes + classifier output)
            if self._show_ann_cb.isChecked() and self._classes and self._label_store is not None:
                label_full = self._label_store.get_slice_2d(self._t, self._z, self._orientation)
                if label_full.any():
                    label_resized = resize_mask(label_full)
                    result = apply_mask(result, label_resized, self._classes)

            # Annotation box border
            if self._annotation_box is not None:
                box = self._annotation_box
                vs, ve = box['v_start'], box['v_end']
                hs, he = box['h_start'], box['h_end']
                # Convert base coords → level coords
                lv_s = vs / v_scale;  lv_e = ve / v_scale
                lh_s = hs / h_scale;  lh_e = he / h_scale
                # Convert level coords → render image coords
                l_h = max(1, row_end - row_start)
                l_w = max(1, col_end - col_start)
                pr = H_render / l_h  # render pixels per level pixel
                pc = W_render / l_w
                br0 = int((lv_s - row_start) * pr)
                br1 = int((lv_e - row_start) * pr)
                bc0 = int((lh_s - col_start) * pc)
                bc1 = int((lh_e - col_start) * pc)
                br0 = max(0, min(H_render - 1, br0))
                br1 = max(0, min(H_render, br1))
                bc0 = max(0, min(W_render - 1, bc0))
                bc1 = max(0, min(W_render, bc1))
                t_ = 2  # border thickness px
                if br1 > br0 and bc1 > bc0:
                    box_color = np.array([255, 160, 0], dtype=np.uint8)
                    result[br0:br0+t_, bc0:bc1] = box_color
                    result[max(0,br1-t_):br1, bc0:bc1] = box_color
                    result[br0:br1, bc0:bc0+t_] = box_color
                    result[br0:br1, max(0,bc1-t_):bc1] = box_color

        except Exception:
            pass

        return result

    # ── Crop: screen → base-array ─────────────────────────────────────────────

    def _screen_to_base(self, sx: float, sy: float) -> Tuple[float, float]:
        """Map widget pixel (sx, sy) to base-array (v_ax, h_ax) coordinates."""
        from zarr_plane_server import compute_fov_region, get_orientation_axes

        x0, y0, dw, dh = self._image_widget._display_rect()
        frac_x = max(0.0, min(1.0, (sx - x0) / max(dw, 1)))
        frac_y = max(0.0, min(1.0, (sy - y0) / max(dh, 1)))

        lp = self._level_paths[min(self._level_idx, len(self._level_paths) - 1)]
        info = get_orientation_axes(self._pyr, lp, self._orientation)
        fov_center = (round(self._fov_center_y), round(self._fov_center_x))
        row_start, row_end, col_start, col_end, fov_h, fov_w = compute_fov_region(
            info, fov_center, self._fov_size
        )

        level_col = col_start + frac_x * (col_end - col_start)
        level_row = row_start + frac_y * (row_end - row_start)
        return level_row * info['v_scale'], level_col * info['h_scale']

    def _on_roi_selected(self, sx1: float, sy1: float, sx2: float, sy2: float):
        if not self._pyr:
            return
        try:
            v1, h1 = self._screen_to_base(sx1, sy1)
            v2, h2 = self._screen_to_base(sx2, sy2)
            v_ax = _ORI[self._orientation][0]
            h_ax = _ORI[self._orientation][1]
            self._crop_spinboxes[f"{v_ax}_start"].setValue(int(min(v1, v2)))
            self._crop_spinboxes[f"{v_ax}_stop"].setValue(int(max(v1, v2)))
            self._crop_spinboxes[f"{h_ax}_start"].setValue(int(min(h1, h2)))
            self._crop_spinboxes[f"{h_ax}_stop"].setValue(int(max(h1, h2)))
        except Exception as exc:
            self._viewer_status.setText(f"ROI error: {exc}")

    # ── Crop: run ─────────────────────────────────────────────────────────────

    def _on_crop_browse(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Cropped OME-Zarr", "", "Zarr (*.zarr)"
        )
        if path:
            if not path.endswith(".zarr"):
                path += ".zarr"
            self._crop_output_path.setText(path)

    def _on_run_crop(self):
        if not self._pyr:
            return
        output_path = self._crop_output_path.text().strip()
        if not output_path:
            self._crop_log.append_line("ERROR: Set an output path first.")
            return

        crop_ranges: Dict[str, Tuple[int, int]] = {}
        for ax in self._axes:
            if ax in self._crop_axis_rows and self._crop_axis_rows[ax].isVisible():
                start = self._crop_spinboxes[f"{ax}_start"].value()
                stop  = self._crop_spinboxes[f"{ax}_stop"].value()
                if start < stop:
                    crop_ranges[ax] = (start, stop)

        # Parse scale factors
        try:
            sf_text = self._crop_scale_factors.text().strip()
            scale_factors = tuple(int(x) for x in sf_text.split(",") if x.strip())
        except ValueError:
            self._crop_log.append_line("ERROR: Invalid scale factors (use comma-separated ints).")
            return

        zarr_format = self._crop_fmt.currentData()
        n_layers    = self._crop_n_levels.value()

        self._crop_log.append_line(f"Starting crop → {output_path}")
        self._crop_log.append_line(f"  crop_ranges={crop_ranges}")
        self._crop_run_btn.setEnabled(False)

        worker = CropWorker(
            source_path=self._path,
            output_path=output_path,
            crop_ranges=crop_ranges,
            zarr_format=zarr_format,
            scale_factors=scale_factors,
            n_layers=n_layers,
            overwrite=True,
        )
        worker.finished.connect(self._on_crop_finished)
        worker.finished.connect(lambda ok, msg: worker.deleteLater())
        worker.start()
        self._crop_worker = worker  # keep reference

    def _on_crop_finished(self, ok: bool, msg: str):
        self._crop_run_btn.setEnabled(True)
        if ok:
            self._crop_log.append_line(f"✓ {msg}")
        else:
            self._crop_log.append_line(f"✗ {msg}")

    # ── Annotate: class management ────────────────────────────────────────────

    def _add_class(self):
        idx = len(self._classes)
        color = _CLASS_PALETTE[idx % len(_CLASS_PALETTE)]
        name  = f"Class {idx + 1}"
        self._classes.append({"name": name, "color": color})
        self._annotations_dirty = True

        item = QListWidgetItem(name)
        r, g, b = color
        item.setForeground(QColor(r, g, b))
        item.setFont(QFont("sans-serif", 10, QFont.Weight.Bold))
        self._class_list.addItem(item)
        self._class_list.setCurrentRow(self._class_list.count() - 1)

    def _remove_class(self):
        row = self._class_list.currentRow()
        if row < 0 or not self._classes:
            return
        self._classes.pop(row)
        self._class_list.takeItem(row)
        # Re-index: remove the class and shift higher indices down by 1
        if self._label_store is not None:
            arr = self._label_store.array
            arr[arr == (row + 1)] = 0
            for cls_idx in range(row + 2, len(self._classes) + 2):
                arr[arr == cls_idx] = cls_idx - 1
        self._annotations_dirty = True
        self._schedule_render()

    def _active_class_idx(self) -> int:
        row = self._class_list.currentRow()
        return row + 1 if row >= 0 else 1

    # ── Annotate: brush paint ─────────────────────────────────────────────────

    def _on_box_selected(self, sx1: float, sy1: float, sx2: float, sy2: float):
        """Store the drawn annotation box in base-array coordinates."""
        if not self._pyr:
            return
        try:
            v1, h1 = self._screen_to_base(sx1, sy1)
            v2, h2 = self._screen_to_base(sx2, sy2)
            v_ax = _ORI[self._orientation][0]
            h_ax = _ORI[self._orientation][1]
            self._annotation_box = {
                'v_ax': v_ax,   'h_ax': h_ax,
                'v_start': int(min(v1, v2)), 'v_end': int(max(v1, v2)),
                'h_start': int(min(h1, h2)), 'h_end': int(max(h1, h2)),
                'orientation': self._orientation,
            }
            self._schedule_render()
        except Exception as exc:
            self._viewer_status.setText(f"Box error: {exc}")

    def _on_annotate_at(self, sx: float, sy: float):
        if not self._pyr or self._label_store is None:
            return
        if not self._classes:
            self._viewer_status.setText("Add at least one class before annotating.")
            return
        try:
            base_row, base_col = self._screen_to_base(sx, sy)
            self._label_store.paint(
                t=self._t, z=self._z, orientation=self._orientation,
                row=int(base_row), col=int(base_col),
                radius=self._brush_size, class_idx=self._active_class_idx(),
            )
            self._annotations_dirty = True
            self._schedule_render()
        except Exception as exc:
            self._viewer_status.setText(f"Annotate error: {exc}")

    def _on_erase_at(self, sx: float, sy: float):
        """Erase labels at the brush position (right-click shortcut)."""
        if not self._pyr or self._label_store is None:
            return
        try:
            base_row, base_col = self._screen_to_base(sx, sy)
            self._label_store.paint(
                t=self._t, z=self._z, orientation=self._orientation,
                row=int(base_row), col=int(base_col),
                radius=self._brush_size, class_idx=0,
            )
            self._annotations_dirty = True
            self._schedule_render()
        except Exception as exc:
            self._viewer_status.setText(f"Erase error: {exc}")

    def _on_erase_btn_toggled(self, checked: bool):
        """Erase button gives visual feedback; actual erasing is via right-click."""
        if checked:
            self._erase_btn.setStyleSheet("background-color: #a04040;")
        else:
            self._erase_btn.setStyleSheet("")

    def _on_clear_slice(self):
        if self._label_store:
            self._label_store.clear_slice(self._t, self._z, self._orientation)
            self._annotations_dirty = True
            self._schedule_render()

    def _on_clear_all(self):
        if self._label_store:
            self._label_store.clear_all()
            self._trained_clf = None
            self._annotations_dirty = True
            self._schedule_render()

    # ── Annotate: classifier ──────────────────────────────────────────────────

    def _on_opacity_changed(self, value: int):
        self._overlay_opacity = value / 100.0
        self._schedule_render()

    def _get_current_slice_data(self) -> Optional[np.ndarray]:
        """Return (H, W) or (H, W, C) float32 data for the current view plane."""
        if self._pyr is None:
            return None
        try:
            base = self._pyr.base_array
            if isinstance(base, zarr.Array):
                base = da.from_zarr(base)

            v_ax, h_ax, through_ax, _ = _ORI[self._orientation]
            axes = self._axes
            idx = []
            for ax in axes:
                if ax == 't':
                    idx.append(min(self._t, max(0, self._dim('t') - 1)))
                elif ax == through_ax:
                    idx.append(min(self._z, max(0, self._dim(through_ax) - 1)))
                else:
                    idx.append(slice(None))

            sliced = base[tuple(idx)]
            data = sliced.compute().astype(np.float32)

            # Determine remaining axes and transpose to (H, W[, C])
            remaining = [ax for ax in axes if ax not in ('t', through_ax)]
            if 'c' in remaining:
                c_pos = remaining.index('c')
                v_pos = remaining.index(v_ax)
                h_pos = remaining.index(h_ax)
                perm  = [v_pos, h_pos, c_pos]
                data  = data.transpose(perm)
            else:
                v_pos = remaining.index(v_ax)
                h_pos = remaining.index(h_ax)
                data  = data.transpose([v_pos, h_pos])
            return data
        except Exception as exc:
            self._clf_status.setText(f"Data error: {exc}")
            return None

    def _on_run_classifier(self):
        if self._label_store is None or not self._classes:
            self._clf_status.setText("Load a dataset and add classes first.")
            return
        if self._clf_running:
            return
        if self._annotation_box is None:
            self._clf_status.setText("Draw an annotation box first (Draw Box mode).")
            return

        slice_data = self._get_current_slice_data()
        if slice_data is None:
            return

        # Crop image data to the box region
        box = self._annotation_box
        vs, ve = box['v_start'], box['v_end']
        hs, he = box['h_start'], box['h_end']
        H_img = slice_data.shape[0]
        W_img = slice_data.shape[1]
        vs, ve = max(0, vs), min(H_img, ve)
        hs, he = max(0, hs), min(W_img, he)
        if ve <= vs or he <= hs:
            self._clf_status.setText("Annotation box is outside the image.")
            return

        slice_region = slice_data[vs:ve, hs:he]

        # Crop merged labels to the box region
        label_full   = self._label_store.get_slice_2d(self._t, self._z, self._orientation)
        label_region = label_full[vs:ve, hs:he].copy()

        reuse_model = (self._trained_clf is not None) and (not self._annotations_dirty)

        if reuse_model:
            # Use the exact feature config that was used during training so the
            # feature dimension matches the fitted scaler/classifier.
            feat = self._feature_config or {}
        else:
            # Collect current UI selections as the new feature config.
            sigmas = [s for s, cb in self._sigma_checks.items() if cb.isChecked()] or [1, 2, 4]
            gauss_types = [t for t, cb in self._gauss_type_checks.items() if cb.isChecked()]
            window_sizes = [w for w, cb in self._window_size_checks.items() if cb.isChecked()]
            window_types = [t for t, cb in self._window_type_checks.items() if cb.isChecked()]
            include_raw = self._include_raw_cb.isChecked()
            feat = dict(
                sigmas=sigmas,
                gaussian_types=gauss_types,
                window_sizes=window_sizes,
                window_types=window_types,
                include_raw=include_raw,
                use_dinov2=self._dinov2_cb.isChecked(),
                dinov2_model=self._dinov2_model_combo.currentData(),
                dinov2_device=self._dinov2_device_combo.currentData(),
                classifier=self._clf_type_combo.currentData(),
            )

        self._clf_running = True
        self._clf_status.setText(
            "Predicting (existing model)…" if reuse_model else "Training…"
        )
        self._clf_run_btn.setEnabled(False)

        # Store box coords so _on_classifier_done can write back
        self._pending_box = (vs, ve, hs, he)
        self._pending_feat = feat  # remember which config was used for this run

        worker = ClassifierWorker(
            label_region=label_region,
            slice_region=slice_region,
            existing_model=self._trained_clf if reuse_model else None,
            **feat,
        )
        worker.finished.connect(self._on_classifier_done)
        worker.error.connect(self._on_classifier_error)
        worker.start()
        self._classifier_worker = worker

    def _on_classifier_done(self, prediction: np.ndarray, clf):
        self._clf_running = False
        self._classifier_worker = None
        self._trained_clf = clf
        self._annotations_dirty = False
        if hasattr(self, '_pending_feat'):
            self._feature_config = self._pending_feat

        # Write predictions back into _label_store within the box region only
        if self._label_store is not None and hasattr(self, '_pending_box'):
            vs, ve, hs, he = self._pending_box
            self._label_store.set_region_2d(
                self._t, self._z, self._orientation,
                vs, ve, hs, he, prediction,
            )

        self._clf_status.setText("Done.")
        self._clf_run_btn.setEnabled(True)
        self._schedule_render()

    def _on_classifier_error(self, msg: str):
        self._clf_running = False
        self._classifier_worker = None
        self._clf_status.setText(f"Error: {msg}")
        self._clf_run_btn.setEnabled(True)

    # ── Annotate: save ────────────────────────────────────────────────────────

    def _on_save_annotations(self):
        if self._label_store is None or not self._path:
            return
        if not self._label_store.array.any():
            self._save_status.setText("No labels to save — annotate or run the classifier first.")
            return

        label_name = self._label_name_edit.text().strip() or "labels"

        # Build mask array from the merged label store; drop C axis (labels have no C)
        axes = self._axes
        ann_array = self._label_store.array  # nD uint8, same shape as image

        if 'c' in axes:
            c_idx = axes.index('c')
            # Max-project along C (class index is consistent across C)
            ann_array = ann_array.max(axis=c_idx)
            label_axes = axes.replace('c', '')
        else:
            label_axes = axes

        # Physical scales (drop C axis too)
        try:
            base_scales = list(self._pyr.meta.get_scale(self._pyr.meta.resolution_paths[0]))
            base_units  = list(self._pyr.meta.unit_list or [])
            if 'c' in axes:
                c_idx_full = list(axes).index('c')
                base_scales.pop(c_idx_full)
                if c_idx_full < len(base_units):
                    base_units.pop(c_idx_full)
        except Exception:
            base_scales = [1.0] * len(label_axes)
            base_units  = [None] * len(label_axes)

        class_colors = [cls['color'] for cls in self._classes] if self._classes else [(255, 0, 0)]

        self._save_status.setText("Saving…")
        self._save_btn.setEnabled(False)

        import threading

        def _do_save():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(write_ngff_labels(
                    zarr_path=self._path,
                    label_name=label_name,
                    mask_array=ann_array,
                    axes=label_axes,
                    scales=base_scales,
                    units=base_units,
                    class_colors=class_colors,
                ))
                QTimer.singleShot(0, lambda: self._save_status.setText("✓ Saved"))
            except Exception as exc:
                msg = str(exc)
                QTimer.singleShot(0, lambda: self._save_status.setText(f"✗ {msg[:60]}"))
            finally:
                loop.close()
                asyncio.set_event_loop(None)
                QTimer.singleShot(0, lambda: self._save_btn.setEnabled(True))

        threading.Thread(target=_do_save, daemon=True).start()
