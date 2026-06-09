"""
Process page — Crop and Annotate sub-tabs sharing a single viewer.

Layout:
  Left  : SidebarBrowser (zarr mode)
  Right : QVBoxLayout
            ZarrViewer (shared, always visible)
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
from eubi_bridge.qt_gui.widgets.log_widget import LogWidget
from eubi_bridge.qt_gui.widgets.sidebar_browser import SidebarBrowser
from eubi_bridge.qt_gui.widgets.zarr_viewer import ZarrViewer, _ORI
from eubi_annotate.qt_gui.workers.classifier_worker import ClassifierWorker
from eubi_annotate.qt_gui.workers.crop_worker import CropWorker

# ── Optional PyTorch / DINOv2 availability ────────────────────────────────────
try:
    import torch as _torch
    _TORCH_AVAILABLE = True
    _CUDA_AVAILABLE  = _torch.cuda.is_available()
except ImportError:
    _TORCH_AVAILABLE = False
    _CUDA_AVAILABLE  = False

# ── Constants ─────────────────────────────────────────────────────────────────

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


def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))  # type: ignore


# ── ProcessPage ───────────────────────────────────────────────────────────────

class ProcessPage(QWidget):
    """Process tab: Crop and Annotate sub-tabs with a shared viewer."""

    status_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        # ── Annotation state ──────────────────────────────────────────────────
        self._label_store: Optional[AnnotationStore] = None
        self._ann_lp: Optional[str] = None
        self._ann_v_scale: float = 1.0
        self._ann_h_scale: float = 1.0
        self._annotation_box: Optional[dict] = None
        self._classes: List[dict] = []
        self._overlay_opacity = 0.5
        self._brush_size = 5
        self._classifier_worker: Optional[ClassifierWorker] = None
        self._clf_running = False
        self._trained_clf = None
        self._feature_config: Optional[dict] = None
        self._save_worker: Optional[QTimer] = None

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

        # Viewer panel (top): ZarrViewer + process-specific controls
        viewer_container = self._build_viewer_panel()
        right_splitter.addWidget(viewer_container)

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
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # ZarrViewer (no reload button, no FOV combo: kept compact for the process page)
        self._viewer = ZarrViewer(
            self,
            enable_minmax=False,
            show_fov_combo=False,
            show_reload_btn=False,
        )
        self._viewer.frame_ready.connect(self._on_frame_ready)
        self._viewer.dataset_loaded.connect(self._on_dataset_loaded)
        self._viewer.status_changed.connect(self.status_changed)

        # Connect image-widget signals used by annotation/crop handlers
        self._viewer.image_widget.roi_selected.connect(self._on_roi_selected)
        self._viewer.image_widget.box_selected.connect(self._on_box_selected)
        self._viewer.image_widget.annotate_at.connect(self._on_annotate_at)
        self._viewer.image_widget.erase_at.connect(self._on_erase_at)

        layout.addWidget(self._viewer)

        # Process-specific overlay controls row
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
        self._crop_pan_btn.clicked.connect(
            lambda: self._viewer.image_widget.set_mode('pan'))
        self._crop_draw_btn.clicked.connect(
            lambda: self._viewer.image_widget.set_mode('crop'))
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
        self._ann_pan_btn.clicked.connect(
            lambda: self._viewer.image_widget.set_mode('pan'))
        self._ann_box_btn.clicked.connect(
            lambda: self._viewer.image_widget.set_mode('box'))
        self._ann_draw_btn.clicked.connect(
            lambda: self._viewer.image_widget.set_mode('annotate'))
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

        # Erase row
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
        bg.addWidget(QLabel("Size (view px):"))
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
            cb.setChecked(sz <= 7)
            w_size_row.addWidget(cb)
            self._window_size_checks[sz] = cb
        w_size_row.addStretch()
        cfg.addLayout(w_size_row)

        self._include_raw_cb = QCheckBox("Raw (unfiltered) intensity")
        self._include_raw_cb.setChecked(True)
        cfg.addWidget(self._include_raw_cb)

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

        clf_type_row = QHBoxLayout()
        clf_type_row.addWidget(QLabel("Classifier:"))
        self._clf_type_combo = QComboBox()
        self._clf_type_combo.addItem("Random Forest", "rf")
        self._clf_type_combo.addItem("XGBoost", "xgb")
        clf_type_row.addWidget(self._clf_type_combo)
        clf_type_row.addStretch()
        cfg.addLayout(clf_type_row)

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
        """Called by SidebarBrowser; delegates to ZarrViewer."""
        self._viewer.load_dataset(path)

    # ── Annotation level management ───────────────────────────────────────────

    def _ensure_ann_level(self) -> bool:
        """Ensure the annotation store matches the current viewer zoom level.

        Called lazily before every annotation action.  If the viewer level has
        changed since the last action, the label store is reinitialised for the
        new level (clearing any existing annotations) and the annotation box is
        reset.  The trained model is intentionally preserved so inference can
        continue at any zoom level.

        Returns True when ready, False if no data is loaded.
        """
        from zarr_plane_server import get_orientation_axes

        if not self._viewer.pyr or not self._viewer.level_paths:
            return False
        level_paths = self._viewer.level_paths
        new_lp = level_paths[min(self._viewer.level_idx, len(level_paths) - 1)]
        if new_lp == self._ann_lp and self._label_store is not None:
            return True
        pyr       = self._viewer.pyr
        new_info  = get_orientation_axes(pyr, new_lp, self._viewer.orientation)
        new_shape = pyr.layers[new_lp].shape
        self._label_store  = AnnotationStore(new_shape, pyr.axes.lower())
        self._ann_lp       = new_lp
        self._ann_v_scale  = new_info['v_scale']
        self._ann_h_scale  = new_info['h_scale']
        self._annotation_box = None
        self._viewer.image_widget.clear_annotation_box()
        return True

    def _on_dataset_loaded(self, pyr):
        """Called when ZarrViewer has finished loading the pyramid."""
        self._label_store    = None
        self._ann_lp         = None
        self._ann_v_scale    = 1.0
        self._ann_h_scale    = 1.0
        self._annotation_box = None
        self._trained_clf    = None
        self._feature_config = None
        self._viewer.image_widget.clear_annotation_box()

        self._init_crop_spinboxes()
        self._crop_run_btn.setEnabled(True)
        self._clf_run_btn.setEnabled(True)
        self._save_btn.setEnabled(True)

    def _dim(self, ax: str) -> int:
        """Size of axis *ax* at base level (used by crop spinboxes)."""
        idx = self._viewer.axes.find(ax)
        return self._viewer.shape[idx] if idx >= 0 else 1

    def _ann_dim(self, ax: str) -> int:
        """Size of axis *ax* at the annotation level."""
        if self._label_store is None:
            return 1
        idx = self._viewer.axes.find(ax)
        return self._label_store.array.shape[idx] if idx >= 0 else 1

    def _init_crop_spinboxes(self):
        axes = self._viewer.axes
        for ax, row_w in self._crop_axis_rows.items():
            present = ax in axes
            row_w.setVisible(present)
            if present:
                dim = self._dim(ax)
                self._crop_spinboxes[f"{ax}_start"].setMaximum(max(0, dim - 1))
                self._crop_spinboxes[f"{ax}_start"].setValue(0)
                self._crop_spinboxes[f"{ax}_stop"].setMaximum(dim)
                self._crop_spinboxes[f"{ax}_stop"].setValue(dim)

    # ── Viewer event callbacks ────────────────────────────────────────────────

    def _on_overlay_toggle(self):
        self._viewer._schedule_render()

    def _on_subtab_changed(self, idx: int):
        self._viewer.image_widget.set_mode('pan')
        if idx == 0:
            self._crop_pan_btn.setChecked(True)
        else:
            self._ann_pan_btn.setChecked(True)

    # ── Frame-ready handler ───────────────────────────────────────────────────

    def _on_frame_ready(self, rgb: np.ndarray, elapsed_ms: float, generation: int):
        rgb = self._composite_overlays(rgb)
        self._viewer.image_widget.set_frame(rgb, 1.0, elapsed_ms)
        self._viewer_status.setText(f"{elapsed_ms:.0f} ms")

    # ── Overlay compositing ───────────────────────────────────────────────────

    def _composite_overlays(self, rgb: np.ndarray) -> np.ndarray:
        """Alpha-composite annotation and classifier masks onto rendered rgb."""
        if self._label_store is None:
            return rgb
        last_params = self._viewer.get_last_render_params()
        if not last_params:
            return rgb

        level_paths = self._viewer.level_paths
        pyr         = self._viewer.pyr

        try:
            from zarr_plane_server import compute_fov_region, get_orientation_axes
            from PIL import Image as PIL_Image

            lp = level_paths[
                min(last_params.get("level_idx", 0), len(level_paths) - 1)
            ]
            # Annotations belong to a specific zoom level — skip at other levels.
            if lp != self._ann_lp:
                return rgb

            ori        = last_params.get("orientation", "XY")
            info       = get_orientation_axes(pyr, lp, ori)
            fov_center = last_params.get("fov_center", (0, 0))
            fov_size   = last_params.get("fov_size", self._viewer.fov_size)
            row_start, row_end, col_start, col_end, fov_h, fov_w = compute_fov_region(
                info, fov_center, fov_size
            )

            H_render, W_render = rgb.shape[:2]
            result = rgb.copy()
            alpha  = self._overlay_opacity

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

            if self._show_ann_cb.isChecked() and self._classes and self._label_store is not None:
                # Annotation and viewer are at the same level — coords are identical.
                label_region = self._label_store.get_region_2d(
                    self._viewer.t, self._viewer.z, self._viewer.orientation,
                    row_start, row_end, col_start, col_end,
                )
                if label_region.any():
                    pil_lbl = PIL_Image.fromarray(label_region)
                    label_resized = np.array(pil_lbl.resize((W_render, H_render), PIL_Image.NEAREST))
                    result = apply_mask(result, label_resized, self._classes)

            if self._annotation_box is not None:
                box  = self._annotation_box
                vs, ve = box['v_start'], box['v_end']   # current-level coords
                hs, he = box['h_start'], box['h_end']
                l_h  = max(1, row_end - row_start)
                l_w  = max(1, col_end - col_start)
                pr   = H_render / l_h
                pc   = W_render / l_w
                br0  = int((vs - row_start) * pr)
                br1  = int((ve - row_start) * pr)
                bc0  = int((hs - col_start) * pc)
                bc1  = int((he - col_start) * pc)
                br0  = max(0, min(H_render - 1, br0))
                br1  = max(0, min(H_render, br1))
                bc0  = max(0, min(W_render - 1, bc0))
                bc1  = max(0, min(W_render, bc1))
                t_   = 2
                if br1 > br0 and bc1 > bc0:
                    box_color = np.array([255, 160, 0], dtype=np.uint8)
                    result[br0:br0+t_, bc0:bc1] = box_color
                    result[max(0,br1-t_):br1, bc0:bc1] = box_color
                    result[br0:br1, bc0:bc0+t_] = box_color
                    result[br0:br1, max(0,bc1-t_):bc1] = box_color

        except Exception:
            pass

        return result

    # ── Screen → level/base/annotation coordinate helpers ─────────────────────

    def _screen_to_level(self, sx: float, sy: float):
        """Map widget pixel (sx, sy) to current-level (row, col) + info dict.

        Returns (level_row, level_col, info).  Multiply by info['v_scale'] /
        info['h_scale'] to convert to base-level coordinates (e.g. for crop
        spinboxes).  Since annotation level == viewer level, annotation coords
        equal the returned level coords directly.
        """
        from zarr_plane_server import compute_fov_region, get_orientation_axes

        x0, y0, dw, dh = self._viewer.image_widget._display_rect()
        frac_x = max(0.0, min(1.0, (sx - x0) / max(dw, 1)))
        frac_y = max(0.0, min(1.0, (sy - y0) / max(dh, 1)))

        level_paths = self._viewer.level_paths
        lp   = level_paths[min(self._viewer.level_idx, len(level_paths) - 1)]
        info = get_orientation_axes(self._viewer.pyr, lp, self._viewer.orientation)
        fov_center = (round(self._viewer.fov_center[0]), round(self._viewer.fov_center[1]))
        row_start, row_end, col_start, col_end, fov_h, fov_w = compute_fov_region(
            info, fov_center, self._viewer.fov_size
        )

        level_row = row_start + frac_y * (row_end - row_start)
        level_col = col_start + frac_x * (col_end - col_start)
        return level_row, level_col, info

    def _on_roi_selected(self, sx1: float, sy1: float, sx2: float, sy2: float):
        if not self._viewer.pyr:
            return
        try:
            lv_r1, lv_c1, info = self._screen_to_level(sx1, sy1)
            lv_r2, lv_c2, _    = self._screen_to_level(sx2, sy2)
            v1 = lv_r1 * info['v_scale']   # convert to base coords for crop
            h1 = lv_c1 * info['h_scale']
            v2 = lv_r2 * info['v_scale']
            h2 = lv_c2 * info['h_scale']
            v_ax = _ORI[self._viewer.orientation][0]
            h_ax = _ORI[self._viewer.orientation][1]
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
        if not self._viewer.pyr:
            return
        output_path = self._crop_output_path.text().strip()
        if not output_path:
            self._crop_log.append_line("ERROR: Set an output path first.")
            return

        crop_ranges: Dict[str, Tuple[int, int]] = {}
        for ax in self._viewer.axes:
            if ax in self._crop_axis_rows and self._crop_axis_rows[ax].isVisible():
                start = self._crop_spinboxes[f"{ax}_start"].value()
                stop  = self._crop_spinboxes[f"{ax}_stop"].value()
                if start < stop:
                    crop_ranges[ax] = (start, stop)

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
            source_path=self._viewer.path,
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
        self._crop_worker = worker

    def _on_crop_finished(self, ok: bool, msg: str):
        self._crop_run_btn.setEnabled(True)
        if ok:
            self._crop_log.append_line(f"✓ {msg}")
        else:
            self._crop_log.append_line(f"✗ {msg}")

    # ── Annotate: class management ────────────────────────────────────────────

    def _add_class(self):
        idx   = len(self._classes)
        color = _CLASS_PALETTE[idx % len(_CLASS_PALETTE)]
        name  = f"Class {idx + 1}"
        self._classes.append({"name": name, "color": color})

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
        if self._label_store is not None:
            arr = self._label_store.array
            arr[arr == (row + 1)] = 0
            for cls_idx in range(row + 2, len(self._classes) + 2):
                arr[arr == cls_idx] = cls_idx - 1
        self._viewer._schedule_render()

    def _active_class_idx(self) -> int:
        row = self._class_list.currentRow()
        return row + 1 if row >= 0 else 1

    # ── Annotate: brush paint ─────────────────────────────────────────────────

    def _on_box_selected(self, sx1: float, sy1: float, sx2: float, sy2: float):
        if not self._viewer.pyr:
            return
        if not self._ensure_ann_level():
            return
        try:
            lv_r1, lv_c1, _ = self._screen_to_level(sx1, sy1)
            lv_r2, lv_c2, _ = self._screen_to_level(sx2, sy2)
            v_ax = _ORI[self._viewer.orientation][0]
            h_ax = _ORI[self._viewer.orientation][1]
            self._annotation_box = {
                'v_ax': v_ax,   'h_ax': h_ax,
                'v_start': int(min(lv_r1, lv_r2)), 'v_end': int(max(lv_r1, lv_r2)),
                'h_start': int(min(lv_c1, lv_c2)), 'h_end': int(max(lv_c1, lv_c2)),
                'orientation': self._viewer.orientation,
            }
            self._viewer._schedule_render()
        except Exception as exc:
            self._viewer_status.setText(f"Box error: {exc}")

    def _on_annotate_at(self, sx: float, sy: float):
        if not self._viewer.pyr:
            return
        if not self._ensure_ann_level():
            return
        if not self._classes:
            self._viewer_status.setText("Add at least one class before annotating.")
            return
        try:
            lv_row, lv_col, _ = self._screen_to_level(sx, sy)
            # Annotation level == viewer level → coordinates are identical.
            self._label_store.paint(
                t=self._viewer.t, z=self._viewer.z, orientation=self._viewer.orientation,
                row=int(lv_row), col=int(lv_col),
                radius=max(1, self._brush_size),
                class_idx=self._active_class_idx(),
            )
            self._viewer._schedule_render()
        except Exception as exc:
            self._viewer_status.setText(f"Annotate error: {exc}")

    def _on_erase_at(self, sx: float, sy: float):
        if not self._viewer.pyr:
            return
        if not self._ensure_ann_level():
            return
        try:
            lv_row, lv_col, _ = self._screen_to_level(sx, sy)
            self._label_store.paint(
                t=self._viewer.t, z=self._viewer.z, orientation=self._viewer.orientation,
                row=int(lv_row), col=int(lv_col),
                radius=max(1, self._brush_size),
                class_idx=0,
            )
            self._viewer._schedule_render()
        except Exception as exc:
            self._viewer_status.setText(f"Erase error: {exc}")

    def _on_erase_btn_toggled(self, checked: bool):
        if checked:
            self._erase_btn.setStyleSheet("background-color: #a04040;")
        else:
            self._erase_btn.setStyleSheet("")

    def _on_clear_slice(self):
        if self._label_store:
            self._label_store.clear_slice(
                self._viewer.t, self._viewer.z, self._viewer.orientation
            )
            self._viewer._schedule_render()

    def _on_clear_all(self):
        if self._label_store:
            self._label_store.clear_all()
            self._viewer._schedule_render()

    # ── Annotate: classifier ──────────────────────────────────────────────────

    def _on_opacity_changed(self, value: int):
        self._overlay_opacity = value / 100.0
        self._viewer._schedule_render()

    def _get_windowed_slice_data(
        self, v0: int, v1: int, h0: int, h1: int
    ) -> Optional[np.ndarray]:
        """Read only the [v0:v1, h0:h1] box from the annotation-level array.

        Coordinates are in annotation-level space.  Only the box chunks are
        fetched — no full-plane read regardless of image size.
        """
        if self._viewer.pyr is None:
            return None
        try:
            base = self._viewer.pyr.layers[self._ann_lp]
            if isinstance(base, zarr.Array):
                base = da.from_zarr(base)

            axes = self._viewer.axes
            v_ax, h_ax, through_ax, _ = _ORI[self._viewer.orientation]
            idx = []
            for ax in axes:
                if ax == 't':
                    idx.append(min(self._viewer.t, max(0, self._dim('t') - 1)))
                elif ax == through_ax:
                    idx.append(min(self._viewer.z, max(0, self._dim(through_ax) - 1)))
                elif ax == v_ax:
                    idx.append(slice(v0, v1))
                elif ax == h_ax:
                    idx.append(slice(h0, h1))
                else:
                    idx.append(slice(None))

            sliced = base[tuple(idx)]
            data   = sliced.compute().astype(np.float32)

            remaining = [ax for ax in axes if ax not in ('t', through_ax)]
            if 'c' in remaining:
                c_pos = remaining.index('c')
                v_pos = remaining.index(v_ax)
                h_pos = remaining.index(h_ax)
                data  = data.transpose([v_pos, h_pos, c_pos])
            else:
                v_pos = remaining.index(v_ax)
                h_pos = remaining.index(h_ax)
                data  = data.transpose([v_pos, h_pos])
            return data
        except Exception as exc:
            self._clf_status.setText(f"Data error: {exc}")
            return None

    def _on_run_classifier(self):
        if not self._ensure_ann_level():
            self._clf_status.setText("Load a dataset first.")
            return
        if not self._classes:
            self._clf_status.setText("Add classes first.")
            return
        if self._clf_running:
            return
        if self._annotation_box is None:
            self._clf_status.setText("Draw an annotation box first (Draw Box mode).")
            return

        box = self._annotation_box
        vs, ve = box['v_start'], box['v_end']
        hs, he = box['h_start'], box['h_end']
        H_img  = self._ann_dim(box['v_ax'])
        W_img  = self._ann_dim(box['h_ax'])
        vs, ve = max(0, vs), min(H_img, ve)
        hs, he = max(0, hs), min(W_img, he)
        if ve <= vs or he <= hs:
            self._clf_status.setText("Annotation box is outside the image.")
            return

        slice_region = self._get_windowed_slice_data(vs, ve, hs, he)
        if slice_region is None:
            return

        label_region = self._label_store.get_region_2d(
            self._viewer.t, self._viewer.z, self._viewer.orientation,
            vs, ve, hs, he,
        )

        # Mode: labels in box → train/refine; no labels + existing model → infer.
        flat       = label_region.ravel()
        labeled    = flat > 0
        has_labels = labeled.any() and len(np.unique(flat[labeled])) >= 2

        if not has_labels and self._trained_clf is None:
            self._clf_status.setText(
                "No labels in box and no existing model. Paint at least 2 classes first."
            )
            return

        if has_labels:
            sigmas       = [s for s, cb in self._sigma_checks.items() if cb.isChecked()] or [1, 2, 4]
            gauss_types  = [t for t, cb in self._gauss_type_checks.items() if cb.isChecked()]
            window_sizes = [w for w, cb in self._window_size_checks.items() if cb.isChecked()]
            window_types = [t for t, cb in self._window_type_checks.items() if cb.isChecked()]
            include_raw  = self._include_raw_cb.isChecked()
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
            status_msg = "Refining…" if self._trained_clf is not None else "Training…"
            self._pending_feat = feat          # update stored config after training
        else:
            # Inference only — must reuse the feature config from training.
            feat = dict(self._feature_config or {})
            feat.setdefault('classifier', self._clf_type_combo.currentData())
            status_msg = "Running inference…"
            self._pending_feat = None          # don't overwrite config after infer

        self._clf_running = True
        self._clf_status.setText(status_msg)
        self._clf_run_btn.setEnabled(False)
        self._pending_box = (vs, ve, hs, he)

        worker = ClassifierWorker(
            label_region=label_region,
            slice_region=slice_region,
            existing_model=self._trained_clf,  # always forwarded; worker decides
            **feat,
        )
        worker.finished.connect(self._on_classifier_done)
        worker.error.connect(self._on_classifier_error)
        worker.start()
        self._classifier_worker = worker

    def _on_classifier_done(self, prediction: np.ndarray, clf):
        self._clf_running       = False
        self._classifier_worker = None
        self._trained_clf       = clf
        if getattr(self, '_pending_feat', None) is not None:
            self._feature_config = self._pending_feat

        if self._label_store is not None and hasattr(self, '_pending_box'):
            vs, ve, hs, he = self._pending_box
            self._label_store.set_region_2d(
                self._viewer.t, self._viewer.z, self._viewer.orientation,
                vs, ve, hs, he, prediction,
            )

        self._clf_status.setText("Done.")
        self._clf_run_btn.setEnabled(True)
        self._viewer._schedule_render()

    def _on_classifier_error(self, msg: str):
        self._clf_running       = False
        self._classifier_worker = None
        self._clf_status.setText(f"Error: {msg}")
        self._clf_run_btn.setEnabled(True)

    # ── Annotate: save ────────────────────────────────────────────────────────

    def _on_save_annotations(self):
        if self._label_store is None or not self._viewer.path:
            return
        if not self._label_store.array.any():
            self._save_status.setText("No labels to save — annotate or run the classifier first.")
            return

        label_name = self._label_name_edit.text().strip() or "labels"

        axes      = self._viewer.axes
        ann_array = self._label_store.array

        if 'c' in axes:
            c_idx     = axes.index('c')
            ann_array = ann_array.max(axis=c_idx)
            label_axes = axes.replace('c', '')
        else:
            label_axes = axes

        try:
            base_scales = list(self._viewer.pyr.meta.get_scale(self._ann_lp))
            base_units  = list(self._viewer.pyr.meta.unit_list or [])
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
        zarr_path = self._viewer.path

        def _do_save():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(write_ngff_labels(
                    zarr_path=zarr_path,
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
