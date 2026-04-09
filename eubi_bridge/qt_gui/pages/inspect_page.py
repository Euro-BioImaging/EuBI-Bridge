"""
Inspect page — OME-Zarr viewer + metadata editor.

Two-pane layout:
  Left  : SidebarBrowser (zarr mode) — navigate filesystem, click OME-Zarr to load
  Right : QTabWidget
            Metadata tab — dataset info, pixel sizes, pyramid layers (read-only tree)
            Viewer tab   — sub-splitter:
                             left  : navigation controls + channel labels editor + intensity panel
                             right : ImageWidget + status bar
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from eubi_bridge.qt_gui.widgets.channel_panel import ChannelPanel
from eubi_bridge.qt_gui.widgets.image_widget import ImageWidget
from eubi_bridge.qt_gui.widgets.sidebar_browser import SidebarBrowser
from eubi_bridge.qt_gui.workers.minmax_worker import MinMaxWorker
from eubi_bridge.qt_gui.workers.render_worker import RenderWorker

_SERVER_DIR = os.path.join(os.path.dirname(__file__), "..", "server")
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

from zarr_plane_server import pyramid_cache, _compressor_info  # type: ignore

# ── Unit lists ────────────────────────────────────────────────────────────────
_SPACE_UNITS = ["picometer", "nanometer", "micrometer", "millimeter", "centimeter", "meter", "kilometer"]
_TIME_UNITS  = ["nanosecond", "microsecond", "millisecond", "second", "minute", "hour"]

_DEFAULT_COLORS = [
    "#FFFFFF", "#FF0000", "#00FF00", "#0000FF",
    "#FFFF00", "#FF00FF", "#00FFFF", "#FF8000",
]

FOV_SIZES = [128, 256, 512, 1024]


def _fmt_compressor(info: dict) -> str:
    """Return a short human-readable compression string, e.g. 'blosc/lz4 L5 shuffle'."""
    name = info.get("name", "none")
    if name == "none":
        return "none"
    p = info.get("params", {})
    parts = [name]
    if "inner_codec" in p:
        parts[0] = f"{name}/{p['inner_codec']}"
    if "level" in p:
        parts.append(f"L{p['level']}")
    if "shuffle" in p and p["shuffle"] not in (None, "noshuffle", 0):
        parts.append(str(p["shuffle"]))
    return " ".join(parts)


def _tree_row(parent: QTreeWidgetItem, key: str, value: str) -> QTreeWidgetItem:
    item = QTreeWidgetItem(parent, [key, value])
    return item

# Mirrors zarr_plane_server.ORIENTATIONS — (v_axis, h_axis, through_axis, label)
_ORI = {
    "XY": ("y", "x", "z", "Z"),
    "XZ": ("z", "x", "y", "Y"),
    "YZ": ("y", "z", "x", "X"),
}


class InspectPage(QWidget):
    """Full inspect / viewer page."""

    status_changed = pyqtSignal(str)   # for main window status bar

    def __init__(self, parent=None):
        super().__init__(parent)

        # ── Viewer state ──────────────────────────────────────────────────────
        self._path         = ""
        self._pyr          = None
        self._axes         = ""
        self._shape: list  = []
        self._level_paths: list[str] = []
        self._channels: list[dict]   = []
        self._pixel_sizes: list[dict] = []

        self._level_idx    = 0
        self._orientation  = "XY"
        self._t            = 0
        self._z            = 0
        self._fov_size     = 512
        self._fov_center_y = 0.0   # float — accumulates sub-pixel pan movements
        self._fov_center_x = 0.0

        # ── Render worker ─────────────────────────────────────────────────────
        self._render_gen = 0   # latest generation sent to render worker
        self._render_worker = RenderWorker(self)
        self._render_worker.frame_ready.connect(self._on_frame_ready)
        self._render_worker.render_error.connect(self._on_render_error)
        self._render_worker.start()

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

        # Panning preview state — when renders are slow, step to coarser pyramid
        # levels during drag and restore full quality on mouse release.
        # _pan_preview_offset: how many levels coarser than _level_idx to use
        # during preview.  Starts at 1, increases while preview frames still
        # exceed 8 ms, and is locked in once a fast enough level is found.
        # Reset when the user changes level, FOV size, or loads a new dataset.
        self._is_panning          = False
        self._last_render_ms      = 0.0   # elapsed of most recent full-res render
        self._pan_preview_offset  = 1     # levels coarser than _level_idx for preview

        # Active minmax workers (channel_idx -> worker)
        self._minmax_workers: dict[int, MinMaxWorker] = {}

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        # ── Left: file browser ────────────────────────────────────────────────
        self._browser = SidebarBrowser(mode="zarr")
        self._browser.zarr_selected.connect(self._load_zarr)
        self._browser.setMinimumWidth(180)
        self._browser.setMaximumWidth(300)
        splitter.addWidget(self._browser)

        # ── Right: tabs (metadata + viewer) ──────────────────────────────────
        self._tabs = QTabWidget()
        splitter.addWidget(self._tabs)

        self._build_metadata_tab()
        self._build_viewer_tab()

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    # ── Metadata tab ──────────────────────────────────────────────────────────

    def _build_metadata_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(6)

        # Dataset info
        self._info_group = QGroupBox("Dataset Info")
        info_layout = QVBoxLayout(self._info_group)
        info_layout.setContentsMargins(6, 22, 6, 6)
        self._info_label = QLabel("No dataset loaded")
        self._info_label.setWordWrap(True)
        info_layout.addWidget(self._info_label)
        content_layout.addWidget(self._info_group)

        # Pixel sizes table
        ps_group = QGroupBox("Axes and Pixel Sizes")
        ps_layout = QVBoxLayout(ps_group)
        ps_layout.setContentsMargins(6, 22, 6, 6)
        self._ps_table = QTableWidget(0, 4)
        self._ps_table.setHorizontalHeaderLabels(["Axis", "Type", "Size", "Unit"])
        self._ps_table.horizontalHeader().setStretchLastSection(True)
        self._ps_table.setMaximumHeight(180)
        self._ps_table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        ps_layout.addWidget(self._ps_table)

        ps_save_row = QHBoxLayout()
        self._ps_confirm_cb = QCheckBox("Confirm")
        self._ps_confirm_cb.setToolTip("Check to enable save")
        ps_save_row.addWidget(self._ps_confirm_cb)
        self._save_ps_btn = QPushButton("Save Pixel Sizes")
        self._save_ps_btn.setEnabled(False)
        self._save_ps_btn.clicked.connect(self._on_save_pixel_sizes)
        self._ps_confirm_cb.toggled.connect(self._save_ps_btn.setEnabled)
        ps_save_row.addWidget(self._save_ps_btn)
        ps_layout.addLayout(ps_save_row)
        self._ps_status_lbl = QLabel("")
        ps_layout.addWidget(self._ps_status_lbl)

        content_layout.addWidget(ps_group)

        # Pyramid layers (read-only tree) — given most vertical space
        self._pyramid_group = QGroupBox("Pyramid Layers")
        pyr_layout = QVBoxLayout(self._pyramid_group)
        pyr_layout.setContentsMargins(6, 22, 6, 6)
        self._pyr_tree = QTreeWidget()
        self._pyr_tree.setHeaderLabels(["Property", "Value"])
        self._pyr_tree.setColumnWidth(0, 120)
        self._pyr_tree.header().setStretchLastSection(True)
        self._pyr_tree.setAlternatingRowColors(True)
        self._pyr_tree.setMinimumHeight(300)
        self._pyr_tree.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        pyr_layout.addWidget(self._pyr_tree)
        self._pyramid_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        content_layout.addWidget(self._pyramid_group, stretch=1)

        content_layout.addStretch()
        scroll.setWidget(content)
        self._tabs.addTab(scroll, "Metadata")

    # ── Viewer controls tab ───────────────────────────────────────────────────

    def _build_viewer_tab(self):
        # The Viewer tab is a horizontal splitter: [controls scroll | image+status]
        viewer_splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left: scrollable controls ─────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(260)
        scroll.setMaximumWidth(380)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(6)

        # Navigation
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

        # FOV size
        fov_row = QHBoxLayout()
        fov_row.addWidget(QLabel("FOV size:"))
        self._fov_combo = QComboBox()
        for s in FOV_SIZES:
            self._fov_combo.addItem(str(s), s)
        self._fov_combo.setCurrentIndex(2)  # 512
        self._fov_combo.currentIndexChanged.connect(self._on_fov_changed)
        fov_row.addWidget(self._fov_combo)
        nav_layout.addLayout(fov_row)

        # Orientation
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

        # T slider
        t_row = QHBoxLayout()
        t_row.addWidget(QLabel("T:"))
        self._t_label = QLabel("0")
        self._t_label.setFixedWidth(35)
        t_row.addWidget(self._t_label)
        nav_layout.addLayout(t_row)
        self._t_slider = QSlider(Qt.Orientation.Horizontal)
        self._t_slider.setMinimum(0)
        self._t_slider.setMaximum(0)
        self._t_slider.valueChanged.connect(self._on_t_changed)
        nav_layout.addWidget(self._t_slider)

        # Z / slice slider
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

        # Channel metadata (labels + intensity controls)
        ch_meta_group = QGroupBox("Channel Metadata")
        ch_meta_layout = QVBoxLayout(ch_meta_group)
        ch_meta_layout.setContentsMargins(6, 22, 6, 6)
        self._channel_panel = ChannelPanel()
        self._channel_panel.channels_changed.connect(self._on_channels_changed)
        self._channel_panel.auto_requested.connect(self._on_auto_requested)
        self._channel_panel.setMinimumHeight(120)
        ch_meta_layout.addWidget(self._channel_panel)
        ch_save_row = QHBoxLayout()
        self._ch_confirm_cb = QCheckBox("Confirm")
        self._ch_confirm_cb.setToolTip("Check to enable save")
        ch_save_row.addWidget(self._ch_confirm_cb)
        self._save_ch_btn = QPushButton("Save Channels")
        self._save_ch_btn.setEnabled(False)
        self._save_ch_btn.clicked.connect(self._on_save_channels)
        self._ch_confirm_cb.toggled.connect(self._save_ch_btn.setEnabled)
        ch_save_row.addWidget(self._save_ch_btn)
        ch_meta_layout.addLayout(ch_save_row)
        self._ch_status_lbl = QLabel("")
        self._ch_status_lbl.setStyleSheet("font-size: 10px;")
        ch_meta_layout.addWidget(self._ch_status_lbl)
        content_layout.addWidget(ch_meta_group)

        content_layout.addStretch()
        scroll.setWidget(content)
        viewer_splitter.addWidget(scroll)

        # ── Right: image widget + status bar ──────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)

        self._image_widget = ImageWidget()
        self._image_widget.pan_changed.connect(self._on_pan)
        self._image_widget.pan_released.connect(self._on_pan_released)
        self._image_widget.wheel_scrolled.connect(self._on_wheel)
        right_layout.addWidget(self._image_widget)

        self._status_bar = QLabel("")
        self._status_bar.setStyleSheet("font-size: 10px; color: #aaa; padding: 2px 4px;")
        right_layout.addWidget(self._status_bar)

        viewer_splitter.addWidget(right)
        viewer_splitter.setStretchFactor(0, 0)
        viewer_splitter.setStretchFactor(1, 1)

        self._tabs.addTab(viewer_splitter, "Viewer")

    # ── Dataset loading ───────────────────────────────────────────────────────

    def _load_zarr(self, path: str):
        try:
            pyr = pyramid_cache.get(path)
        except Exception as exc:
            self._info_label.setText(f"Error loading:\n{exc}")
            return

        self._path        = path
        self._pyr         = pyr
        self._axes        = pyr.axes.lower()
        self._shape       = list(pyr.shape)
        self._level_paths = list(pyr.meta.resolution_paths)

        self._pixel_sizes = self._read_pixel_sizes(path)
        self._channels    = self._read_channels(path, pyr)

        self._init_sliders()
        self._populate_metadata_tab()
        self._channel_panel.set_channels(self._channels)

        self.status_changed.emit(path)
        self._pan_preview_offset = 1   # re-profile for the new dataset
        self._last_render_ms = 0.0
        self._schedule_render()

    def _dim(self, ax: str) -> int:
        idx = self._axes.find(ax)
        return self._shape[idx] if idx >= 0 else 1

    def _init_sliders(self):
        # Always reset to XY so all slider ranges and FOV centres start correct
        self._orientation = "XY"
        for o in ("XY", "XZ", "YZ"):
            getattr(self, f"_ori_btn_{o}").setChecked(o == "XY")
        self._z_axis_label.setText("Z:")

        n_levels = len(self._level_paths)
        self._zoom_slider.blockSignals(True)
        self._zoom_slider.setMaximum(max(0, n_levels - 1))
        self._zoom_slider.setValue(0)
        self._zoom_slider.blockSignals(False)
        self._zoom_label.setText(f"0 / {n_levels - 1}")
        self._level_idx = 0

        t_max = max(0, self._dim("t") - 1)
        self._t_slider.blockSignals(True)
        self._t_slider.setMaximum(t_max)
        self._t_slider.setValue(t_max // 2)
        self._t_slider.blockSignals(False)
        self._t_slider.setVisible(t_max > 0)

        # Depth slider: always the through-plane axis for the current orientation (Z for XY)
        v_ax, h_ax, through_ax, _ = _ORI[self._orientation]
        depth_max = max(0, self._dim(through_ax) - 1)
        self._z_slider.blockSignals(True)
        self._z_slider.setMaximum(depth_max)
        self._z_slider.setValue(depth_max // 2)
        self._z_slider.blockSignals(False)
        self._z_slider.setVisible(depth_max > 0)

        self._fov_center_y = self._dim(v_ax) // 2   # height-axis centre
        self._fov_center_x = self._dim(h_ax) // 2   # width-axis centre
        self._t = self._t_slider.value()
        self._z = self._z_slider.value()

    # ── Metadata tab population ───────────────────────────────────────────────

    def _populate_metadata_tab(self):
        pyr    = self._pyr
        n_lev  = len(self._level_paths)
        axes   = self._axes.upper()
        shape  = "×".join(str(s) for s in self._shape)

        # Base layer info for summary
        base_layer = pyr.layers.get(self._level_paths[0]) if self._level_paths else None
        dtype_str  = str(base_layer.dtype) if base_layer is not None else "?"
        comp_info  = _compressor_info(base_layer) if base_layer is not None else {"name": "none", "params": {}}
        comp_str   = _fmt_compressor(comp_info)

        self._info_label.setText(
            f"Axes:   {axes}\n"
            f"Shape:  {shape}\n"
            f"Dtype:  {dtype_str}\n"
            f"Levels: {n_lev}\n"
            f"Compression: {comp_str}\n"
            f"Path:   {self._path}"
        )

        # ── Axes & Pixel Sizes table ──────────────────────────────────────────
        axes_meta = self._pixel_sizes
        self._ps_table.setRowCount(len(axes_meta))
        ax_types = {"t": "time", "c": "channel", "z": "space", "y": "space", "x": "space"}
        for row, ps in enumerate(axes_meta):
            ax   = ps["axis"]
            size = ps.get("value", 1.0)
            unit = ps.get("unit", "micrometer" if ax in ("z", "y", "x") else "second")

            self._ps_table.setItem(row, 0, QTableWidgetItem(ax))
            self._ps_table.setItem(row, 1, QTableWidgetItem(ax_types.get(ax, "space")))

            size_item = QTableWidgetItem(str(size))
            size_item.setFlags(size_item.flags() | Qt.ItemFlag.ItemIsEditable)
            self._ps_table.setItem(row, 2, size_item)

            unit_combo = QComboBox()
            units = _TIME_UNITS if ax == "t" else _SPACE_UNITS
            unit_combo.addItems(units)
            idx = unit_combo.findText(unit)
            if idx >= 0:
                unit_combo.setCurrentIndex(idx)
            self._ps_table.setCellWidget(row, 3, unit_combo)

        is_local = not self._path.startswith(("s3://", "gs://", "http://", "https://"))
        # Confirm checkboxes enabled only for local paths; reset to unchecked on each load
        for cb, btn in [
            (self._ps_confirm_cb, self._save_ps_btn),
            (self._ch_confirm_cb, self._save_ch_btn),
        ]:
            cb.setEnabled(is_local)
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
            btn.setEnabled(False)

        # ── Pyramid Layers tree ───────────────────────────────────────────────
        self._pyr_tree.clear()
        ax_list  = list(self._axes)         # e.g. ['t','c','z','y','x']
        size_map = {ps["axis"]: ps for ps in self._pixel_sizes}

        # Try to get per-level scales from pyramid metadata
        try:
            level_scales = {lp: pyr.meta.get_scale(lp) for lp in self._level_paths}
        except Exception:
            level_scales = {}

        for i, lp in enumerate(self._level_paths):
            layer = pyr.layers.get(lp)
            if layer is None:
                continue

            lshape  = list(layer.shape)
            lchunks = list(layer.chunks) if hasattr(layer, "chunks") and layer.chunks else []
            lcomp   = _compressor_info(layer)
            ldtype  = str(layer.dtype)
            raw_scales = level_scales.get(lp)

            root = QTreeWidgetItem(self._pyr_tree, [f"Level {i}  ({lp})", ""])
            root.setExpanded(i == 0)

            # ── Axes table: columns = axes, rows = chunk/shape/pixel size/unit ──
            n_cols  = len(ax_list)
            row_labels = ["chunk", "shape", "pixel size", "unit"]
            tbl = QTableWidget(len(row_labels), n_cols)
            tbl.setHorizontalHeaderLabels([ax.upper() for ax in ax_list])
            tbl.setVerticalHeaderLabels(row_labels)
            tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            tbl.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
            tbl.horizontalHeader().setDefaultSectionSize(54)
            tbl.verticalHeader().setDefaultSectionSize(20)
            tbl.verticalHeader().setFixedWidth(72)

            shards = lcomp.get("params", {}).get("shard_shape") or lcomp.get("params", {}).get("shards")

            for j, ax in enumerate(ax_list):
                chunk_val = str(lchunks[j]) if j < len(lchunks) else "—"
                shape_val = str(lshape[j])  if j < len(lshape)  else "—"
                if raw_scales and len(raw_scales) == len(ax_list):
                    ps_val   = f"{raw_scales[j]:.6g}"
                    unit_val = size_map.get(ax, {}).get("unit", "")
                else:
                    ps_val   = "—"
                    unit_val = "—"
                for row_i, val in enumerate([chunk_val, shape_val, ps_val, unit_val]):
                    cell = QTableWidgetItem(val)
                    cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    tbl.setItem(row_i, j, cell)

            # Size table to fit without scrollbars
            tbl.resizeColumnsToContents()
            hdr_h  = tbl.horizontalHeader().height()
            rows_h = sum(tbl.rowHeight(r) for r in range(tbl.rowCount()))
            tbl.setFixedHeight(hdr_h + rows_h + 4)
            tbl.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            tbl.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

            tbl_item = QTreeWidgetItem(root)
            self._pyr_tree.setItemWidget(tbl_item, 0, tbl)
            root_idx = self._pyr_tree.indexFromItem(root)
            self._pyr_tree.setFirstColumnSpanned(0, root_idx, True)

            # ── Shards (zarr v3) ──────────────────────────────────────────────
            if shards:
                shard_str = "×".join(str(s) for s in shards) if hasattr(shards, "__len__") else str(shards)
                _tree_row(root, "shards", shard_str)

            # ── dtype + compression ───────────────────────────────────────────
            _tree_row(root, "dtype", ldtype)
            _tree_row(root, "compression", _fmt_compressor(lcomp))
            if lcomp.get("params"):
                comp_node = _tree_row(root, "codec params", "")
                for k, v in lcomp["params"].items():
                    if k not in ("shard_shape", "shards"):
                        QTreeWidgetItem(comp_node, [f"  {k}", str(v)])

    # ── Pixel size helpers ────────────────────────────────────────────────────

    def _read_pixel_sizes(self, path: str) -> list[dict]:
        try:
            for fname in [".zattrs", "zarr.json"]:
                fpath = os.path.join(path, fname)
                if not os.path.exists(fpath):
                    continue
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
                attrs = data.get("attributes", data)
                ome   = attrs.get("ome", attrs)
                ms_list = ome.get("multiscales", attrs.get("multiscales", []))
                if not ms_list:
                    continue
                ms = ms_list[0]
                axes = ms.get("axes", [])
                datasets = ms.get("datasets", [])
                if not datasets:
                    continue
                ct = datasets[0].get("coordinateTransformations", [])
                for xform in ct:
                    if xform.get("type") == "scale":
                        scales = xform.get("scale", [])
                        return [
                            {
                                "axis": ax.get("name", "?"),
                                "value": float(s),
                                "unit": ax.get("unit", "micrometer"),
                            }
                            for ax, s in zip(axes, scales)
                        ]
        except Exception:
            pass
        # Fallback: one entry per axis from pyr
        return [{"axis": ax, "value": 1.0, "unit": "micrometer" if ax in ("z","y","x") else "second"}
                for ax in self._axes]

    def _read_channels(self, path: str, pyr) -> list[dict]:
        axes = pyr.axes.lower()
        c_idx = axes.find("c")
        n_ch  = pyr.shape[c_idx] if c_idx >= 0 else 1
        defaults = [
            {
                "index": i,
                "label": f"Ch {i}",
                "color": _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)],
                "visible": True,
                "intensityMin": None,
                "intensityMax": None,
            }
            for i in range(n_ch)
        ]
        try:
            for fname in [".zattrs", "zarr.json"]:
                fpath = os.path.join(path, fname)
                if not os.path.exists(fpath):
                    continue
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
                attrs  = data.get("attributes", data)
                omero  = attrs.get("omero")
                if omero is None:
                    continue
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
                break
        except Exception:
            pass
        return defaults

    # ── Save metadata ─────────────────────────────────────────────────────────

    def _on_save_pixel_sizes(self):
        if not self._path:
            return
        scales = []
        for row in range(self._ps_table.rowCount()):
            ax_item   = self._ps_table.item(row, 0)
            size_item = self._ps_table.item(row, 2)
            unit_w    = self._ps_table.cellWidget(row, 3)
            if ax_item and size_item:
                try:
                    val = float(size_item.text())
                except ValueError:
                    val = 1.0
                scales.append({
                    "axis":  ax_item.text(),
                    "value": val,
                    "unit":  unit_w.currentText() if unit_w else "micrometer",
                })
        try:
            _update_scales_on_disk(self._path, scales)
            # Invalidate pyramid cache (mirrors /update_scales endpoint)
            with pyramid_cache._lock:
                pyramid_cache._cache.pop(self._path, None)
            self._pixel_sizes = scales
            # Reload pyramid and refresh pyramid layers tree immediately
            self._pyr = pyramid_cache.get(self._path)
            self._populate_metadata_tab()
            self._show_save_status(self._ps_status_lbl, self._ps_confirm_cb, self._save_ps_btn, ok=True)
        except Exception as exc:
            self._show_save_status(self._ps_status_lbl, self._ps_confirm_cb, self._save_ps_btn, ok=False, msg=str(exc))

    def _on_save_channels(self):
        if not self._path:
            return
        try:
            _update_channels_on_disk(self._path, self._channels)
            self._show_save_status(self._ch_status_lbl, self._ch_confirm_cb, self._save_ch_btn, ok=True)
        except Exception as exc:
            self._show_save_status(self._ch_status_lbl, self._ch_confirm_cb, self._save_ch_btn, ok=False, msg=str(exc))

    def _show_save_status(self, lbl: "QLabel", cb: "QCheckBox", btn: "QPushButton",
                          ok: bool, msg: str = ""):
        if ok:
            lbl.setText("✓ Saved")
            lbl.setStyleSheet("font-size: 10px; color: #6f6;")
        else:
            lbl.setText(f"✗ {msg[:80]}")
            lbl.setStyleSheet("font-size: 10px; color: #f66;")
        # Reset confirmation state
        cb.blockSignals(True)
        cb.setChecked(False)
        cb.blockSignals(False)
        btn.setEnabled(False)
        # Auto-clear message after 4 s
        QTimer.singleShot(4000, lambda: lbl.setText(""))

    # ── Viewer control callbacks ──────────────────────────────────────────────

    def _on_zoom_changed(self, value: int):
        self._level_idx = value
        n = len(self._level_paths)
        self._zoom_label.setText(f"{value} / {max(0, n - 1)}")
        self._pan_preview_offset = 1   # re-profile at the new level
        self._last_render_ms = 0.0
        self._schedule_render()

    def _on_fov_changed(self, _idx: int):
        self._fov_size = self._fov_combo.currentData()
        self._pan_preview_offset = 1   # re-profile at the new FOV size
        self._last_render_ms = 0.0
        self._schedule_render()

    def _on_orientation(self, ori: str):
        self._orientation = ori
        for o in ("XY", "XZ", "YZ"):
            getattr(self, f"_ori_btn_{o}").setChecked(o == ori)

        v_ax, h_ax, through_ax, through_lbl = _ORI[ori]
        self._z_axis_label.setText(f"{through_lbl}:")

        # Reconfigure the through-plane slider for the new axis
        ax_max = max(0, self._dim(through_ax) - 1)
        self._z_slider.blockSignals(True)
        self._z_slider.setMaximum(ax_max)
        mid = ax_max // 2
        self._z_slider.setValue(mid)
        self._z_slider.blockSignals(False)
        self._z = mid
        self._z_label.setText(str(mid))

        # Reset FOV center to the middle of the new view plane
        self._fov_center_y = self._dim(v_ax) // 2
        self._fov_center_x = self._dim(h_ax) // 2
        self._schedule_render()

    def _on_fit(self):
        v_ax, h_ax, _, _ = _ORI[self._orientation]
        self._fov_center_y = self._dim(v_ax) // 2
        self._fov_center_x = self._dim(h_ax) // 2
        self._schedule_render()

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

    def _on_auto_requested(self, channel_idx: int):
        if not self._path:
            return
        # Cancel any existing worker for this channel
        if channel_idx in self._minmax_workers:
            self._minmax_workers[channel_idx].quit()
        worker = MinMaxWorker(self._path, channel_idx, self)
        worker.result.connect(self._on_minmax_result)
        worker.failed.connect(lambda idx, msg: self._status_bar.setText(f"Auto ch{idx}: {msg}"))
        self._minmax_workers[channel_idx] = worker
        worker.start()

    def _on_minmax_result(self, channel_idx: int, vmin: float, vmax: float):
        self._channel_panel.set_channel_range(channel_idx, vmin, vmax)
        # channels_changed will fire from within set_channel_range via ChannelPanel
        self._minmax_workers.pop(channel_idx, None)

    def _on_pan(self, delta_row: float, delta_col: float):
        self._is_panning = True
        v_ax, h_ax, _, _ = _ORI[self._orientation]
        v_scale, h_scale = self._current_level_scales()
        # delta_row/col are in current-level pixel space; convert to world (level-0) coords
        step_y = delta_row * v_scale
        step_x = delta_col * h_scale
        # Clamp center to [fov_half_world, dim - fov_half_world] so center stops
        # exactly where the image visually hits the edge at the current level.
        half_y = (self._fov_size / 2.0) * v_scale
        half_x = (self._fov_size / 2.0) * h_scale
        y_lo, y_hi = half_y, max(half_y, float(self._dim(v_ax)) - half_y)
        x_lo, x_hi = half_x, max(half_x, float(self._dim(h_ax)) - half_x)
        self._fov_center_y = max(y_lo, min(y_hi, self._fov_center_y + step_y))
        self._fov_center_x = max(x_lo, min(x_hi, self._fov_center_x + step_x))
        # Rate-limit renders: accumulate pan deltas freely, but only queue one
        # _trigger_render per event-loop cycle. Without this, fast mouse moves
        # fire _trigger_render faster than renders complete → _render_gen races
        # ahead → every in-flight frame is discarded → visible jump at end of drag.
        if not self._pan_render_timer.isActive():
            self._pan_render_timer.start(0)

    def _on_pan_released(self):
        """Mouse button up: cancel any queued preview and render full resolution."""
        self._pan_render_timer.stop()
        self._is_panning = False
        self._trigger_render()

    def _current_level_scales(self) -> tuple:
        """(v_scale, h_scale): world-pixels / current-level-pixel for each spatial axis.
        Cheap — get_orientation_axes results are memoized in zarr_plane_server.
        """
        if not self._pyr or not self._level_paths:
            return 1.0, 1.0
        level_path = self._level_paths[min(self._level_idx, len(self._level_paths) - 1)]
        try:
            from zarr_plane_server import get_orientation_axes  # already on sys.path via render_worker
            info = get_orientation_axes(self._pyr, level_path, self._orientation)
            return info['v_scale'], info['h_scale']
        except Exception:
            return 1.0, 1.0

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

        level_idx = max(0, min(self._level_idx, len(self._level_paths) - 1))
        fov_size  = self._fov_size

        # Coarse-level preview during drag: use _pan_preview_offset levels coarser
        # than the selected level.  The offset is raised by _on_frame_ready while
        # preview frames are still slower than 8 ms, then locked in once fast enough.
        target_fov_size = 0   # 0 means no upsampling needed
        if self._is_panning and self._last_render_ms > 8.0:
            candidate = min(level_idx + self._pan_preview_offset,
                            len(self._level_paths) - 1)
            if candidate > level_idx:
                try:
                    from zarr_plane_server import get_orientation_axes
                    cur_info = get_orientation_axes(
                        self._pyr, self._level_paths[level_idx], self._orientation)
                    prv_info = get_orientation_axes(
                        self._pyr, self._level_paths[candidate], self._orientation)
                    ratio = cur_info['v_scale'] / prv_info['v_scale']
                    target_fov_size = fov_size
                    fov_size  = max(64, int(self._fov_size * ratio))
                    level_idx = candidate
                except Exception:
                    target_fov_size = 0  # fall back to full-res on any error

        v_ax, h_ax, _, _ = _ORI[self._orientation]
        v_scale, h_scale = self._current_level_scales()
        half_y = int(self._fov_size / 2 * v_scale)
        half_x = int(self._fov_size / 2 * h_scale)
        center_y = max(half_y, min(self._dim(v_ax) - half_y, round(self._fov_center_y)))
        center_x = max(half_x, min(self._dim(h_ax) - half_x, round(self._fov_center_x)))

        self._render_worker.request({
            "path":            self._path,
            "level_idx":       level_idx,
            "orientation":     self._orientation,
            "indices":         indices,
            "fov_center":      (center_y, center_x),
            "fov_size":        fov_size,
            "target_fov_size": target_fov_size,
            "channels":        [dict(ch) for ch in self._channels],
        })
        self._render_gen = self._render_worker._generation

    def _on_frame_ready(self, rgb: np.ndarray, elapsed_ms: float, generation: int):
        # Discard stale frames from superseded render requests.
        if generation < self._render_gen:
            return
        if not self._is_panning:
            # Full-res render completed — update the threshold used to decide
            # whether a preview is needed on the next drag.
            self._last_render_ms = elapsed_ms
        elif elapsed_ms > 8.0:
            # Preview frame was still too slow: step one level coarser and
            # immediately fire another render so the user sees the convergence
            # within the current drag rather than waiting for the next one.
            max_offset = len(self._level_paths) - 1 - self._level_idx
            if self._pan_preview_offset < max_offset:
                self._pan_preview_offset += 1
                if not self._pan_render_timer.isActive():
                    self._pan_render_timer.start(0)
        # else: preview was fast enough — offset is already correct, keep it.
        aspect = self._compute_pixel_aspect()
        self._image_widget.set_frame(rgb, aspect, elapsed_ms)
        self._update_status_bar(elapsed_ms)

    def _on_render_error(self, msg: str):
        self._status_bar.setText(f"Render error: {msg[:120]}")

    def _compute_pixel_aspect(self) -> float:
        if not self._pixel_sizes:
            return 1.0
        size_map = {ps["axis"]: ps.get("value", 1.0) for ps in self._pixel_sizes}
        v_ax, h_ax, _, _ = _ORI[self._orientation]
        row_sz = size_map.get(v_ax, 1.0)
        col_sz = size_map.get(h_ax, 1.0)
        if col_sz <= 0:
            return 1.0
        ratio = row_sz / col_sz
        return ratio if 0.01 < ratio < 100 else 1.0

    def _update_status_bar(self, elapsed_ms: float):
        if not self._pyr:
            return
        level_path = self._level_paths[self._level_idx] if self._level_idx < len(self._level_paths) else "?"
        layer = self._pyr.layers.get(level_path)
        shape_str = str(list(layer.shape)) if layer else "?"
        aspect = self._compute_pixel_aspect()
        ar_str = f" · AR {aspect:.3f}" if abs(aspect - 1.0) > 0.01 else ""
        _, _, _, through_lbl = _ORI[self._orientation]
        self._status_bar.setText(
            f"Level {self._level_idx}  {shape_str}  "
            f"{self._orientation}  {through_lbl}={self._z}  "
            f"center=({round(self._fov_center_y)},{round(self._fov_center_x)})"
            f"  {elapsed_ms:.0f}ms{ar_str}"
        )

    def closeEvent(self, _event):
        self._render_worker.stop()
        self._render_worker.wait(2000)


# ── Metadata file-write helpers ───────────────────────────────────────────────

def _load_zattrs(path: str) -> tuple[dict, str]:
    """Load the metadata file and return (data, filepath)."""
    for fname in [".zattrs", "zarr.json"]:
        fpath = os.path.join(path, fname)
        if os.path.exists(fpath):
            with open(fpath, encoding="utf-8") as f:
                return json.load(f), fpath
    raise FileNotFoundError(f"No .zattrs or zarr.json found in {path}")


def _save_zattrs(fpath: str, data: dict):
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _update_scales_on_disk(zarr_path: str, scales: list[dict]):
    """Write updated pixel sizes using the Pyramid class (mirrors /update_scales endpoint)."""
    from eubi_bridge.ngff.multiscales import Pyramid  # type: ignore
    pyr = Pyramid(gr=zarr_path)
    scale_kwargs = {s["axis"]: s["value"] for s in scales}
    unit_kwargs  = {s["axis"]: s["unit"]  for s in scales}
    pyr.update_scales(**scale_kwargs)
    pyr.update_units(**unit_kwargs)
    pyr.meta.save_changes()


def _update_channels_on_disk(zarr_path: str, channels: list[dict]):
    """Write updated channel metadata to .zattrs / zarr.json."""
    data, fpath = _load_zattrs(zarr_path)
    is_v3 = fpath.endswith("zarr.json")
    attrs = data.get("attributes", data) if is_v3 else data

    omero = attrs.get("omero", {})
    existing = omero.get("channels", [])

    # Extend existing list if needed
    while len(existing) < len(channels):
        existing.append({})

    for i, ch in enumerate(channels):
        color = ch.get("color", "#FFFFFF").lstrip("#")
        existing[i]["label"]  = ch.get("label", f"Ch {i}")
        existing[i]["color"]  = color
        existing[i]["active"] = ch.get("visible", True)
        existing[i]["window"] = {
            "min":   ch.get("intensityMin", 0) or 0,
            "max":   ch.get("intensityMax", 65535) or 65535,
            "start": ch.get("intensityMin", 0) or 0,
            "end":   ch.get("intensityMax", 65535) or 65535,
        }

    omero["channels"] = existing
    attrs["omero"] = omero
    if is_v3:
        data["attributes"] = attrs
    _save_zattrs(fpath, data)
