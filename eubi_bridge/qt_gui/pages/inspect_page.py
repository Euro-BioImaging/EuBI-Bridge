"""
Inspect page — OME-Zarr viewer + metadata editor.

Two-pane layout:
  Left  : SidebarBrowser (zarr mode) — navigate filesystem, click OME-Zarr to load
  Right : QTabWidget
            Metadata tab — dataset info, pixel sizes, pyramid layers (read-only tree)
            Viewer tab   — ZarrViewer + status bar
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
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from eubi_bridge.qt_gui.widgets.sidebar_browser import SidebarBrowser
from eubi_bridge.qt_gui.widgets.zarr_viewer import ZarrViewer, _ORI

_SERVER_DIR = os.path.join(os.path.dirname(__file__), "..", "server")
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

from zarr_plane_server import pyramid_cache, _compressor_info  # type: ignore

# ── Unit lists ────────────────────────────────────────────────────────────────
_SPACE_UNITS = ["picometer", "nanometer", "micrometer", "millimeter", "centimeter", "meter", "kilometer"]
_TIME_UNITS  = ["nanosecond", "microsecond", "millisecond", "second", "minute", "hour"]


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


class InspectPage(QWidget):
    """Full inspect / viewer page."""

    status_changed = pyqtSignal(str)   # for main window status bar

    def __init__(self, parent=None):
        super().__init__(parent)

        # Per-page state that is not part of ZarrViewer
        self._path         = ""
        self._pyr          = None
        self._pixel_sizes: list[dict] = []

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
        self._ps_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
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
        self._pyr_tree.header().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self._pyr_tree.header().setStretchLastSection(True)
        self._pyr_tree.setColumnWidth(0, 120)
        self._pyr_tree.setAlternatingRowColors(True)
        self._pyr_tree.setMinimumHeight(300)
        self._pyr_tree.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        pyr_layout.addWidget(self._pyr_tree)
        self._pyramid_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        content_layout.addWidget(self._pyramid_group, stretch=1)

        content_layout.addStretch()
        scroll.setWidget(content)
        self._tabs.addTab(scroll, "Metadata")

    # ── Viewer tab ────────────────────────────────────────────────────────────

    def _build_viewer_tab(self):
        # The Viewer tab is a vertical splitter: [ZarrViewer | status bar]
        # (The status bar is kept in InspectPage because it shows pixel-aspect info
        # that comes from pixel-size metadata, which ZarrViewer does not own.)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._viewer = ZarrViewer(
            self,
            enable_minmax=True,
            show_fov_combo=True,
            show_reload_btn=True,
        )
        # Wire up the channel-save row next to the ChannelPanel (InspectPage-only feature)
        self._viewer.channel_panel.channels_changed.connect(self._on_channels_changed_for_save)

        self._viewer.frame_ready.connect(self._on_frame_ready)
        self._viewer.dataset_loaded.connect(self._on_dataset_loaded)
        self._viewer.status_changed.connect(self.status_changed)
        layout.addWidget(self._viewer)

        # Channel-save controls (below the ZarrViewer)
        ch_save_group = QGroupBox("Save Channel Metadata")
        ch_save_layout = QHBoxLayout(ch_save_group)
        ch_save_layout.setContentsMargins(6, 22, 6, 6)
        self._ch_confirm_cb = QCheckBox("Confirm")
        self._ch_confirm_cb.setToolTip("Check to enable save")
        ch_save_layout.addWidget(self._ch_confirm_cb)
        self._save_ch_btn = QPushButton("Save Channels")
        self._save_ch_btn.setEnabled(False)
        self._save_ch_btn.clicked.connect(self._on_save_channels)
        self._ch_confirm_cb.toggled.connect(self._save_ch_btn.setEnabled)
        ch_save_layout.addWidget(self._save_ch_btn)
        self._ch_status_lbl = QLabel("")
        self._ch_status_lbl.setStyleSheet("font-size: 10px;")
        ch_save_layout.addWidget(self._ch_status_lbl)
        ch_save_layout.addStretch()
        layout.addWidget(ch_save_group)

        self._status_bar = QLabel("")
        self._status_bar.setStyleSheet("font-size: 10px; color: #aaa; padding: 2px 4px;")
        layout.addWidget(self._status_bar)

        self._tabs.addTab(container, "Viewer")

    # ── Dataset loading ───────────────────────────────────────────────────────

    def _load_zarr(self, path: str):
        """Called by SidebarBrowser; delegates actual loading to ZarrViewer."""
        # Read pixel sizes before delegating (used in _on_dataset_loaded)
        self._path = path
        self._pixel_sizes = self._read_pixel_sizes(path)
        self._viewer.load_dataset(path)

    def _on_dataset_loaded(self, pyr):
        """Called when ZarrViewer has finished loading the pyramid."""
        self._pyr = pyr
        self._path = self._viewer.path
        self._populate_metadata_tab()

    def _on_channels_changed_for_save(self, _channels: list):
        """Re-connect so the Save Channels button reflects unsaved changes."""
        # No action needed here; the viewer already tracks channels internally.
        pass

    # ── Metadata tab population ───────────────────────────────────────────────

    def _populate_metadata_tab(self):
        pyr         = self._pyr
        level_paths = self._viewer.level_paths
        axes        = self._viewer.axes
        shape       = self._viewer.shape

        n_lev     = len(level_paths)
        axes_upper = axes.upper()
        shape_str  = "×".join(str(s) for s in shape)

        base_layer = pyr.layers.get(level_paths[0]) if level_paths else None
        dtype_str  = str(base_layer.dtype) if base_layer is not None else "?"
        comp_info  = _compressor_info(base_layer) if base_layer is not None else {"name": "none", "params": {}}
        comp_str   = _fmt_compressor(comp_info)

        ngff_ver  = getattr(pyr.meta, "version", "?")
        zarr_fmt  = getattr(pyr.meta, "zarr_format", "?")
        fmt_str   = f"OME-NGFF v{ngff_ver}  (zarr v{zarr_fmt})"

        self._info_label.setText(
            f"Format: {fmt_str}\n"
            f"Axes:   {axes_upper}\n"
            f"Shape:  {shape_str}\n"
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

            from PyQt6.QtCore import Qt as _Qt
            size_item = QTableWidgetItem(str(size))
            size_item.setFlags(size_item.flags() | _Qt.ItemFlag.ItemIsEditable)
            self._ps_table.setItem(row, 2, size_item)

            unit_combo = QComboBox()
            units = _TIME_UNITS if ax == "t" else _SPACE_UNITS
            unit_combo.addItems(units)
            idx = unit_combo.findText(unit)
            if idx >= 0:
                unit_combo.setCurrentIndex(idx)
            self._ps_table.setCellWidget(row, 3, unit_combo)

        is_local = not self._path.startswith(("s3://", "gs://", "http://", "https://"))
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
        ax_list  = list(axes)
        size_map = {ps["axis"]: ps for ps in self._pixel_sizes}

        try:
            level_scales = {lp: pyr.meta.get_scale(lp) for lp in level_paths}
        except Exception:
            level_scales = {}

        for i, lp in enumerate(level_paths):
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

            shards = getattr(layer, "shards", None)
            n_cols     = len(ax_list)
            row_labels = ["chunk", "nchunks/shard", "shape", "pixel size", "unit"]
            tbl = QTableWidget(len(row_labels), n_cols)
            tbl.setHorizontalHeaderLabels([ax.upper() for ax in ax_list])
            tbl.setVerticalHeaderLabels(row_labels)
            tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            tbl.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
            tbl.horizontalHeader().setDefaultSectionSize(54)
            tbl.verticalHeader().setDefaultSectionSize(20)
            tbl.verticalHeader().setFixedWidth(96)

            from PyQt6.QtCore import Qt as _Qt2
            for j, ax in enumerate(ax_list):
                chunk_val = str(lchunks[j]) if j < len(lchunks) else "—"
                if shards and j < len(shards) and j < len(lchunks) and lchunks[j]:
                    ncs_val = str(shards[j] // lchunks[j])
                else:
                    ncs_val = "—"
                shape_val = str(lshape[j])  if j < len(lshape)  else "—"
                if raw_scales and len(raw_scales) == len(ax_list):
                    ps_val   = f"{raw_scales[j]:.6g}"
                    unit_val = size_map.get(ax, {}).get("unit", "")
                else:
                    ps_val   = "—"
                    unit_val = "—"
                for row_i, val in enumerate([chunk_val, ncs_val, shape_val, ps_val, unit_val]):
                    cell = QTableWidgetItem(val)
                    cell.setTextAlignment(_Qt2.AlignmentFlag.AlignCenter)
                    tbl.setItem(row_i, j, cell)

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

            _tree_row(root, "dtype", ldtype)
            _tree_row(root, "compression", _fmt_compressor(lcomp))
            if lcomp.get("params"):
                comp_node = _tree_row(root, "codec params", "")
                for k, v in lcomp["params"].items():
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
                                "axis":  ax.get("name", "?"),
                                "value": float(s),
                                "unit":  ax.get("unit", "micrometer"),
                            }
                            for ax, s in zip(axes, scales)
                        ]
        except Exception:
            pass
        # Fallback
        return [
            {"axis": ax, "value": 1.0, "unit": "micrometer" if ax in ("z", "y", "x") else "second"}
            for ax in self._viewer.axes
        ]

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
            with pyramid_cache._lock:
                pyramid_cache._cache.pop(self._path, None)
            self._pixel_sizes = scales
            self._pyr = pyramid_cache.get(self._path)
            self._populate_metadata_tab()
            self._show_save_status(self._ps_status_lbl, self._ps_confirm_cb, self._save_ps_btn, ok=True)
        except Exception as exc:
            self._show_save_status(self._ps_status_lbl, self._ps_confirm_cb, self._save_ps_btn, ok=False, msg=str(exc))

    def _on_save_channels(self):
        if not self._path:
            return
        try:
            _update_channels_on_disk(self._path, self._viewer.channels)
            with pyramid_cache._lock:
                pyramid_cache._cache.pop(self._path, None)
            self._viewer.channel_panel.commit_reset_baseline()
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
        cb.blockSignals(True)
        cb.setChecked(False)
        cb.blockSignals(False)
        btn.setEnabled(False)
        QTimer.singleShot(4000, lambda: lbl.setText(""))

    # ── Frame-ready handler ───────────────────────────────────────────────────

    def _on_frame_ready(self, rgb: np.ndarray, elapsed_ms: float, generation: int):
        """Receives a rendered frame from ZarrViewer; updates image + status bar."""
        aspect = self._compute_pixel_aspect()
        self._viewer.image_widget.set_frame(rgb, aspect, elapsed_ms)
        self._update_status_bar(elapsed_ms)

    def _compute_pixel_aspect(self) -> float:
        if not self._pixel_sizes:
            return 1.0
        size_map = {ps["axis"]: ps.get("value", 1.0) for ps in self._pixel_sizes}
        v_ax, h_ax, _, _ = _ORI[self._viewer.orientation]
        row_sz = size_map.get(v_ax, 1.0)
        col_sz = size_map.get(h_ax, 1.0)
        if col_sz <= 0:
            return 1.0
        ratio = row_sz / col_sz
        return ratio if 0.01 < ratio < 100 else 1.0

    def _update_status_bar(self, elapsed_ms: float):
        if not self._pyr:
            return
        level_idx   = self._viewer.level_idx
        level_paths = self._viewer.level_paths
        level_path  = level_paths[level_idx] if level_idx < len(level_paths) else "?"
        layer       = self._pyr.layers.get(level_path)
        shape_str   = str(list(layer.shape)) if layer else "?"
        aspect      = self._compute_pixel_aspect()
        ar_str      = f" · AR {aspect:.3f}" if abs(aspect - 1.0) > 0.01 else ""
        _, _, _, through_lbl = _ORI[self._viewer.orientation]
        fov_y, fov_x = self._viewer.fov_center
        self._status_bar.setText(
            f"Level {level_idx}  {shape_str}  "
            f"{self._viewer.orientation}  {through_lbl}={self._viewer.z}  "
            f"center=({round(fov_y)},{round(fov_x)})"
            f"  {elapsed_ms:.0f}ms{ar_str}"
        )

    def closeEvent(self, _event):
        self._viewer.closeEvent(_event)


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
    """Write updated channel metadata via Pyramid so v2/v3 format is handled correctly."""
    from eubi_bridge.ngff.multiscales import Pyramid  # type: ignore
    pyr = Pyramid(gr=zarr_path)
    omero    = pyr.meta.metadata.get("omero", {})
    existing = omero.get("channels", [])

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
    pyr.meta.metadata["omero"] = omero
    pyr.meta._pending_changes = True
    pyr.meta.save_changes()
