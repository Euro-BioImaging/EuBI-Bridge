"""PropertiesPanel — right-sidebar stacked panel showing node details.

Three sub-panels:
  - FlowInfoPanel: flow-level metadata (read-only)
  - HeaveInfoPanel: heave metadata + on-disk badge
  - WaveInfoPanel: processor name, params form, output preview, Apply/Discard
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from eubi_flow.qt_gui.flow_page.params_form import ParamsForm

if TYPE_CHECKING:
    from eubi_flow.models import FlowSpec, HeaveSpec, WaveSpec


# ---------------------------------------------------------------------------
# FlowInfoPanel
# ---------------------------------------------------------------------------

class FlowInfoPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        lbl = QLabel("Flow")
        lbl.setStyleSheet("font-weight: bold; font-size: 11px; color: #aaaaaa;")
        layout.addWidget(lbl)

        self._form = QFormLayout()
        self._form.setSpacing(4)
        self._id_lbl    = QLabel("—")
        self._name_lbl  = QLabel("—")
        self._wd_lbl    = QLabel("—")
        self._wd_lbl.setWordWrap(True)
        self._stat_lbl  = QLabel("—")
        self._form.addRow("ID:",      self._id_lbl)
        self._form.addRow("Name:",    self._name_lbl)
        self._form.addRow("Workdir:", self._wd_lbl)
        self._form.addRow("Status:",  self._stat_lbl)

        wrapper = QWidget()
        wrapper.setLayout(self._form)
        layout.addWidget(wrapper)
        layout.addStretch()

    def load_flow(self, flow: "FlowSpec", name: str) -> None:
        self._id_lbl.setText(flow.flow_id)
        self._name_lbl.setText(name)
        self._wd_lbl.setText(flow.workdir or "—")
        self._stat_lbl.setText(flow.status)


# ---------------------------------------------------------------------------
# HeaveInfoPanel
# ---------------------------------------------------------------------------

def _find_heave_on_disk(
    spec_path: str,
    heave_id: str,
    flow_workdir: str | None,
) -> bool:
    """Check whether a heave has been written.

    Checks in order:
    1. Template path directly (single-zarr flows where path == output).
    2. Heave-first directory ``<flow_workdir>/**/<heave_id>/`` — the new
       layout where every heave has its own folder.
    3. Broad glob fallback for the template filename anywhere in the tree.
    """
    import glob as _glob
    if Path(spec_path).exists():
        return True
    if not flow_workdir:
        return False
    # Heave-first: workdir/flow_name/heave_id/ (non-empty dir means written)
    heave_dir_patterns = _glob.glob(
        str(Path(flow_workdir) / "**" / heave_id), recursive=True
    )
    if any(Path(p).is_dir() for p in heave_dir_patterns):
        return True
    # Fallback: filename anywhere in tree (old dataset-first layout)
    name = Path(spec_path).name
    return bool(_glob.glob(str(Path(flow_workdir) / "**" / name), recursive=True))


def _is_zarr_collection(path: str) -> bool:
    """True when *path* is a plain directory containing OME-Zarr stores."""
    from eubi_bridge.utils.path_utils import is_ome_zarr
    p = Path(path)
    return p.is_dir() and not is_ome_zarr(path)


class HeaveInfoPanel(QWidget):
    """Properties panel for a selected HeaveItem.

    Switches between two views:
    - **Metadata view** (single zarr or pending): axes, shape, dtype, scales, units.
    - **Collection view** (directory of zarrs): lists discovered OME-Zarrs;
      clicking one shows its metadata inline.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ── Header ───────────────────────────────────────────────────
        hdr = QLabel("Heave")
        hdr.setStyleSheet("font-weight: bold; font-size: 11px; color: #3ca0ff;")
        layout.addWidget(hdr)

        # heave_id
        id_row = QFormLayout()
        id_row.setSpacing(3)
        self._id_lbl = QLabel("—")
        self._id_lbl.setStyleSheet("font-weight: bold;")
        id_row.addRow("ID:", self._id_lbl)
        layout.addLayout(id_row)

        # path (read-only QLineEdit so text can be selected and copied)
        path_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setReadOnly(True)
        self._path_edit.setToolTip("Path is read-only. Select text and Ctrl+C to copy.")
        copy_btn = QPushButton("Copy")
        copy_btn.setFixedWidth(44)
        copy_btn.setFixedHeight(22)
        copy_btn.clicked.connect(self._copy_path)
        path_row.addWidget(self._path_edit)
        path_row.addWidget(copy_btn)
        path_label = QLabel("Path:")
        path_form = QFormLayout()
        path_form.setSpacing(3)
        path_form.addRow(path_label, _wrap_hbox(path_row))
        layout.addLayout(path_form)

        # disk badge
        disk_form = QFormLayout()
        disk_form.setSpacing(3)
        self._disk_lbl = QLabel("—")
        disk_form.addRow("Disk:", self._disk_lbl)
        layout.addLayout(disk_form)

        # ── Stacked content: metadata vs collection browser ───────────
        self._stack = QStackedWidget()

        # Page 0 — single zarr metadata
        meta_page = QWidget()
        mf = QFormLayout(meta_page)
        mf.setSpacing(4)
        self._axes_lbl   = QLabel("—")
        self._shape_lbl  = QLabel("—")
        self._dtype_lbl  = QLabel("—")
        self._scales_lbl = QLabel("—")
        self._scales_lbl.setWordWrap(True)
        self._units_lbl  = QLabel("—")
        self._units_lbl.setWordWrap(True)
        for lbl_text, w in [("Axes:", self._axes_lbl), ("Shape:", self._shape_lbl),
                             ("Dtype:", self._dtype_lbl), ("Scales:", self._scales_lbl),
                             ("Units:", self._units_lbl)]:
            mf.addRow(lbl_text, w)
        self._stack.addWidget(meta_page)    # index 0

        # Page 1 — collection browser
        coll_page = QWidget()
        cl = QVBoxLayout(coll_page)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(4)
        cl.addWidget(QLabel("OME-Zarr stores in this collection:"))
        self._zarr_list = QListWidget()
        self._zarr_list.setMaximumHeight(120)
        self._zarr_list.currentTextChanged.connect(self._on_zarr_selected)
        cl.addWidget(self._zarr_list)
        coll_meta_form = QFormLayout()
        coll_meta_form.setSpacing(4)
        self._c_axes_lbl   = QLabel("—")
        self._c_shape_lbl  = QLabel("—")
        self._c_dtype_lbl  = QLabel("—")
        for lbl_text, w in [("Axes:", self._c_axes_lbl),
                             ("Shape:", self._c_shape_lbl),
                             ("Dtype:", self._c_dtype_lbl)]:
            coll_meta_form.addRow(lbl_text, w)
        cl.addLayout(coll_meta_form)
        cl.addStretch()
        self._stack.addWidget(coll_page)    # index 1

        layout.addWidget(self._stack, stretch=1)

    # ------------------------------------------------------------------

    def load_heave(self, spec: "HeaveSpec", flow=None, flow_name: str | None = None) -> None:
        workdir = flow.workdir if flow else None
        self._id_lbl.setText(spec.heave_id)

        # Show the actual heave-first directory when possible, otherwise
        # the template path.  Output heaves live at workdir/flow_name/heave_id/.
        if flow and flow_name and spec.heave_id != "heave_000":
            actual = Path(flow.workdir) / flow_name / spec.heave_id
            display_path = str(actual)
        else:
            display_path = str(spec.path)
        self._path_edit.setText(display_path)

        # Use the resolved display_path for all on-disk checks and collection
        # detection — it already points to the correct heave-first directory.
        exists = Path(display_path).exists() or _find_heave_on_disk(
            str(spec.path), spec.heave_id, workdir
        )
        if exists:
            self._disk_lbl.setText("✓ on disk")
            self._disk_lbl.setStyleSheet("color: #4caf50;")
        else:
            self._disk_lbl.setText("not yet written")
            self._disk_lbl.setStyleSheet("color: #666666;")

        if _is_zarr_collection(display_path):
            self._load_collection_view(display_path)
        else:
            self._load_metadata_view(spec)

    def _copy_path(self) -> None:
        QApplication.clipboard().setText(self._path_edit.text())

    def _load_metadata_view(self, spec: "HeaveSpec") -> None:
        self._stack.setCurrentIndex(0)
        self._axes_lbl.setText(spec.axes or "—")
        self._shape_lbl.setText(
            " × ".join(str(s) for s in spec.shape) if spec.shape else "—"
        )
        self._dtype_lbl.setText(spec.dtype or "—")
        self._scales_lbl.setText(
            "  ".join(f"{k}={v}" for k, v in (spec.scales or {}).items()) or "—"
        )
        self._units_lbl.setText(
            "  ".join(f"{k}={v}" for k, v in (spec.units or {}).items()) or "—"
        )

    def _load_collection_view(self, directory: str) -> None:
        self._stack.setCurrentIndex(1)
        self._zarr_list.clear()
        self._c_axes_lbl.setText("—")
        self._c_shape_lbl.setText("—")
        self._c_dtype_lbl.setText("—")
        try:
            from eubi_flow.batch import scan_ome_zarrs
            zarr_paths = scan_ome_zarrs(directory)
            for p in sorted(zarr_paths):
                item = QListWidgetItem(Path(p).stem)
                item.setData(256, p)   # Qt.UserRole = 256
                self._zarr_list.addItem(item)
        except Exception:
            pass

    def _on_zarr_selected(self, _text: str) -> None:
        item = self._zarr_list.currentItem()
        if item is None:
            return
        zarr_path = item.data(256)
        try:
            from eubi_bridge.core.pyramid_reader import read_pyramid
            reader = read_pyramid(zarr_path)
            pyr5d  = reader.pyr.to5D()
            arr    = pyr5d.base_array
            self._c_axes_lbl.setText(pyr5d.axes or "—")
            self._c_shape_lbl.setText(" × ".join(str(s) for s in arr.shape))
            self._c_dtype_lbl.setText(str(arr.dtype))
        except Exception as exc:
            self._c_axes_lbl.setText(f"error: {exc}")


def _wrap_hbox(hbox: QHBoxLayout) -> QWidget:
    """Wrap a QHBoxLayout in a QWidget so it can be used in a QFormLayout."""
    w = QWidget()
    w.setLayout(hbox)
    return w


# ---------------------------------------------------------------------------
# CollapsibleSection
# ---------------------------------------------------------------------------

class CollapsibleSection(QWidget):
    """A titled section with a toggle arrow that shows/hides its content."""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._toggle = QToolButton()
        self._toggle.setCheckable(True)
        self._toggle.setChecked(True)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.ArrowType.DownArrow)
        self._toggle.setText(f"  {title}")
        self._toggle.setStyleSheet(
            "QToolButton { border: none; font-weight: bold; padding: 2px 0px; }"
        )
        self._toggle.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._toggle.clicked.connect(self._on_toggle)
        outer.addWidget(self._toggle)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(8, 4, 4, 4)
        self._content_layout.setSpacing(4)
        outer.addWidget(self._content)

    def _on_toggle(self, checked: bool) -> None:
        self._content.setVisible(checked)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )

    def add_widget(self, w: QWidget) -> None:
        self._content_layout.addWidget(w)

    def add_layout(self, lay) -> None:
        self._content_layout.addLayout(lay)


# ---------------------------------------------------------------------------
# WaveInfoPanel
# ---------------------------------------------------------------------------

class WaveInfoPanel(QWidget):
    """Configuration panel for a selected wave node.

    Layout (top to bottom):
      1. wave_id title (bold)
      2. Processor: combo + Swap button
      3. Collapsible "Parameters" section (ParamsForm + Apply/Discard)
      4. Collapsible "Connections" section (input combo + output name + Apply)
    """

    wave_params_committed = pyqtSignal(str, dict)  # wave_id, payload dict

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── 1. wave_id title ──────────────────────────────────────────
        self._wave_id_lbl = QLabel("—")
        self._wave_id_lbl.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(self._wave_id_lbl)

        # ── 2. Processor selection ────────────────────────────────────
        proc_row = QHBoxLayout()
        proc_row.setSpacing(4)
        proc_label = QLabel("Processor:")
        proc_label.setStyleSheet("font-size: 9px; color: #aaaaaa;")
        self._proc_combo = QComboBox()
        self._swap_btn = QPushButton("Swap")
        self._swap_btn.setFixedWidth(50)
        self._swap_btn.setEnabled(False)
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setFixedWidth(56)
        self._delete_btn.setStyleSheet(
            "QPushButton { background: #7a2020; color: white; border-radius: 3px; }"
            "QPushButton:hover { background: #993030; }"
            "QPushButton:disabled { background: #444; color: #888; }"
        )
        self._proc_combo.currentTextChanged.connect(self._on_proc_changed)
        self._swap_btn.clicked.connect(self._on_swap)
        self._delete_btn.clicked.connect(self._on_delete)
        proc_row.addWidget(proc_label)
        proc_row.addWidget(self._proc_combo, stretch=1)
        proc_row.addWidget(self._swap_btn)
        proc_row.addWidget(self._delete_btn)
        layout.addLayout(proc_row)

        # ── 3. Parameters (collapsible) ───────────────────────────────
        self._params_section = CollapsibleSection("Parameters")

        self._params_form = ParamsForm()
        self._params_form.validation_changed.connect(self._on_validation_changed)
        self._params_form.params_changed.connect(self._on_params_changed)
        params_scroll = QScrollArea()
        params_scroll.setWidgetResizable(True)
        params_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        params_scroll.setMinimumHeight(100)
        params_scroll.setWidget(self._params_form)
        self._params_section.add_widget(params_scroll)

        param_btn_row = QHBoxLayout()
        self._discard_btn = QPushButton("Discard")
        self._apply_btn = QPushButton("Apply params")
        self._apply_btn.setEnabled(False)
        self._discard_btn.clicked.connect(self._on_discard)
        self._apply_btn.clicked.connect(self._on_apply)
        param_btn_row.addWidget(self._discard_btn)
        param_btn_row.addStretch()
        param_btn_row.addWidget(self._apply_btn)
        self._params_section.add_layout(param_btn_row)

        layout.addWidget(self._params_section, stretch=1)

        # ── 4. Connections (collapsible) ──────────────────────────────
        self._conn_section = CollapsibleSection("Connections")

        self._input_combo = QComboBox()
        self._input_combo.setToolTip(
            "Select which heave this wave reads from.\n"
            "Click 'Apply connections' to commit."
        )
        self._output_name_edit = QLineEdit()
        self._output_name_edit.setPlaceholderText("e.g. blurred, binary_mask")
        self._output_name_edit.setToolTip(
            "Rename the output heave produced by this wave.\n"
            "Click 'Apply connections' to commit."
        )
        self._conn_apply_btn = QPushButton("Apply connections")
        self._conn_apply_btn.clicked.connect(self._on_conn_apply)

        conn_form = QFormLayout()
        conn_form.setSpacing(4)
        conn_form.addRow("Input heave:", self._input_combo)
        conn_form.addRow("Output heave:", self._output_name_edit)
        conn_form.addRow("", self._conn_apply_btn)
        conn_widget = QWidget()
        conn_widget.setLayout(conn_form)
        self._conn_section.add_widget(conn_widget)

        layout.addWidget(self._conn_section)

        self._current_wave: "WaveSpec | None" = None
        self._current_flow: "FlowSpec | None" = None
        self._pending_params: dict | None = None

    # ------------------------------------------------------------------

    def load_wave(self, spec: "WaveSpec", flow: "FlowSpec | None" = None) -> None:
        self._current_wave = spec
        self._current_flow = flow
        self._pending_params = None

        self._wave_id_lbl.setText(spec.wave_id)

        # Populate processor combo
        self._proc_combo.blockSignals(True)
        self._proc_combo.clear()
        try:
            from eubi_flow.registry import list_processors
            procs = sorted(list_processors().keys())
            self._proc_combo.addItems(procs)
            self._proc_combo.setCurrentText(spec.name)
        except Exception:
            pass
        self._proc_combo.blockSignals(False)
        self._swap_btn.setEnabled(False)

        # Populate connections section
        self._input_combo.blockSignals(True)
        self._input_combo.clear()
        if flow:
            for hid in sorted(flow.heaves.keys()):
                self._input_combo.addItem(hid)
            current_input = spec.input_heave_ids[0] if spec.input_heave_ids else ""
            idx = self._input_combo.findText(current_input)
            if idx >= 0:
                self._input_combo.setCurrentIndex(idx)
        self._input_combo.blockSignals(False)
        self._output_name_edit.setText(spec.output_heave_id)

        self._params_form.load_wave(spec)
        self._apply_btn.setEnabled(False)

    def _on_proc_changed(self, new_name: str) -> None:
        self._swap_btn.setEnabled(
            self._current_wave is not None and new_name != self._current_wave.name
        )

    def _on_swap(self) -> None:
        if self._current_wave is None:
            return
        new_name = self._proc_combo.currentText()
        self.wave_params_committed.emit(
            self._current_wave.wave_id,
            {"__swap_processor__": new_name}
        )

    def _on_delete(self) -> None:
        if self._current_wave is None:
            return
        self.wave_params_committed.emit(
            self._current_wave.wave_id,
            {"__delete_wave__": True}
        )

    def _on_conn_apply(self) -> None:
        if self._current_wave is None:
            return
        new_input  = self._input_combo.currentText()
        new_output = self._output_name_edit.text().strip()

        conn: dict = {}
        old_input = (self._current_wave.input_heave_ids[0]
                     if self._current_wave.input_heave_ids else "")
        if new_input and new_input != old_input:
            conn["input_heave"] = new_input
        if new_output and new_output != self._current_wave.output_heave_id:
            conn["output_heave"] = new_output

        if conn:
            self.wave_params_committed.emit(
                self._current_wave.wave_id,
                {"__connections__": conn}
            )

    def _on_validation_changed(self, valid: bool) -> None:
        self._apply_btn.setEnabled(
            valid and self._pending_params is not None
        )

    def _on_params_changed(self, params: dict) -> None:
        self._pending_params = params
        same = (self._current_wave is not None and
                params == self._current_wave.params)
        self._apply_btn.setEnabled(self._params_form.is_valid and not same)

    def _on_apply(self) -> None:
        if self._current_wave and self._pending_params is not None:
            self.wave_params_committed.emit(
                self._current_wave.wave_id, self._pending_params
            )
            self._apply_btn.setEnabled(False)

    def _on_discard(self) -> None:
        if self._current_wave:
            self._params_form.load_wave(self._current_wave)
            self._apply_btn.setEnabled(False)


# ---------------------------------------------------------------------------
# PropertiesPanel
# ---------------------------------------------------------------------------

class PropertiesPanel(QStackedWidget):
    """Right sidebar; switches between flow/heave/wave panels."""

    wave_params_committed = pyqtSignal(str, dict)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(280)

        self._flow_panel  = FlowInfoPanel()
        self._heave_panel = HeaveInfoPanel()
        self._wave_panel  = WaveInfoPanel()
        self._wave_panel.wave_params_committed.connect(self.wave_params_committed)

        self.addWidget(self._flow_panel)   # index 0
        self.addWidget(self._heave_panel)  # index 1
        self._wave_scroll = QScrollArea()
        self._wave_scroll.setWidgetResizable(True)
        self._wave_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._wave_scroll.setWidget(self._wave_panel)
        self.addWidget(self._wave_scroll)  # index 2

    def show_flow(self, flow: "FlowSpec", name: str) -> None:
        self._flow_panel.load_flow(flow, name)
        self.setCurrentIndex(0)

    def show_heave(self, spec: "HeaveSpec", flow=None, flow_name: str | None = None) -> None:
        self._heave_panel.load_heave(spec, flow, flow_name)
        self.setCurrentIndex(1)

    def show_wave(self, spec: "WaveSpec", flow: "FlowSpec | None" = None) -> None:
        self._wave_panel.load_wave(spec, flow)
        self.setCurrentIndex(2)

    def show_empty(self) -> None:
        self.setCurrentIndex(0)
