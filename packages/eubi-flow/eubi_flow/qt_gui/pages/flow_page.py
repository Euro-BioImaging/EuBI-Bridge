"""FlowPage — the "Flow" tab of the EuBI-Bridge main window.

Hosts the WavePalette, FlowCanvas, PropertiesPanel and FlowToolbar.
All DAG edits are persisted immediately via FlowEditor; the page reloads
the FlowSpec from disk after every mutation so the canvas stays in sync.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QPointF, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from eubi_flow.qt_gui.flow_page.canvas import FlowCanvas, FlowScene
from eubi_flow.qt_gui.flow_page.items import HeaveItem, WaveItem
from eubi_flow.qt_gui.flow_page.palette import WavePalette
from eubi_flow.qt_gui.flow_page.properties import PropertiesPanel
from eubi_flow.qt_gui.flow_page.toolbar import FlowToolbar
from eubi_flow.qt_gui.flow_page.worker import FlowRunWorker
from eubi_bridge.qt_gui.widgets.log_widget import LogWidget


class FlowPage(QWidget):
    """Fourth tab: drag-and-drop flow builder and executor."""

    status_changed = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._flow_name: Optional[str] = None
        self._flow = None
        self._worker: Optional[FlowRunWorker] = None

        # ── scene + canvas ────────────────────────────────────────────
        self._scene  = FlowScene()
        self._canvas = FlowCanvas(self._scene)

        # ── palette (left) ────────────────────────────────────────────
        self._palette = WavePalette()

        # ── properties (right) ────────────────────────────────────────
        self._props = PropertiesPanel()

        # ── toolbar (top) ─────────────────────────────────────────────
        self._toolbar = FlowToolbar()

        # ── log (bottom) ──────────────────────────────────────────────
        self._log = LogWidget()
        self._log.setMinimumHeight(80)
        self._log.setMaximumHeight(200)

        # ── layout ────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._palette)
        splitter.addWidget(self._canvas)
        splitter.addWidget(self._props)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([200, 800, 280])

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._toolbar)
        root.addWidget(splitter, stretch=1)
        root.addWidget(self._log)

        # ── debounced layout-save timer ───────────────────────────────
        self._layout_save_timer = QTimer()
        self._layout_save_timer.setSingleShot(True)
        self._layout_save_timer.setInterval(2000)
        self._layout_save_timer.timeout.connect(self._save_layout)

        self._connect_signals()

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self._toolbar.new_flow_requested.connect(self._on_new_flow)
        self._toolbar.open_flow_requested.connect(self._on_open_flow)
        self._toolbar.save_requested.connect(self._save_layout)
        self._toolbar.lint_requested.connect(self._on_lint)
        self._toolbar.run_requested.connect(self._on_run)
        self._toolbar.stop_requested.connect(self._on_stop)
        self._toolbar.clear_log_requested.connect(self._log.clear)

        self._scene.item_selected.connect(self._on_item_selected)
        self._scene.layout_dirty_changed.connect(self._on_layout_dirty)
        self._scene.node_dropped.connect(self._on_node_dropped)

        self._props.wave_params_committed.connect(self._on_params_committed)

    # ------------------------------------------------------------------
    # Flow load / save
    # ------------------------------------------------------------------

    def _load_flow(self, name: str) -> None:
        from eubi_flow.eubiflow import _load
        from eubi_flow.serialization import load_flow as _lf
        try:
            flow = _load(name)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return

        self._flow_name = name
        self._flow = flow
        self._scene.load_flow(name, flow)
        self._toolbar.set_flow_loaded(True)
        self._props.show_flow(flow, name)
        self.status_changed.emit(f"Flow '{name}' loaded — {len(flow.waves)} wave(s)")

    def _save_layout(self) -> None:
        if self._flow_name:
            self._scene.save_layout()

    def _on_layout_dirty(self) -> None:
        self._layout_save_timer.start()

    # ------------------------------------------------------------------
    # Toolbar actions
    # ------------------------------------------------------------------

    def _on_new_flow(self) -> None:
        dlg = _NewFlowDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        name, input_path, workdir = dlg.values()
        try:
            from eubi_flow.eubiflow import EuBIFlow
            EuBIFlow().create(name, input_path, workdir)
        except Exception as exc:
            QMessageBox.critical(self, "Create error", str(exc))
            return
        self._load_flow(name)

    def _on_open_flow(self) -> None:
        from eubi_flow.eubiflow import _flows_dir
        flows_dir = _flows_dir()
        entries = sorted(p.stem for p in flows_dir.glob("*.json"))
        if not entries:
            QMessageBox.information(self, "No flows",
                                    f"No flows found in {flows_dir}.")
            return
        name, ok = QInputDialog.getItem(
            self, "Open Flow", "Select a flow:", entries, 0, False
        )
        if ok and name:
            self._load_flow(name)

    def _on_lint(self) -> None:
        if self._flow is None:
            return
        from eubi_flow.validation import lint_flow
        errors = lint_flow(self._flow)
        self._log.show()
        self._log.clear()
        if errors:
            self._log.append_line(f"✗  {len(errors)} validation error(s):")
            for e in errors:
                self._log.append_line(f"   {e}")
        else:
            self._log.append_line("✓  Flow is valid — all wave parameters OK.")

    def _on_run(self) -> None:
        if self._flow is None or self._flow_name is None:
            return

        # Pre-flight validation
        from eubi_flow.validation import lint_flow
        errors = lint_flow(self._flow)
        if errors:
            QMessageBox.warning(self, "Validation errors",
                                "Fix these before running:\n\n" + "\n".join(errors))
            return

        # Workdir setup, OME-Zarr discovery, and batch detection all happen
        # inside _run_flow_async (the shared CLI/GUI execution core).
        self._scene.reset_all_statuses()
        self._log.clear()
        self._log.append_line(f"Preparing flow '{self._flow_name}'…")

        dc = self._flow.config.downscale
        cc = self._flow.config.cluster
        self._worker = FlowRunWorker(
            self._flow_name,
            self._flow,
            max_workers=cc.max_workers,
            region_size_mb=cc.region_size_mb,
            n_layers=dc.n_layers,
            scale_factor=dc.as_scale_factor_tuple(),
            downscale_method=dc.downscale_method,
        )
        self._worker.wave_status_changed.connect(self._on_wave_status)
        self._worker.flow_finished.connect(self._on_flow_finished)
        self._worker.flow_failed.connect(self._on_flow_failed)
        self._worker.log_line.connect(self._log.append_line)
        # Clean up AFTER run() returns — not inside the result callbacks,
        # which fire while the thread's finally block is still executing.
        self._worker.finished.connect(self._on_worker_done)
        self._worker.start()
        self._toolbar.set_running(True)

    def _on_stop(self) -> None:
        if self._worker:
            self._worker.stop()

    # ------------------------------------------------------------------
    # Run worker callbacks
    # ------------------------------------------------------------------

    def _on_wave_status(self, wave_id: str, status: str) -> None:
        self._scene.update_wave_status(wave_id, status)

    def _on_worker_done(self) -> None:
        """Called by QThread.finished — fires after run() returns completely."""
        if self._worker:
            self._worker.deleteLater()
            self._worker = None

    def _on_flow_finished(self) -> None:
        self._toolbar.set_running(False)
        self._toolbar.set_flow_loaded(True)
        self._log.append_line("✓  Flow completed successfully.")
        self.status_changed.emit("Flow completed")
        if self._flow_name:
            self._load_flow(self._flow_name)

    def _on_flow_failed(self, error: str) -> None:
        self._toolbar.set_running(False)
        self._toolbar.set_flow_loaded(True)
        summary = error.strip().splitlines()[-1] if error.strip() else "Unknown error"
        self._log.append_line(f"✗  {summary}")
        self.status_changed.emit(f"Flow failed: {summary}")
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Flow failed")
        dlg.setIcon(QMessageBox.Icon.Critical)
        dlg.setText(summary)
        dlg.setDetailedText(error)
        dlg.exec()

    # ------------------------------------------------------------------
    # Item selection → properties panel
    # ------------------------------------------------------------------

    def _on_item_selected(self, item) -> None:
        if isinstance(item, HeaveItem):
            self._props.show_heave(item.heave_spec, self._flow, self._flow_name)
        elif isinstance(item, WaveItem):
            self._props.show_wave(item.wave_spec, self._flow)
        else:
            if self._flow and self._flow_name:
                self._props.show_flow(self._flow, self._flow_name)
            else:
                self._props.show_empty()

    # ------------------------------------------------------------------
    # Drop handling — create new wave
    # ------------------------------------------------------------------

    def _on_node_dropped(self, wave_name: str, scene_pos: QPointF) -> None:
        if self._flow is None or self._flow_name is None:
            self.status_changed.emit("Open or create a flow before adding waves.")
            return

        # Default to the last heave in the chain.
        # Use the sidebar Connections section to rewire after dropping.
        input_heave = self._flow.last_heave_id()

        try:
            from eubi_flow.eubiflow import FlowEditor
            FlowEditor(self._flow_name).add_wave(
                wave_name.strip(), input_heave=input_heave
            )
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return

        # Reload so new items appear
        self._load_flow(self._flow_name)

        # Position the new WaveItem at the drop point
        if self._flow and self._flow.waves:
            new_wave_id = self._flow.waves[-1].wave_id
            if new_wave_id in self._scene._wave_items:
                self._scene._wave_items[new_wave_id].setPos(scene_pos)
                # Also position its output heave slightly to the right
                out_hid = self._flow.waves[-1].output_heave_id
                if out_hid in self._scene._heave_items:
                    from eubi_flow.qt_gui.flow_page.canvas import _COL_SPACING
                    self._scene._heave_items[out_hid].setPos(
                        scene_pos + QPointF(_COL_SPACING, 0)
                    )
                for edge in self._scene._edge_items:
                    edge.update_path()
                self._on_layout_dirty()

    # ------------------------------------------------------------------
    # Params committed from properties panel
    # ------------------------------------------------------------------

    def _on_params_committed(self, wave_id: str, params: dict) -> None:
        if self._flow_name is None:
            return
        try:
            from eubi_flow.eubiflow import FlowEditor
            editor = FlowEditor(self._flow_name)

            if "__delete_wave__" in params:
                editor.remove_wave(wave_id)
                old_positions = self._scene.current_positions()
                self._load_flow(self._flow_name)
                self._scene.apply_layout(old_positions)
                # Wave is gone — show flow-level panel
                if self._flow and self._flow_name:
                    self._props.show_flow(self._flow, self._flow_name)
                return
            elif "__swap_processor__" in params:
                editor.update_wave(wave_id, wave_name=params["__swap_processor__"])
            elif "__connections__" in params:
                conn = params["__connections__"]
                editor.update_wave(
                    wave_id,
                    input_heave=conn.get("input_heave"),
                    output_heave=conn.get("output_heave"),
                )
            else:
                editor.update_wave(wave_id, **params)
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return

        # Preserve current positions across the reload
        old_positions = self._scene.current_positions()
        self._load_flow(self._flow_name)
        self._scene.apply_layout(old_positions)
        self._scene.update_wave_item(wave_id)

        # Re-select the same wave so the form refreshes
        if wave_id in self._scene._wave_items:
            self._scene.clearSelection()
            self._scene._wave_items[wave_id].setSelected(True)


# ---------------------------------------------------------------------------
# Helper dialogs
# ---------------------------------------------------------------------------

class _NewFlowDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Flow")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._name_edit   = QLineEdit()
        self._input_edit  = QLineEdit()
        self._input_btn   = _browse_btn(self._input_edit, directory=True)
        self._workdir_edit = QLineEdit()
        self._workdir_btn = _browse_btn(self._workdir_edit, directory=True)

        form.addRow("Flow name:", self._name_edit)

        input_row = _row_with_browse(self._input_edit, self._input_btn)
        form.addRow("Input path:", input_row)

        work_row = _row_with_browse(self._workdir_edit, self._workdir_btn)
        form.addRow("Workdir:", work_row)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> tuple[str, str, str]:
        return (
            self._name_edit.text().strip(),
            self._input_edit.text().strip(),
            self._workdir_edit.text().strip(),
        )


def _browse_btn(line_edit: QLineEdit, directory: bool = True):
    from PyQt6.QtWidgets import QPushButton
    btn = QPushButton("…")
    btn.setFixedWidth(28)

    def _browse():
        if directory:
            path = QFileDialog.getExistingDirectory(None, "Select directory")
        else:
            path, _ = QFileDialog.getOpenFileName(None, "Select file")
        if path:
            line_edit.setText(path)

    btn.clicked.connect(_browse)
    return btn


def _row_with_browse(line_edit: QLineEdit, btn):
    from PyQt6.QtWidgets import QHBoxLayout
    row = QWidget()
    hl = QHBoxLayout(row)
    hl.setContentsMargins(0, 0, 0, 0)
    hl.addWidget(line_edit)
    hl.addWidget(btn)
    return row
