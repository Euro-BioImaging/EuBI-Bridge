"""FlowToolbar — buttons for flow lifecycle actions."""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QWidget


class FlowToolbar(QWidget):
    """Horizontal toolbar above the canvas."""

    new_flow_requested  = pyqtSignal()
    open_flow_requested = pyqtSignal()
    save_requested      = pyqtSignal()
    lint_requested      = pyqtSignal()
    run_requested       = pyqtSignal()
    stop_requested      = pyqtSignal()
    clear_log_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(34)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        self._new_btn   = QPushButton("New Flow")
        self._open_btn  = QPushButton("Open Flow")
        self._save_btn  = QPushButton("Save Layout")
        self._lint_btn  = QPushButton("Lint")
        self._run_btn   = QPushButton("▶  Run")
        self._stop_btn  = QPushButton("■  Stop")
        self._clear_btn = QPushButton("Clear Log")

        self._run_btn.setStyleSheet(
            "QPushButton { background: #2d6e3e; color: white; }"
            "QPushButton:hover { background: #3a8a50; }"
            "QPushButton:disabled { background: #444; color: #888; }"
        )
        self._stop_btn.setStyleSheet(
            "QPushButton { background: #7a2020; color: white; }"
            "QPushButton:hover { background: #993030; }"
            "QPushButton:disabled { background: #444; color: #888; }"
        )

        for btn in (self._new_btn, self._open_btn, self._save_btn, self._lint_btn,
                    self._run_btn, self._stop_btn, self._clear_btn):
            btn.setFixedHeight(26)
            layout.addWidget(btn)
        layout.addStretch()

        self._new_btn.clicked.connect(self.new_flow_requested)
        self._open_btn.clicked.connect(self.open_flow_requested)
        self._save_btn.clicked.connect(self.save_requested)
        self._lint_btn.clicked.connect(self.lint_requested)
        self._run_btn.clicked.connect(self.run_requested)
        self._stop_btn.clicked.connect(self.stop_requested)
        self._clear_btn.clicked.connect(self.clear_log_requested)

        self.set_flow_loaded(False)
        self.set_running(False)

    def set_flow_loaded(self, loaded: bool) -> None:
        for btn in (self._save_btn, self._lint_btn, self._run_btn):
            btn.setEnabled(loaded)
        self._stop_btn.setEnabled(False)

    def set_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        for btn in (self._new_btn, self._open_btn, self._save_btn, self._lint_btn):
            btn.setEnabled(not running)
