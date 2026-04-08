"""
Scrollable log output widget with color-coded severity lines.
"""
from __future__ import annotations

from PyQt6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


_COLORS = {
    "ERROR":   "#ff6b6b",
    "WARN":    "#ffd93d",
    "WARNING": "#ffd93d",
    "INFO":    "#c8c8c8",
    "DEBUG":   "#888888",
}
_DEFAULT_COLOR = "#c8c8c8"


def _severity(line: str) -> str:
    upper = line.upper()
    for key in _COLORS:
        if key in upper:
            return key
    return ""


class LogWidget(QWidget):
    """Read-only monospace log viewer with per-line colorization."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setFont(QFont("Courier New", 9))
        self._text.setMaximumBlockCount(5000)
        self._text.setStyleSheet(
            "QPlainTextEdit { background: #1e1e1e; color: #c8c8c8; border: none; }"
        )

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedHeight(22)
        clear_btn.clicked.connect(self.clear)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(clear_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._text)
        layout.addLayout(btn_row)

    def append_line(self, msg: str):
        """Append *msg* with appropriate color and auto-scroll to bottom."""
        sev = _severity(msg)
        hex_color = _COLORS.get(sev, _DEFAULT_COLOR)

        fmt = QTextCharFormat()
        fmt.setForeground(QColor(hex_color))

        cursor = self._text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not self._text.toPlainText() == "":
            cursor.insertBlock()
        cursor.insertText(msg, fmt)
        self._text.setTextCursor(cursor)
        self._text.ensureCursorVisible()

    def clear(self):
        self._text.clear()

    def text(self) -> str:
        return self._text.toPlainText()
