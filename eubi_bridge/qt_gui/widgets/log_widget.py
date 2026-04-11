"""
Scrollable log output widget styled to match the Rich CLI output.

Structured lines arrive as:  ``HH:MM:SS\x01LEVELNAME\x01module.py:line\x01message``
Plain lines (subprocess output, banner lines) fall back to whole-line colouring.
"""
from __future__ import annotations

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

_SEP = "\x01"

# Colours for each severity level — foreground
_LEVEL_FG: dict[str, str] = {
    "DEBUG":    "#7c7c7c",
    "INFO":     "#4ec994",   # Rich green
    "WARNING":  "#ffd93d",
    "WARN":     "#ffd93d",
    "ERROR":    "#ff6b6b",
    "CRITICAL": "#ff3333",
}
# Background tint for the level badge
_LEVEL_BG: dict[str, str] = {
    "DEBUG":    "#2a2a2a",
    "INFO":     "#1a3328",
    "WARNING":  "#332d10",
    "WARN":     "#332d10",
    "ERROR":    "#3a1a1a",
    "CRITICAL": "#4a0000",
}

_COLOR_TIME    = "#5c6370"   # dim grey  (like Rich's dimmed timestamp)
_COLOR_MODULE  = "#5c8fa8"   # muted blue-cyan (like Rich's path)
_COLOR_MESSAGE = "#d4d4d4"   # near-white
_COLOR_PLAIN   = "#c8c8c8"   # fallback for unstructured lines


def _plain_severity(line: str) -> str:
    upper = line.upper()
    for key in _LEVEL_FG:
        if key in upper:
            return key
    return ""


def _fmt(color: str, bg: str | None = None, bold: bool = False) -> QTextCharFormat:
    f = QTextCharFormat()
    f.setForeground(QColor(color))
    if bg:
        f.setBackground(QColor(bg))
    if bold:
        f.setFontWeight(700)
    return f


class LogWidget(QWidget):
    """Read-only monospace log viewer, Rich-style per-segment colorisation."""

    _FLUSH_INTERVAL_MS = 100   # batch flush period
    _MAX_BATCH = 200           # max lines rendered per flush to stay responsive

    def __init__(self, parent=None):
        super().__init__(parent)
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setFont(QFont("Courier New", 9))
        self._text.setMaximumBlockCount(5000)
        self._text.setStyleSheet(
            "QPlainTextEdit { background: #1e1e1e; color: #d4d4d4; border: none; }"
        )

        self._pending: list[str] = []
        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(self._FLUSH_INTERVAL_MS)
        self._flush_timer.timeout.connect(self._flush)
        self._flush_timer.start()

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

    # ------------------------------------------------------------------
    def append_line(self, msg: str):
        """Buffer *msg*; the flush timer renders it in batches."""
        self._pending.append(msg)

    def _flush(self):
        """Render up to _MAX_BATCH buffered lines in one pass."""
        if not self._pending:
            return
        batch, self._pending = self._pending[:self._MAX_BATCH], self._pending[self._MAX_BATCH:]

        cursor = self._text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        has_content = self._text.document().blockCount() > 1 or self._text.document().firstBlock().text()

        for i, msg in enumerate(batch):
            if has_content or i > 0:
                cursor.insertBlock()
            if _SEP in msg:
                self._insert_structured(cursor, msg)
            else:
                self._insert_plain(cursor, msg)

        self._text.setTextCursor(cursor)
        self._text.ensureCursorVisible()

    # ------------------------------------------------------------------
    def _insert_structured(self, cursor: QTextCursor, msg: str):
        """Render a ``ts\x01LEVEL\x01module:line\x01message`` line."""
        parts = msg.split(_SEP, 3)
        if len(parts) != 4:
            self._insert_plain(cursor, msg)
            return

        ts, level, module, message = parts
        level_key = level.upper()
        fg = _LEVEL_FG.get(level_key, _COLOR_MESSAGE)
        bg = _LEVEL_BG.get(level_key, "#1e1e1e")

        # [HH:MM:SS]
        cursor.insertText(f"[{ts}] ", _fmt(_COLOR_TIME))
        # LEVELNAME badge
        cursor.insertText(f"{level:<8}", _fmt(fg, bg, bold=True))
        cursor.insertText(" ", _fmt(_COLOR_MESSAGE))
        # message
        cursor.insertText(message, _fmt(_COLOR_MESSAGE))
        # right-hand module:line — padded with spaces
        padding = max(1, 60 - len(message))
        cursor.insertText(" " * padding, _fmt(_COLOR_MESSAGE))
        cursor.insertText(module, _fmt(_COLOR_MODULE))

    def _insert_plain(self, cursor: QTextCursor, msg: str):
        sev = _plain_severity(msg)
        color = _LEVEL_FG.get(sev, _COLOR_PLAIN)
        cursor.insertText(msg, _fmt(color))

    # ------------------------------------------------------------------
    def clear(self):
        self._pending.clear()
        self._text.clear()

    def text(self) -> str:
        return self._text.toPlainText()
