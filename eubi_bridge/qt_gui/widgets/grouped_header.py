"""
Three-row spanning table header: tab / group / parameter.

Qt's QHeaderView paints one label per section, so a hierarchy has to be drawn.
:class:`GroupedHeaderView` keeps the section geometry Qt already manages and only
takes over painting: it merges the horizontal runs of adjacent columns that share
a tab (top row) or a tab+group (middle row) and centres one label across each
run, leaving the parameter name on the bottom row.

    ┌──────────────────┬──────────┬─────────────────────┐
    │      paths       │  Reader  │     Conversion      │   tab
    ├──────────────────┼──────────┼─────────────────────┤
    │                  │  Scenes  │      Chunking       │   group
    ├─────────┬────────┼──────────┼──────────┬──────────┤
    │  input  │ output │  Scene   │   Auto   │ Chunk x  │   parameter
    │  _path  │ _path  │  index   │  chunk   │          │
    └─────────┴────────┴──────────┴──────────┴──────────┘

Because only ``paintSection`` and ``sizeHint`` are overridden, resizing,
scrolling, and hit-testing keep working.  A click anywhere in a column still
selects that column's section, which is what the batch table's cell selection
relies on.
"""
from __future__ import annotations

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QHeaderView, QStyle, QStyleOptionHeader


class GroupedHeaderView(QHeaderView):
    """Header that spans shared tab/group labels across adjacent columns.

    The owner supplies one ``(tab, group, label)`` triple per column via
    :meth:`set_hierarchy`; empty strings collapse the corresponding row for that
    column, so a parameter with no group box simply has its label span the two
    lower rows.
    """

    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._hierarchy: list[tuple[str, str, str]] = []
        self.setSectionsClickable(True)
        self.setHighlightSections(False)
        self.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)

    # -- data --

    def set_hierarchy(self, hierarchy: list[tuple[str, str, str]]) -> None:
        """Set one ``(tab, group, label)`` triple per column and repaint."""
        self._hierarchy = list(hierarchy)
        self.updateGeometries()
        self.viewport().update()

    def _triple(self, index: int) -> tuple[str, str, str]:
        if 0 <= index < len(self._hierarchy):
            return self._hierarchy[index]
        return "", "", ""

    # -- geometry --

    def _row_height(self) -> int:
        return max(18, self.fontMetrics().height() + 6)

    def sizeHint(self):
        size = super().sizeHint()
        # Two extra rows on top of whatever Qt sized the label row as.
        size.setHeight(self._row_height() * 2 + max(
            size.height(), self._row_height() * 2))
        return size

    def _span(self, index: int, depth: int) -> tuple[int, int]:
        """First and last column sharing *index*'s key at *depth* (0=tab, 1=group).

        Runs are contiguous by construction: the model orders columns by the
        conversion form's own sequence, so columns of one group are adjacent.
        """
        def key(i: int) -> tuple[str, ...]:
            tab, group, _ = self._triple(i)
            return (tab,) if depth == 0 else (tab, group)

        target = key(index)
        if not any(target):
            return index, index

        first = index
        while first > 0 and key(first - 1) == target:
            first -= 1
        last = index
        count = self.count()
        while last + 1 < count and key(last + 1) == target:
            last += 1
        return first, last

    # -- painting --

    def paintSection(self, painter: QPainter, rect: QRect, index: int) -> None:
        """Paint only the bottom row: the parameter name for this column.

        The two rows above span several columns, and Qt clips every
        ``paintSection`` call to its own section, so they cannot be drawn from
        here.  :meth:`paintEvent` draws them across the whole header instead.
        """
        if not rect.isValid() or not self._hierarchy:
            super().paintSection(painter, rect, index)
            return

        tab, group, label = self._triple(index)
        row = self._row_height()
        top_rows = (1 if tab else 0) + (1 if group else 0)
        label_rect = QRect(rect.left(), rect.top() + row * top_rows,
                           rect.width(), rect.height() - row * top_rows)
        self._paint_cell(painter, label_rect, label, index, bold=False)

    def paintEvent(self, event):
        """Draw the sections, then the spanning tab/group rows on top."""
        super().paintEvent(event)
        if not self._hierarchy:
            return

        painter = QPainter(self.viewport())
        row = self._row_height()
        count = self.count()

        for depth in (0, 1):
            index = 0
            while index < count:
                if self.isSectionHidden(index):
                    index += 1
                    continue
                tab, group, _ = self._triple(index)
                text = tab if depth == 0 else group
                first, last = self._span(index, depth)
                if not text:
                    index = last + 1
                    continue

                left = self.sectionViewportPosition(first)
                right = (self.sectionViewportPosition(last)
                         + self.sectionSize(last))
                # A tab with no group still owns the group row's height only
                # when some column in the run has a group; otherwise the label
                # row below already claimed that space.
                band = QRect(left, row * depth, right - left, row)
                self._paint_cell(painter, band, text, first,
                                 bold=(depth == 0))
                index = last + 1
        painter.end()

    def _paint_cell(self, painter, rect, text, index, bold, divider=True):
        painter.save()

        option = QStyleOptionHeader()
        self.initStyleOption(option)
        option.rect = rect
        option.section = index
        option.text = ""
        option.state |= QStyle.StateFlag.State_Enabled
        self.style().drawControl(
            QStyle.ControlElement.CE_HeaderSection, option, painter, self)

        if divider:
            painter.setPen(QColor(0, 0, 0, 60))
            painter.drawLine(rect.topRight(), rect.bottomRight())
            painter.drawLine(rect.bottomLeft(), rect.bottomRight())

        if text:
            font = painter.font()
            font.setBold(bold)
            painter.setFont(font)
            painter.setPen(self.palette().buttonText().color())
            painter.drawText(
                rect.adjusted(3, 0, -3, 0),
                int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap),
                text)
        painter.restore()
