"""
Persistent sidebar file/folder browser with pagination and OME-Zarr detection.

Two modes:
  mode="zarr"        : navigate filesystem; double-click an OME-Zarr → zarr_selected(path)
  mode="conversion"  : navigate filesystem; check files/OME-Zarr stores → selection_changed(paths)
  Both modes also accept drag-and-drop: files to select, or one store to open.
"""
from __future__ import annotations

import fnmatch
import json
import os
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from eubi_bridge.qt_gui.core.file_service import (
    PAGE_SIZE,
    FileEntry,
    _is_ome_zarr_local,
    get_parent,
    is_remote,
    list_local,
    list_local_recursive,
    list_s3,
    paginate,
)

# Unicode icons (no Qt resource system needed)
_ICON_FOLDER  = "\U0001F4C1"   # 📁
_ICON_ZARR    = "\U0001F52C"   # 🔬
_ICON_FILE    = "\U0001F4C4"   # 📄
_ICON_HOME    = "\U0001F3E0"   # 🏠

_DROP_HINTS = {
    # mode: (idle text, text while a drag hovers)
    "conversion": ("Select input files in the browser below, or drag files here",
                   "Drop to select"),
    "zarr":       ("Click an OME-Zarr below to inspect it, or drag one here",
                   "Drop to inspect"),
}
_DROP_HINT_STYLE = (
    "font-size: 11px; color: #888; padding: 4px;"
    "border: 1px dashed #888; border-radius: 3px;"
)
_DROP_HINT_STYLE_ACTIVE = (
    "font-size: 11px; color: #4a9eff; padding: 4px;"
    "border: 1px dashed #4a9eff; border-radius: 3px;"
)
_LIST_STYLE_DROP_ACTIVE = "QListWidget { border: 2px dashed #4a9eff; }"
# Translucent, so the same tint reads on the light and the dark theme.
_OPEN_STORE_BG = QColor(74, 158, 255, 70)

_RECENTS_MAX = 3
_RECENTS_FILE = Path.home() / ".eubi_bridge" / "gui_recents_cache" / "recent_dirs.json"


def _load_recents() -> list[str]:
    try:
        return json.loads(_RECENTS_FILE.read_text())
    except Exception:
        return []


def _push_recent(path: str) -> None:
    recents = _load_recents()
    if path in recents:
        recents.remove(path)
    recents.insert(0, path)
    try:
        _RECENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _RECENTS_FILE.write_text(json.dumps(recents[:_RECENTS_MAX], indent=2))
    except Exception:
        pass


def _pat_match(name: str, pat: str) -> bool:
    """Match *name* against *pat*.

    If *pat* contains a wildcard character (``*``, ``?``, or ``[``), use
    :func:`fnmatch.fnmatch`; otherwise do a case-insensitive substring check so
    that plain strings like ``tif`` match any file whose name contains ``tif``.
    """
    if any(c in pat for c in "*?["):
        return fnmatch.fnmatch(name, pat)
    return pat.lower() in name.lower()


def _is_selectable(entry: FileEntry) -> bool:
    """Files and OME-Zarr stores are conversion inputs; a plain folder is not.

    A ticked folder used to reach the reader as if it were an image and fail
    with an unsupported-format error.  Picking a folder's files is what
    Select All is for, and it shows exactly which files that means.
    """
    return entry["isOmeZarr"] or not entry["isDirectory"]


class SidebarBrowser(QWidget):
    """Persistent sidebar file browser.

    Signals (zarr mode):
        zarr_selected(str)          : user single-clicked an OME-Zarr store

    Signals (conversion mode):
        selection_changed(list[str]): checked paths changed

    Signals (all modes):
        path_navigated(str)         : current directory changed (any navigation)
    """

    zarr_selected    = pyqtSignal(str)
    selection_changed = pyqtSignal(list)
    path_navigated   = pyqtSignal(str)

    def __init__(self, mode: str = "zarr", initial_path: str = "", parent=None):
        super().__init__(parent)
        assert mode in ("zarr", "conversion", "output"), f"Unknown mode: {mode}"
        self._mode = mode
        self._current_path = ""
        self._entries: list[FileEntry] = []
        self._page = 0
        self._total = 0           # for S3 server-side pagination
        self._s3_mode = False
        self._checked_paths: set[str] = set()
        self._include_filter: list[str] = []   # live filter patterns (fnmatch)
        self._exclude_filter: list[str] = []
        self._recursive_entries: list[FileEntry] = []   # populated when filters active
        self._recursive_mode: bool = False              # True = showing recursive results

        # Single-click timer: fires zarr_selected only if no double-click follows
        self._click_timer = QTimer(self)
        self._click_timer.setSingleShot(True)
        self._click_timer.setInterval(200)
        self._click_timer.timeout.connect(self._on_click_confirmed)
        self._pending_click_path: str = ""
        # zarr mode: the store open in the viewer, marked in the list so it
        # stays identifiable after navigating, clicking elsewhere or a drop.
        self._open_store: str = ""

        self._build_ui()

        start = initial_path or os.path.expanduser("~")
        self._navigate(start)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(3)

        # ── Top bar ──
        top = QHBoxLayout()
        top.setSpacing(2)

        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Path or s3://...")
        self._path_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._path_edit.returnPressed.connect(self._on_path_entered)
        top.addWidget(self._path_edit)

        up_btn = QPushButton("↑")
        up_btn.setFixedSize(24, 24)
        up_btn.setToolTip("Go up one directory")
        up_btn.clicked.connect(self._on_up)
        top.addWidget(up_btn)

        home_btn = QPushButton(_ICON_HOME)
        home_btn.setFixedSize(24, 24)
        home_btn.setToolTip("Go to home directory")
        home_btn.clicked.connect(self._on_home)
        top.addWidget(home_btn)

        self._recent_btn = QPushButton("⏱")
        self._recent_btn.setFixedSize(24, 24)
        self._recent_btn.setToolTip("Recent directories")
        self._recent_btn.clicked.connect(self._show_recents_menu)
        self._recent_btn.setEnabled(bool(_load_recents()))
        top.addWidget(self._recent_btn)

        layout.addLayout(top)

        # ── Extra action bar (mode-specific) ──
        if self._mode == "output":
            new_folder_btn = QPushButton("\U0001F4C2 New Folder")
            new_folder_btn.setFixedHeight(24)
            new_folder_btn.setToolTip("Create a new sub-folder in the current directory")
            new_folder_btn.clicked.connect(self._on_new_folder)
            layout.addWidget(new_folder_btn)

        if self._mode == "conversion":
            sel_bar = QHBoxLayout()
            sel_bar.setSpacing(4)
            self._select_all_btn = QPushButton("Select All")
            self._select_all_btn.setFixedHeight(22)
            self._select_all_btn.setToolTip("Select all items matching the current filter")
            self._select_all_btn.clicked.connect(self._on_select_all)
            sel_bar.addWidget(self._select_all_btn)
            self._deselect_all_btn = QPushButton("Deselect All")
            self._deselect_all_btn.setFixedHeight(22)
            self._deselect_all_btn.setToolTip("Clear the entire selection (all folders)")
            self._deselect_all_btn.clicked.connect(self._on_deselect_all)
            sel_bar.addWidget(self._deselect_all_btn)
            self._filter_info_label = QLabel("")
            self._filter_info_label.setStyleSheet("font-size: 9px; color: #aaa;")
            sel_bar.addWidget(self._filter_info_label, 1)
            layout.addLayout(sel_bar)

        if self._mode in _DROP_HINTS:
            # Drag-and-drop is an alternative to ticking (conversion) or
            # clicking a store (zarr), with the same effect.  The whole browser
            # accepts drops (the list has no drop handling of its own, so Qt
            # hands its drops up to this widget); this row announces it.
            self._drop_hint = QLabel(_DROP_HINTS[self._mode][0])
            self._drop_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._drop_hint.setWordWrap(True)
            self._drop_hint.setStyleSheet(_DROP_HINT_STYLE)
            layout.addWidget(self._drop_hint)
            self.setAcceptDrops(True)

        # ── List ──
        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.itemClicked.connect(self._on_single_click)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        if self._mode == "conversion":
            self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            self._list.itemChanged.connect(self._on_item_changed)
        self._list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._list)

        # ── Pagination bar ──
        pag = QHBoxLayout()
        pag.setSpacing(4)
        self._prev_btn = QPushButton("◀")
        self._prev_btn.setFixedSize(26, 22)
        self._prev_btn.clicked.connect(self._on_prev_page)
        pag.addWidget(self._prev_btn)

        self._page_label = QLabel("")
        self._page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._page_label.setStyleSheet("font-size: 10px; color: #aaa;")
        pag.addWidget(self._page_label, 1)

        self._next_btn = QPushButton("▶")
        self._next_btn.setFixedSize(26, 22)
        self._next_btn.clicked.connect(self._on_next_page)
        pag.addWidget(self._next_btn)

        layout.addLayout(pag)

    # ── Navigation ────────────────────────────────────────────────────────────

    def _navigate(self, path: str, page: int = 0):
        # Leaving recursive mode when user navigates
        self._recursive_mode = False
        self._recursive_entries = []
        self._page = page
        self._s3_mode = is_remote(path)

        if self._s3_mode:
            result = list_s3(path, page=page, page_size=PAGE_SIZE)
            self._entries     = result["items"]
            self._total       = result["total"]
            self._current_path = result["currentPath"]
        else:
            all_entries = list_local(path)
            page_entries, total = paginate(all_entries, page, PAGE_SIZE)
            self._entries     = page_entries
            self._total       = total
            self._current_path = path
            if page == 0:
                _push_recent(path)
                self._recent_btn.setEnabled(True)

        self._path_edit.setText(self._current_path)
        self._refresh_list()
        self._update_pagination()
        self.path_navigated.emit(self._current_path)

    def _refresh_list(self):
        self._list.blockSignals(True)
        self._list.clear()

        for entry in self._entries:
            if not self._recursive_mode:
                # Apply live filter to non-recursive listing (files and folders)
                if self._include_filter or self._exclude_filter:
                    name = entry["name"]
                    if self._include_filter and not any(
                        _pat_match(name, pat) for pat in self._include_filter
                    ):
                        continue
                    if self._exclude_filter and any(
                        _pat_match(name, pat) for pat in self._exclude_filter
                    ):
                        continue

            if entry["isOmeZarr"]:
                icon = _ICON_ZARR
            elif entry["isDirectory"]:
                icon = _ICON_FOLDER
            else:
                icon = _ICON_FILE

            text = f"{icon} {entry['name']}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, entry)

            if self._mode == "conversion" and _is_selectable(entry):
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                state = Qt.CheckState.Checked if entry["path"] in self._checked_paths else Qt.CheckState.Unchecked
                item.setCheckState(state)
            elif self._mode == "conversion":
                # Items are user-checkable by default; a plain folder must not
                # be, or the S3 Select All (which goes by this flag) ticks it.
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)

            if self._mode == "zarr" and not entry["isDirectory"] and not entry["isOmeZarr"]:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)

            if self._mode == "zarr" and entry["path"] == self._open_store:
                self._mark_open(item, True)

            if self._mode == "output" and not entry["isDirectory"]:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)

            self._list.addItem(item)

        # Update info label
        if self._mode == "conversion" and hasattr(self, "_select_all_btn"):
            total_visible = self._list.count()
            n_total_selected = len(self._checked_paths)  # always the global selection count

            if self._recursive_mode:
                n_found = len(self._recursive_entries)
                n_found_selected = sum(1 for e in self._recursive_entries if e["path"] in self._checked_paths)
                self._filter_info_label.setText(
                    f"{n_found} found, {n_total_selected} selected total"
                )
            else:
                if (self._include_filter or self._exclude_filter) and not self._s3_mode:
                    all_entries = list_local(self._current_path)
                    matched = [
                        e for e in all_entries
                        if (
                            not self._include_filter
                            or any(_pat_match(e["name"], p) for p in self._include_filter)
                        ) and (
                            not self._exclude_filter
                            or not any(_pat_match(e["name"], p) for p in self._exclude_filter)
                        )
                    ]
                    n_found = len(matched)
                    self._filter_info_label.setText(
                        f"{n_found} found, {n_total_selected} selected total"
                    )
                else:
                    label = f"{n_total_selected} selected" if n_total_selected else ""
                    self._filter_info_label.setText(label)

        self._list.blockSignals(False)

    def _update_pagination(self):
        total_pages = max(1, -(-self._total // PAGE_SIZE))  # ceil division
        self._page_label.setText(f"{self._page + 1}/{total_pages} ({self._total})")
        self._prev_btn.setEnabled(self._page > 0)
        self._next_btn.setEnabled((self._page + 1) * PAGE_SIZE < self._total)
        visible = self._total > PAGE_SIZE
        self._prev_btn.setVisible(visible)
        self._next_btn.setVisible(visible)
        self._page_label.setVisible(visible)

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_path_entered(self):
        self._navigate(self._path_edit.text().strip())

    def _on_up(self):
        parent = get_parent(self._current_path)
        if parent:
            self._navigate(parent)

    def _on_home(self):
        self._navigate(os.path.expanduser("~"))

    def _show_recents_menu(self):
        recents = _load_recents()
        if not recents:
            return
        menu = QMenu(self)
        for path in recents:
            action = menu.addAction(f"{_ICON_FOLDER}  {path}")
            action.triggered.connect(lambda checked, p=path: self._navigate(p))
        menu.exec(self._recent_btn.mapToGlobal(self._recent_btn.rect().bottomLeft()))

    def _on_prev_page(self):
        if self._recursive_mode:
            if self._page > 0:
                self._show_recursive_page(self._page - 1)
        elif self._page > 0:
            self._navigate(self._current_path, self._page - 1)

    def _on_next_page(self):
        if self._recursive_mode:
            self._show_recursive_page(self._page + 1)
        else:
            self._navigate(self._current_path, self._page + 1)

    def _on_click_confirmed(self):
        if self._pending_click_path:
            self._set_open_store(self._pending_click_path)
            self.zarr_selected.emit(self._pending_click_path)
            self._pending_click_path = ""

    # ── Open-store marker (zarr mode) ─────────────────────────────────────────

    @staticmethod
    def _mark_open(item: QListWidgetItem, is_open: bool):
        font = item.font()
        font.setBold(is_open)
        item.setFont(font)
        item.setBackground(QBrush(_OPEN_STORE_BG) if is_open else QBrush())
        item.setToolTip("Open in the viewer" if is_open else "")

    def _set_open_store(self, path: str):
        """Mark *path* as the open store, restyling items in place.

        Rebuilding the list instead would reset its scroll position under the
        user's click.
        """
        self._open_store = path
        for row in range(self._list.count()):
            item = self._list.item(row)
            entry = item.data(Qt.ItemDataRole.UserRole)
            self._mark_open(item, bool(entry) and entry["path"] == path)

    def _scroll_to_open_store(self):
        for row in range(self._list.count()):
            item = self._list.item(row)
            entry = item.data(Qt.ItemDataRole.UserRole)
            if entry and entry["path"] == self._open_store:
                self._list.setCurrentItem(item)
                self._list.scrollToItem(item)
                return

    def _on_single_click(self, item: QListWidgetItem):
        """Single-click behaviour depends on mode.

        zarr mode : click OME-Zarr → schedule zarr_selected (cancelled if double-click follows)
        """
        entry: FileEntry = item.data(Qt.ItemDataRole.UserRole)
        if not entry:
            return

        if self._mode == "zarr" and entry["isOmeZarr"]:
            self._pending_click_path = entry["path"]
            self._click_timer.start()

    def _on_double_click(self, item: QListWidgetItem):
        # Cancel any pending single-click zarr load, since double-click means "navigate into"
        self._click_timer.stop()
        self._pending_click_path = ""
        entry: FileEntry = item.data(Qt.ItemDataRole.UserRole)
        if entry["isDirectory"] and not self._recursive_mode:
            self._navigate(entry["path"])

    def _on_select_all(self):
        """Select all items matching the current filter in the current directory."""
        if self._recursive_mode:
            for entry in self._recursive_entries:
                self._checked_paths.add(entry["path"])
        elif not self._s3_mode:
            all_entries = list_local(self._current_path)
            for entry in all_entries:
                if not _is_selectable(entry):
                    continue
                if self._include_filter or self._exclude_filter:
                    name = entry["name"]
                    if self._include_filter and not any(
                        _pat_match(name, pat) for pat in self._include_filter
                    ):
                        continue
                    if self._exclude_filter and any(
                        _pat_match(name, pat) for pat in self._exclude_filter
                    ):
                        continue
                self._checked_paths.add(entry["path"])
        else:
            self._list.blockSignals(True)
            for i in range(self._list.count()):
                item = self._list.item(i)
                if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                    item.setCheckState(Qt.CheckState.Checked)
                    entry: FileEntry = item.data(Qt.ItemDataRole.UserRole)
                    if entry:
                        self._checked_paths.add(entry["path"])
            self._list.blockSignals(False)
        self._refresh_list()
        self.selection_changed.emit(list(self._checked_paths))

    def _on_deselect_all(self):
        """Clear the entire selection regardless of current folder or filter."""
        self._checked_paths.clear()
        self._refresh_list()
        self.selection_changed.emit([])

    def _on_new_folder(self):
        """Prompt user for a folder name and create it under the current path."""
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "New Folder", "Folder name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        new_path = os.path.join(self._current_path, name)
        try:
            os.makedirs(new_path, exist_ok=True)
            self._navigate(self._current_path)  # refresh
        except OSError as exc:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", f"Could not create folder:\n{exc}")

    def _on_item_changed(self, item: QListWidgetItem):
        entry: FileEntry = item.data(Qt.ItemDataRole.UserRole)
        if entry is None:
            return
        if item.checkState() == Qt.CheckState.Checked:
            self._checked_paths.add(entry["path"])
        else:
            self._checked_paths.discard(entry["path"])
        self.selection_changed.emit(list(self._checked_paths))

    # ── Drag and drop (conversion and zarr modes) ─────────────────────────────

    @staticmethod
    def _local_drop_paths(mime) -> list[str]:
        """Local paths carried by a drag, in order and without duplicates.

        Normalised because Qt hands over ``C:/dir/f.nd2`` while the listing
        holds ``C:\\dir\\f.nd2``: selection is matched by string, so an
        unnormalised drop would neither show as ticked nor be deduplicated
        against the same file ticked by hand.
        """
        if not mime.hasUrls():
            return []
        paths = [os.path.normpath(url.toLocalFile())
                 for url in mime.urls() if url.isLocalFile()]
        return list(dict.fromkeys(paths))

    def _set_drop_highlight(self, active: bool):
        idle, hover = _DROP_HINTS[self._mode]
        self._drop_hint.setText(hover if active else idle)
        self._drop_hint.setStyleSheet(
            _DROP_HINT_STYLE_ACTIVE if active else _DROP_HINT_STYLE)
        self._list.setStyleSheet(_LIST_STYLE_DROP_ACTIVE if active else "")

    def dragEnterEvent(self, event):
        if self._local_drop_paths(event.mimeData()):
            event.acceptProposedAction()
            self._set_drop_highlight(True)
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if self._local_drop_paths(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._set_drop_highlight(False)

    def dropEvent(self, event):
        self._set_drop_highlight(False)
        paths = self._local_drop_paths(event.mimeData())
        if not paths:
            event.ignore()
            return
        event.acceptProposedAction()
        if self._mode == "zarr":
            self.open_dropped_store(paths)
        else:
            self.add_dropped_paths(paths)

    def add_dropped_paths(self, paths: list[str]) -> bool:
        """Add dropped files to the selection, as if ticked by hand.

        Refused while a filter is active: the browser then lists only matching
        entries, so a dropped file could be selected yet invisible, and a later
        filter change prunes the selection to the filtered listing.  Plain
        folders are skipped, as they cannot be ticked either; OME-Zarr stores
        are datasets and are kept.  Afterwards the browser opens the folder the
        drop came from, so the new ticks are on screen.
        """
        if self._include_filter or self._exclude_filter:
            QMessageBox.warning(
                self, "Filters are active",
                "Files cannot be dropped while include/exclude filters are "
                "set, because the browser only lists entries matching them and "
                "dropped files could end up selected but hidden.\n\n"
                "Clear the filters and drop the files again.")
            return False

        folders = [p for p in paths
                   if os.path.isdir(p) and not _is_ome_zarr_local(p)]
        inputs = [p for p in paths if p not in folders]
        if folders:
            names = "\n".join(f"  {os.path.basename(p)}" for p in folders)
            QMessageBox.warning(
                self, "Folders skipped",
                f"Folders cannot be selected as inputs, so these were "
                f"skipped:\n{names}\n\n"
                "To convert a folder's files, open it in the browser and use "
                "Select All.")
        if not inputs:
            return False

        self._checked_paths.update(inputs)
        self._navigate(os.path.dirname(inputs[0]) or inputs[0])
        self.selection_changed.emit(list(self._checked_paths))
        return True

    def open_dropped_store(self, paths: list[str]) -> bool:
        """Open a dropped OME-Zarr store, as if it had been clicked.

        The viewer shows one dataset at a time, so a drop must be exactly one
        store.  The browser moves to the folder holding it and marks the store,
        as a click does, so it is identifiable in the list.
        """
        if len(paths) != 1:
            QMessageBox.warning(
                self, "One store at a time",
                "Drop a single OME-Zarr store to inspect it.")
            return False
        path = paths[0]
        if not (os.path.isdir(path) and _is_ome_zarr_local(path)):
            QMessageBox.warning(
                self, "Not an OME-Zarr store",
                f"'{os.path.basename(path)}' is not an OME-Zarr store, so it "
                "cannot be inspected here. Convert it on the Convert page "
                "first.")
            return False

        # Open the page of the listing that holds the store, not page one: in a
        # long folder the marked store would otherwise be out of sight.
        parent = os.path.dirname(path) or path
        names = [e["path"] for e in list_local(parent)]
        page = names.index(path) // PAGE_SIZE if path in names else 0
        self._open_store = path
        self._navigate(parent, page)
        self._scroll_to_open_store()
        self.zarr_selected.emit(path)
        return True

    # ── Public helpers ────────────────────────────────────────────────────────

    def current_path(self) -> str:
        return self._current_path

    def selected_paths(self) -> list[str]:
        return list(self._checked_paths)

    def clear_selection(self):
        self._checked_paths.clear()
        self._refresh_list()
        self.selection_changed.emit([])

    def navigate_to(self, path: str):
        """Programmatically navigate to *path*."""
        self._navigate(path)

    def set_filters(self, include: str, exclude: str):
        """Apply include/exclude glob filters.

        Include patterns (e.g. ``*.tif``) trigger recursive mode: the entire
        directory tree is walked and all matching files are shown with relative
        paths.  Exclude-only patterns stay in the normal single-folder view and
        hide matching entries from the current page.
        """
        self._include_filter = [p.strip() for p in include.split(",") if p.strip()]
        self._exclude_filter = [p.strip() for p in exclude.split(",") if p.strip()]

        # Only go recursive when at least one include pattern contains a glob wildcard.
        # Plain strings (e.g. "tif") stay non-recursive and match via substring in the
        # current folder view.
        has_wildcard = any(any(c in pat for c in "*?[") for pat in self._include_filter)
        use_recursive = bool(self._include_filter) and has_wildcard and not self._s3_mode and self._mode == "conversion"
        if use_recursive:
            self._recursive_mode = True
            self._recursive_entries = list_local_recursive(
                self._current_path,
                include_patterns=self._include_filter or None,
                exclude_patterns=self._exclude_filter or None,
            )
            # Prune checked paths to only those that survived the new filter
            self._checked_paths &= {e["path"] for e in self._recursive_entries}
            self._show_recursive_page(0)
        else:
            was_recursive = self._recursive_mode
            self._recursive_mode = False
            self._recursive_entries = []
            if was_recursive:
                # Re-fetch normal directory listing so _entries isn't stale
                self._navigate(self._current_path)
            else:
                self._refresh_list()
                self._update_pagination()

    def _show_recursive_page(self, page: int):
        """Slice the full recursive result list and display the requested page."""
        page_entries, total = paginate(self._recursive_entries, page, PAGE_SIZE)
        self._page = page
        self._entries = page_entries
        self._total = total
        self._refresh_list()
        self._update_pagination()
