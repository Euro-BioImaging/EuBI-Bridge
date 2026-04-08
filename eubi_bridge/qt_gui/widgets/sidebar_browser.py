"""
Persistent sidebar file/folder browser with pagination and OME-Zarr detection.

Two modes:
  mode="zarr"        — navigate filesystem; double-click an OME-Zarr → zarr_selected(path)
  mode="conversion"  — navigate filesystem; check files/dirs → selection_changed(paths)
"""
from __future__ import annotations

import fnmatch
import os

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from eubi_bridge.qt_gui.core.file_service import (
    PAGE_SIZE,
    FileEntry,
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


class SidebarBrowser(QWidget):
    """Persistent sidebar file browser.

    Signals (zarr mode):
        zarr_selected(str)          — user single-clicked an OME-Zarr store

    Signals (conversion mode):
        selection_changed(list[str])— checked paths changed

    Signals (all modes):
        path_navigated(str)         — current directory changed (any navigation)
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
            self._select_all_btn.setCheckable(True)
            self._select_all_btn.setToolTip("Select / deselect all visible items")
            self._select_all_btn.toggled.connect(self._on_select_all_toggled)
            sel_bar.addWidget(self._select_all_btn)
            self._filter_info_label = QLabel("")
            self._filter_info_label.setStyleSheet("font-size: 9px; color: #aaa;")
            sel_bar.addWidget(self._filter_info_label, 1)
            layout.addLayout(sel_bar)

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
                        fnmatch.fnmatch(name, pat) for pat in self._include_filter
                    ):
                        continue
                    if self._exclude_filter and any(
                        fnmatch.fnmatch(name, pat) for pat in self._exclude_filter
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

            if self._mode == "conversion":
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                state = Qt.CheckState.Checked if entry["path"] in self._checked_paths else Qt.CheckState.Unchecked
                item.setCheckState(state)

            if self._mode == "zarr" and not entry["isDirectory"] and not entry["isOmeZarr"]:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)

            if self._mode == "output" and not entry["isDirectory"]:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)

            self._list.addItem(item)

        # Update Select All button state
        if self._mode == "conversion" and hasattr(self, "_select_all_btn"):
            total_visible = self._list.count()
            if self._recursive_mode:
                n_total = len(self._recursive_entries)
                n_checked = sum(1 for e in self._recursive_entries if e["path"] in self._checked_paths)
                all_selected = n_total > 0 and n_checked == n_total
                self._filter_info_label.setText(
                    f"{n_total} file(s) found — {n_checked} selected"
                )
            else:
                checked_visible = sum(
                    1 for i in range(total_visible)
                    if self._list.item(i).checkState() == Qt.CheckState.Checked
                )
                all_selected = total_visible > 0 and checked_visible == total_visible
                if self._include_filter or self._exclude_filter:
                    self._filter_info_label.setText(f"filter active — {total_visible} shown")
                else:
                    self._filter_info_label.setText("")
            self._select_all_btn.blockSignals(True)
            self._select_all_btn.setChecked(all_selected)
            self._select_all_btn.setText("Deselect All" if all_selected else "Select All")
            self._select_all_btn.blockSignals(False)

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
            self.zarr_selected.emit(self._pending_click_path)
            self._pending_click_path = ""

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
        # Cancel any pending single-click zarr load — double-click means "navigate into"
        self._click_timer.stop()
        self._pending_click_path = ""
        entry: FileEntry = item.data(Qt.ItemDataRole.UserRole)
        if entry["isDirectory"] and not self._recursive_mode:
            self._navigate(entry["path"])

    def _on_select_all_toggled(self, checked: bool):
        """Check or uncheck items.

        In recursive mode selects *all* matching entries across all pages.
        In normal mode selects only the currently visible page.
        """
        if self._recursive_mode and checked:
            # Select every entry in the full recursive results, not just the current page
            for entry in self._recursive_entries:
                self._checked_paths.add(entry["path"])
        elif self._recursive_mode and not checked:
            for entry in self._recursive_entries:
                self._checked_paths.discard(entry["path"])
        else:
            self._list.blockSignals(True)
            for i in range(self._list.count()):
                item = self._list.item(i)
                if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                    item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
                    entry: FileEntry = item.data(Qt.ItemDataRole.UserRole)
                    if entry:
                        if checked:
                            self._checked_paths.add(entry["path"])
                        else:
                            self._checked_paths.discard(entry["path"])
            self._list.blockSignals(False)
        self._select_all_btn.setText("Deselect All" if checked else "Select All")
        # Re-render current page to reflect new check state
        self._refresh_list()
        self.selection_changed.emit(list(self._checked_paths))

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

        use_recursive = bool(self._include_filter) and not self._s3_mode and self._mode == "conversion"
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
