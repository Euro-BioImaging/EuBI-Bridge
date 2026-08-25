"""
Convert page: full conversion config UI with sidebar browser and run panel.

Layout:
  Left : SidebarBrowser(mode="conversion"), select input files/folders
  Right: QTabWidget (Cluster | Reader | Conversion | Downscaling | Metadata | Run)
         + Config management toolbar above tabs
"""
from __future__ import annotations

import os
from copy import deepcopy

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QDoubleSpinBox,
    QSpinBox,
    QSplitter,
    QTabBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from eubi_bridge.utils.metadata_utils import (
    DEFAULT_CHANNEL_COLORS, auto_channel_color)
from eubi_bridge.qt_gui.core.batch import (
    CONFIG_SNAPSHOT_NAME,
    DEFAULT_BATCH_NAME,
    BatchModel,
    can_batch,
    make_baseline,
    to_cell,
    column_header,
    parameter_tabs,
    with_separators,
    SEPARATOR,
    uneditable_reason,
)
from eubi_bridge.qt_gui.widgets.batch_cell_editor import (
    BatchCellEditor, _ADD_PARAMETER)
from eubi_bridge.qt_gui.widgets.grouped_header import GroupedHeaderView
from eubi_bridge.qt_gui.core.config import (
    DEFAULT_CONFIG_DIR,
    load_config,
    reset_config,
    save_config,
)
from eubi_bridge.qt_gui.widgets.log_widget import LogWidget
from eubi_bridge.qt_gui.widgets.sidebar_browser import SidebarBrowser
from eubi_bridge.qt_gui.workers.conversion_worker import ConversionWorker

# Pre-import dask_jobqueue in the main thread so its module-level
# signal.signal() call (in dask_jobqueue/runner.py) never runs inside a
# QThread, which would raise "signal only works in main thread".
try:
    import dask_jobqueue as _dask_jobqueue  # noqa: F401
except Exception:
    pass


# ── Small helpers ─────────────────────────────────────────────────────────────

# Fixed pixel width shared by all form-row labels so values align vertically.
_LABEL_W = 256

# Parameter tabs that are always present (Cluster … Metadata).  The final tab is
# Run or Batch depending on the execution mode, so it always sits at _LAST_TAB.
_N_PARAM_TABS = 5
_LAST_TAB = _N_PARAM_TABS
_MODE_RUN, _MODE_BATCH = 0, 1


def _labeled_spin(label: str, minimum: int, maximum: int, value: int, step: int = 1) -> tuple[QLabel, QSpinBox]:
    lbl = QLabel(label)
    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    spin.setValue(value)
    spin.setSingleStep(step)
    return lbl, spin


def _form_row(label: str, *widgets) -> QHBoxLayout:
    """Fixed-width label followed by one or more widgets, so values align across rows."""
    h = QHBoxLayout()
    h.setSpacing(4)
    lbl = QLabel(label)
    lbl.setFixedWidth(_LABEL_W)
    h.addWidget(lbl)
    for w in widgets:
        h.addWidget(w)
    return h


def _row(*widgets) -> QHBoxLayout:
    h = QHBoxLayout()
    h.setSpacing(4)
    _first_label_fixed = False
    for w in widgets:
        if isinstance(w, int) and w == 0:
            h.addStretch()
        else:
            if isinstance(w, QLabel) and not _first_label_fixed:
                w.setFixedWidth(_LABEL_W)
                _first_label_fixed = True
            h.addWidget(w)
    return h


class ConvertPage(QWidget):
    """Full conversion page."""

    status_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: ConversionWorker | None = None
        self._config_path: str = ""
        self._batch = BatchModel()
        self._build_ui()
        self._active_log = self._log
        self._load_config_to_ui(load_config())
        # Populate parameter tree now so it's visible before the first run
        self._populate_param_tree(self._ui_to_config())
        self._sync_batch_comparison()
        self._refresh_batch_table()
        self._update_batch_availability()
        # Keep tree in sync whenever the user switches to the Run tab
        self._tabs.currentChanged.connect(self._on_tab_changed)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        # ── Left: input + output browsers ─────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(200)
        left.setMaximumWidth(340)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(2, 2, 2, 2)
        left_layout.setSpacing(4)

        left_split = QSplitter(Qt.Orientation.Vertical)
        left_layout.addWidget(left_split)

        # ── Input group ───────────────────────────────────────────────────────
        in_group = QGroupBox("Input")
        in_layout = QVBoxLayout(in_group)
        in_layout.setContentsMargins(6, 14, 6, 6)
        in_layout.setSpacing(4)

        # Include / exclude filters belong here (applied to input paths)
        inc_row = QHBoxLayout()
        inc_row.addWidget(QLabel("Include:"))
        self._include_edit = QLineEdit()
        self._include_edit.setPlaceholderText("*.tif,*.nd2")
        inc_row.addWidget(self._include_edit)
        in_layout.addLayout(inc_row)

        exc_row = QHBoxLayout()
        exc_row.addWidget(QLabel("Exclude:"))
        self._exclude_edit = QLineEdit()
        self._exclude_edit.setPlaceholderText("*thumb*")
        exc_row.addWidget(self._exclude_edit)
        in_layout.addLayout(exc_row)

        self._browser = SidebarBrowser(mode="conversion")
        self._browser.selection_changed.connect(self._on_selection_changed)
        self._browser.path_navigated.connect(self._on_input_path_navigated)
        in_layout.addWidget(self._browser)

        # Apply filters when user hits Enter in either filter field
        self._include_edit.returnPressed.connect(self._apply_filters)
        self._exclude_edit.returnPressed.connect(self._apply_filters)
        left_split.addWidget(in_group)

        # ── Output group ──────────────────────────────────────────────────────
        out_group = QGroupBox("Output")
        out_layout = QVBoxLayout(out_group)
        out_layout.setContentsMargins(6, 14, 6, 6)
        out_layout.setSpacing(4)

        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("Navigate below to set output path...")
        self._output_edit.setToolTip("Output directory: edit directly or navigate the browser below")
        out_layout.addWidget(self._output_edit)

        self._output_browser = SidebarBrowser(mode="output")
        self._output_browser.path_navigated.connect(self._output_edit.setText)
        out_layout.addWidget(self._output_browser)
        left_split.addWidget(out_group)

        left_split.setStretchFactor(0, 2)
        left_split.setStretchFactor(1, 1)

        splitter.addWidget(left)

        # ── Right: config toolbar + tabs ──────────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(2, 2, 2, 2)
        right_layout.setSpacing(4)

        # Config toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(4)
        load_btn = QPushButton("Load Config")
        load_btn.setFixedHeight(24)
        load_btn.clicked.connect(self._on_load_config)
        toolbar.addWidget(load_btn)

        save_btn = QPushButton("Save Config")
        save_btn.setFixedHeight(24)
        save_btn.clicked.connect(self._on_save_config)
        toolbar.addWidget(save_btn)

        revert_btn = QPushButton("Restore Current Config")
        revert_btn.setFixedHeight(24)
        revert_btn.setToolTip(
            "Discard unsaved edits and reload every parameter from the config "
            "file currently in use.\n"
            "Leaves the file untouched, unlike Reset to Installation Defaults."
        )
        revert_btn.clicked.connect(self._on_revert_config)
        toolbar.addWidget(revert_btn)

        reset_btn = QPushButton("Reset to Installation Defaults")
        reset_btn.setFixedHeight(24)
        reset_btn.setToolTip(
            "Reset every parameter to the defaults shipped with EuBI-Bridge and "
            "write them to the config file.\n"
            "Use Restore Current Config to go back to your saved settings instead."
        )
        reset_btn.clicked.connect(self._on_reset_config)
        toolbar.addWidget(reset_btn)

        self._config_path_label = QLabel("")
        self._config_path_label.setStyleSheet("font-size: 9px; color: #888;")
        self._config_path_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(self._config_path_label, 1)

        right_layout.addLayout(toolbar)

        # Execution mode, which chooses what the final parameter tab is.  Run converts
        # the current selection straight away; Batch queues conversions into a
        # table that is executed later.
        self._mode_bar = QTabBar()
        self._mode_bar.addTab("Run")
        self._mode_bar.addTab("Batch")
        self._mode_bar.setExpanding(False)
        self._mode_bar.setToolTip(
            "Run: convert the selected files immediately.\n"
            "Batch: queue conversions into a table and run them together later."
        )
        self._mode_bar.currentChanged.connect(self._on_mode_changed)
        right_layout.addWidget(self._mode_bar)

        # Tabs
        self._tabs = QTabWidget()
        right_layout.addWidget(self._tabs)

        self._build_cluster_tab()
        self._build_reader_tab()
        self._build_conversion_tab()
        self._build_downscaling_tab()
        self._build_metadata_tab()
        # Both are built up front but only the one matching the mode is attached.
        self._build_run_tab()
        self._build_batch_tab()
        self._apply_mode()

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    # ── Tab builders ──────────────────────────────────────────────────────────

    def _scrolled_tab(self, title: str) -> tuple[QScrollArea, QVBoxLayout]:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)
        scroll.setWidget(content)
        self._tabs.addTab(scroll, title)
        return scroll, lay

    def _build_cluster_tab(self):
        _, lay = self._scrolled_tab("Cluster")

        _, self._max_workers = _labeled_spin("Max Workers:", 1, 256, 4)
        self._max_workers.setToolTip(
            "Maximum number of parallel worker processes used for conversion.\n"
            "Higher values speed up batch jobs but consume more CPU and RAM."
        )
        lay.addLayout(_row(QLabel("Max Workers:"), self._max_workers))

        _, self._queue_size = _labeled_spin("Queue Size:", 1, 500, 4)
        self._queue_size.setToolTip(
            "Maximum number of conversion jobs that can be queued at once.\n"
            "Jobs beyond this limit are held until a worker slot opens."
        )
        lay.addLayout(_row(QLabel("Queue Size:"), self._queue_size))

        _, self._max_concurrency = _labeled_spin("Max Concurrency:", 1, 128, 4)
        self._max_concurrency.setToolTip(
            "Maximum number of chunk-write operations allowed to run concurrently\n"
            "inside a single worker. Reduce if you hit memory or I/O bottlenecks."
        )
        lay.addLayout(_row(QLabel("Max Concurrency:"), self._max_concurrency))

        _, self._max_concurrent_downscale_layers = _labeled_spin("Max Concurrent Downscale Layers:", 1, 16, 3)
        self._max_concurrent_downscale_layers.setToolTip(
            "How many pyramid levels to downscale simultaneously. "
            "Reduce for large 3D datasets to avoid OOM errors."
        )
        lay.addLayout(_row(QLabel("Max Concurrent Downscale Layers:"), self._max_concurrent_downscale_layers))

        _, self._max_concurrent_scenes = _labeled_spin("Max Concurrent Scenes:", 1, 64, 1)
        self._max_concurrent_scenes.setToolTip(
            "Maximum number of scenes (series) that are converted in parallel\n"
            "within a single multi-scene file. Reduce for very large scenes."
        )
        lay.addLayout(_row(QLabel("Max Concurrent Scenes:"), self._max_concurrent_scenes))

        self._region_size_mb = QDoubleSpinBox()
        self._region_size_mb.setRange(1.0, 65536.0)
        self._region_size_mb.setValue(256.0)
        self._region_size_mb.setSingleStep(32.0)
        self._region_size_mb.setDecimals(1)
        self._region_size_mb.setToolTip(
            "Size (in MB) of each read region when streaming pixel data from disk.\n"
            "Larger values can improve throughput at the cost of peak memory usage."
        )
        lay.addLayout(_form_row("Region Size (MB):", self._region_size_mb))

        self._memory_per_worker = QDoubleSpinBox()
        self._memory_per_worker.setRange(0.5, 1024.0)
        self._memory_per_worker.setValue(4.0)
        self._memory_per_worker.setSingleStep(1.0)
        self._memory_per_worker.setDecimals(1)
        self._memory_per_worker.setToolTip(
            "Memory (GB) requested per worker when using a Local Dask cluster or SLURM.\n"
            "Has no effect when running without a distributed cluster."
        )
        # Memory/Worker only applies to the LocalCluster / SLURM backends, so the
        # row is wrapped in a container and shown only when one of them is active.
        self._memory_row = QWidget()
        self._memory_row.setLayout(_form_row("Memory/Worker (GB):", self._memory_per_worker))
        lay.addWidget(self._memory_row)

        self._use_local_dask = QCheckBox("Use Local Dask")
        self._use_local_dask.setToolTip(
            "Spin up a Dask LocalCluster on this machine to parallelize conversion\n"
            "across multiple CPU cores. Useful for large files on a workstation."
        )
        lay.addWidget(self._use_local_dask)

        self._use_slurm = QCheckBox("Use SLURM")
        self._use_slurm.setToolTip(
            "Submit conversion workers to a SLURM HPC cluster.\n"
            "Configure partition, account, and time limit in the fields below."
        )
        lay.addWidget(self._use_slurm)

        def _update_memory_visibility(*_):
            self._memory_row.setVisible(
                self._use_local_dask.isChecked() or self._use_slurm.isChecked())
        self._use_local_dask.toggled.connect(_update_memory_visibility)
        self._use_slurm.toggled.connect(_update_memory_visibility)
        _update_memory_visibility()

        # SLURM-specific fields, wrapped in containers so setVisible hides label too
        def _slurm_row(label: str, widget: QWidget) -> QWidget:
            container = QWidget()
            container.setLayout(_form_row(label, widget))
            lay.addWidget(container)
            return container

        self._slurm_partition = QLineEdit()
        self._slurm_partition.setPlaceholderText("e.g. gpu, cpu (leave blank for default)")
        self._slurm_partition.setToolTip(
            "SLURM partition (queue) to submit jobs to.\n"
            "Leave blank to use the cluster's default partition."
        )
        _slurm_row_partition = _slurm_row("SLURM Partition:", self._slurm_partition)

        self._slurm_account = QLineEdit()
        self._slurm_account.setPlaceholderText("e.g. myproject (leave blank for default)")
        self._slurm_account.setToolTip(
            "SLURM billing account to charge compute time to.\n"
            "Leave blank if your cluster does not require an account."
        )
        _slurm_row_account = _slurm_row("SLURM Account:", self._slurm_account)

        self._slurm_time = QLineEdit("24:00:00")
        self._slurm_time.setPlaceholderText("HH:MM:SS")
        self._slurm_time.setToolTip(
            "Maximum wall-clock time for each SLURM worker job (HH:MM:SS).\n"
            "Jobs that exceed this limit are cancelled by the scheduler."
        )
        _slurm_row_time = _slurm_row("SLURM Time Limit:", self._slurm_time)

        self._slurm_sif_path = QLineEdit()
        self._slurm_sif_path.setPlaceholderText("e.g. /path/to/eubi-bridge.sif (leave blank to use host Python)")
        self._slurm_sif_path.setToolTip(
            "Path to an Apptainer/Singularity SIF container image.\n"
            "Workers will run inside this container on compute nodes.\n"
            "Leave blank to use the Python environment available on the nodes."
        )
        _slurm_row_sif = _slurm_row("Apptainer SIF:", self._slurm_sif_path)

        self._slurm_worker_timeout = QSpinBox()
        self._slurm_worker_timeout.setRange(1, 100000)
        self._slurm_worker_timeout.setValue(300)
        self._slurm_worker_timeout.setSuffix(" s")
        self._slurm_worker_timeout.setToolTip("Seconds to wait for SLURM workers to start.")
        _slurm_row_timeout = _slurm_row("Worker Start Timeout:", self._slurm_worker_timeout)

        self._slurm_rows = (_slurm_row_partition, _slurm_row_account, _slurm_row_time,
                            _slurm_row_sif, _slurm_row_timeout)

        def _toggle_slurm(checked: bool):
            for w in self._slurm_rows:
                w.setVisible(checked)
        self._use_slurm.toggled.connect(_toggle_slurm)
        _toggle_slurm(False)

        # ── Bio-Formats group ─────────────────────────────────────────────────
        bf_group = QGroupBox("Bio-Formats Settings")
        bf_lay = QVBoxLayout(bf_group)
        bf_lay.setSpacing(6)

        note = QLabel(
            "These settings apply only when the bfio/Bio-Formats fallback reader is "
            "active (e.g. MRC, BMP, and other formats without a native plugin) "
            "or when the Force Bio-Formats option is enabled in the Reader tab. "
            "They have no effect when a native reader (CZI, ND2, LIF, TIFF, IMS…) is used."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-style: italic;")
        bf_lay.addWidget(note)

        self._bf_tile_size_mb = QDoubleSpinBox()
        self._bf_tile_size_mb.setRange(1.0, 65536.0)
        self._bf_tile_size_mb.setValue(512.0)
        self._bf_tile_size_mb.setSingleStep(64.0)
        self._bf_tile_size_mb.setToolTip(
            "Size (MB) of each tile read by the Bio-Formats tiled reader.\n"
            "Larger tiles improve throughput; smaller tiles reduce peak memory."
        )
        bf_lay.addLayout(_form_row("BF Tile Size (MB):", self._bf_tile_size_mb))

        _, self._bf_read_concurrency = _labeled_spin("BF Read Concurrency:", 1, 64, 4)
        self._bf_read_concurrency.setToolTip(
            "Number of tile-read calls that Bio-Formats issues concurrently.\n"
            "Increase for fast NVMe storage; keep low for network file systems."
        )
        bf_lay.addLayout(_row(QLabel("BF Read Concurrency:"), self._bf_read_concurrency))

        self._jvm_memory = QDoubleSpinBox()
        self._jvm_memory.setRange(0.5, 512.0)
        self._jvm_memory.setValue(2.0)
        self._jvm_memory.setSingleStep(1.0)
        self._jvm_memory.setDecimals(1)
        self._jvm_memory.setToolTip(
            "Heap memory (GB) allocated to the Java Virtual Machine that\n"
            "Bio-Formats runs inside. Increase for very large files (e.g. > 10 GB)."
        )
        bf_lay.addLayout(_form_row("JVM Memory (GB):", self._jvm_memory))

        lay.addWidget(bf_group)

        lay.addStretch()

    def _build_reader_tab(self):
        _, lay = self._scrolled_tab("Reader")

        # ── Scenes ────────────────────────────────────────────────────────────
        scene_group = QGroupBox("Scenes")
        scene_lay = QVBoxLayout(scene_group)
        self._read_all_scenes = QCheckBox("Read All")
        self._read_all_scenes.setChecked(True)
        self._read_all_scenes.setToolTip(
            "Convert every scene (series) found in the file.\n"
            "Uncheck to specify a subset of scene indices below."
        )
        scene_lay.addWidget(self._read_all_scenes)
        self._scene_indices = QLineEdit()
        self._scene_indices.setPlaceholderText("0,1,2  (blank = all)")
        self._scene_indices.setToolTip(
            "Comma-separated list of scene (series) indices to convert.\n"
            "Only active when 'Read All' is unchecked. Example: 0,2,4"
        )
        scene_lay.addLayout(_form_row("Indices:", self._scene_indices))
        self._read_all_scenes.toggled.connect(
            lambda c: self._scene_indices.setEnabled(not c)
        )
        self._scene_indices.setEnabled(False)
        lay.addWidget(scene_group)

        # ── Tiles ─────────────────────────────────────────────────────────────
        tile_group = QGroupBox("Tiles")
        tile_lay = QVBoxLayout(tile_group)
        self._read_all_tiles = QCheckBox("Read All")
        self._read_all_tiles.setChecked(True)
        self._read_all_tiles.setToolTip(
            "Convert every tile in a tiled/mosaic acquisition.\n"
            "Uncheck to specify individual tile indices below."
        )
        tile_lay.addWidget(self._read_all_tiles)
        self._mosaic_tile_indices = QLineEdit()
        self._mosaic_tile_indices.setPlaceholderText("0,1  (blank = all)")
        self._mosaic_tile_indices.setToolTip(
            "Comma-separated list of tile (mosaic position) indices to convert.\n"
            "Only active when 'Read All' is unchecked. Example: 0,1"
        )
        tile_lay.addLayout(_form_row("Indices:", self._mosaic_tile_indices))
        self._read_all_tiles.toggled.connect(
            lambda c: self._mosaic_tile_indices.setEnabled(not c)
        )
        self._mosaic_tile_indices.setEnabled(False)
        self._read_as_mosaic = QCheckBox("Read as Mosaic (stitch tiles)")
        self._read_as_mosaic.setToolTip(
            "Stitch all tiles into a single continuous mosaic image at read time.\n"
            "When unchecked, each tile is saved as a separate OME-Zarr output."
        )
        tile_lay.addWidget(self._read_as_mosaic)
        lay.addWidget(tile_group)

        # ── Views ─────────────────────────────────────────────────────────────
        view_group = QGroupBox("Views")
        view_lay = QVBoxLayout(view_group)
        self._read_all_views = QCheckBox("Read All")
        self._read_all_views.setChecked(True)
        self._read_all_views.setToolTip(
            "Export every view (angle/camera position) in a multi-view acquisition.\n"
            "Uncheck to convert only the view indices specified below."
        )
        view_lay.addWidget(self._read_all_views)
        self._view_indices = QLineEdit()
        self._view_indices.setPlaceholderText("0,1  (blank = all)")
        self._view_indices.setToolTip(
            "Comma-separated list of view indices to convert.\n"
            "Only active when 'Read All' is unchecked. Example: 0,1"
        )
        view_lay.addLayout(_form_row("Indices:", self._view_indices))
        self._read_all_views.toggled.connect(
            lambda c: self._view_indices.setEnabled(not c)
        )
        self._view_indices.setEnabled(False)
        self._concat_views = QCheckBox("Concatenate along Channels")
        self._concat_views.setToolTip(
            "Merge multiple views into a single OME-Zarr output by stacking them\n"
            "along the channel axis, instead of writing one file per view."
        )
        view_lay.addWidget(self._concat_views)
        lay.addWidget(view_group)

        # ── Illuminations ─────────────────────────────────────────────────────
        illu_group = QGroupBox("Illuminations")
        illu_lay = QVBoxLayout(illu_group)
        self._read_all_illuminations = QCheckBox("Read All")
        self._read_all_illuminations.setChecked(True)
        self._read_all_illuminations.setToolTip(
            "Export every illumination direction in the acquisition.\n"
            "Uncheck to convert only the illumination indices specified below."
        )
        illu_lay.addWidget(self._read_all_illuminations)
        self._illumination_indices = QLineEdit()
        self._illumination_indices.setPlaceholderText("0,1  (blank = all)")
        self._illumination_indices.setToolTip(
            "Comma-separated list of illumination indices to convert.\n"
            "Only active when 'Read All' is unchecked. Example: 0,1"
        )
        illu_lay.addLayout(_form_row("Indices:", self._illumination_indices))
        self._read_all_illuminations.toggled.connect(
            lambda c: self._illumination_indices.setEnabled(not c)
        )
        self._illumination_indices.setEnabled(False)
        self._concat_illuminations = QCheckBox("Concatenate along Channels")
        self._concat_illuminations.setToolTip(
            "Merge multiple illuminations into a single OME-Zarr output by stacking\n"
            "them along the channel axis, instead of writing one file per illumination."
        )
        illu_lay.addWidget(self._concat_illuminations)
        lay.addWidget(illu_group)

        # ── Other Indices (Experimental) ──────────────────────────────────────
        other_group = QGroupBox("Other Indices (Experimental)")
        other_lay = QVBoxLayout(other_group)
        self._phase_index = QLineEdit("0")
        self._phase_index.setToolTip(
            "Phase index to read in formats that capture multiple\n"
            "acquisition phases (e.g. structured illumination microscopy)."
        )
        other_lay.addLayout(_form_row("Phase Index:", self._phase_index))
        _other_tooltips = {
            "_rotation_index": (
                "Rotation index for light-sheet or multi-angle formats that\n"
                "store each sample rotation as a separate dataset."
            ),
            "_sample_index": (
                "Sample index in formats where multiple biological samples\n"
                "are stored inside a single file."
            ),
        }
        for label, attr in [
            ("Rotation Index:", "_rotation_index"),
            ("Sample Index:",   "_sample_index"),
        ]:
            edit = QLineEdit("0")
            edit.setToolTip(_other_tooltips[attr])
            setattr(self, attr, edit)
            other_lay.addLayout(_form_row(label, edit))
        lay.addWidget(other_group)

        # ── Separator ─────────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        lay.addWidget(sep)

        # ── Reader Backend ────────────────────────────────────────────────────
        self._force_bioformats = QCheckBox("Force Bio-Formats (bfio tiled reader)")
        self._force_bioformats.setToolTip(
            "Force the bfio tiled reader even for natively-supported formats (CZI, ND2, LIF…).\n"
            "Useful when the native reader gives incorrect results."
        )
        lay.addWidget(self._force_bioformats)

        lay.addStretch()

    def _build_conversion_tab(self):
        _, lay = self._scrolled_tab("Conversion")

        self._data_type = QComboBox()
        for t in ("auto", "uint8", "uint16", "uint32", "float32", "float64"):
            self._data_type.addItem(t)
        self._data_type.setToolTip(
            "Output pixel data type.\n"
            "'auto' preserves the source bit-depth (recommended).\n"
            "Choose a specific type to cast pixel values during conversion."
        )
        lay.addLayout(_form_row("Data Type:", self._data_type))

        _cb_tooltips = {
            "_verbose": (
                "Print detailed progress messages for each chunk written.\n"
                "Useful for debugging; disable for cleaner logs in production."
            ),
            "_overwrite": (
                "Overwrite an existing OME-Zarr output at the destination path.\n"
                "When unchecked, conversion is skipped if the output already exists."
            ),
            "_squeeze": (
                "Remove size-1 dimensions from the output array.\n"
                "For example, a single time-point dataset stored as T=1,C,Z,Y,X\n"
                "becomes C,Z,Y,X when squeezed."
            ),
            "_save_omexml": (
                "Write a copy of the source OME-XML metadata as a sidecar\n"
                ".xml file alongside the OME-Zarr output."
            ),
            "_skip_dask": (
                "Read and write data synchronously without Dask.\n"
                "Useful for small files or when Dask overhead outweighs its benefit."
            ),
        }
        for attr, label in [
            ("_verbose",    "Verbose"),
            ("_overwrite",  "Overwrite"),
            ("_squeeze",    "Squeeze Dimensions"),
            ("_save_omexml", "Save OME-XML"),
            ("_skip_dask",  "Skip Dask"),
        ]:
            cb = QCheckBox(label)
            cb.setToolTip(_cb_tooltips[attr])
            setattr(self, attr, cb)
            lay.addWidget(cb)
        self._squeeze.setChecked(True)
        self._save_omexml.setChecked(True)

        # Compression
        comp_group = QGroupBox("Compression")
        comp_layout = QVBoxLayout(comp_group)

        self._codec = QComboBox()
        for c in ("blosc", "gzip", "zstd", "lz4", "none"):
            self._codec.addItem(c)
        self._codec.setToolTip(
            "Compression codec applied to every Zarr chunk.\n"
            "'blosc' is the fastest general-purpose choice (configurable below).\n"
            "'none' stores raw uncompressed data."
        )
        comp_layout.addLayout(_form_row("Codec:", self._codec))

        self._comp_level = QSpinBox()
        self._comp_level.setRange(0, 22)
        self._comp_level.setValue(5)
        self._comp_level.setToolTip(
            "Compression level (0 = fastest / least compressed, higher = smaller files).\n"
            "For blosc/lz4: 0–9. For gzip/zstd: 0–9 (gzip) or 0–22 (zstd)."
        )
        comp_layout.addLayout(_form_row("Level:", self._comp_level))

        self._blosc_group = QGroupBox("Blosc options")
        blosc_layout = QVBoxLayout(self._blosc_group)

        self._blosc_inner = QComboBox()
        for c in ("lz4", "lz4hc", "zstd", "zlib", "blosclz"):
            self._blosc_inner.addItem(c)
        self._blosc_inner.setToolTip(
            "Inner codec that Blosc uses to compress each chunk.\n"
            "'lz4' is very fast; 'zstd' gives better ratios at moderate speed."
        )
        blosc_layout.addLayout(_form_row("Inner codec:", self._blosc_inner))

        self._blosc_shuffle = QComboBox()
        for s in ("noshuffle", "shuffle", "bitshuffle"):
            self._blosc_shuffle.addItem(s)
        self._blosc_shuffle.setCurrentIndex(1)
        self._blosc_shuffle.setToolTip(
            "Byte/bit shuffle filter applied before compression.\n"
            "'shuffle' improves compression of integer data.\n"
            "'bitshuffle' works better for floating-point or noisy data."
        )
        blosc_layout.addLayout(_form_row("Shuffle:", self._blosc_shuffle))

        comp_layout.addWidget(self._blosc_group)
        lay.addWidget(comp_group)
        self._codec.currentTextChanged.connect(
            lambda t: self._blosc_group.setVisible(t == "blosc")
        )

        # Chunking
        chunk_group = QGroupBox("Chunking")
        chunk_layout = QVBoxLayout(chunk_group)

        self._auto_chunk = QCheckBox("Auto Chunk")
        self._auto_chunk.setChecked(True)
        self._auto_chunk.setToolTip(
            "Automatically compute chunk sizes to hit the target chunk size (MB).\n"
            "Uncheck to specify chunk dimensions manually."
        )
        chunk_layout.addWidget(self._auto_chunk)

        self._target_chunk_mb = QDoubleSpinBox()
        self._target_chunk_mb.setRange(0.1, 2048.0)
        self._target_chunk_mb.setDecimals(2)
        self._target_chunk_mb.setSingleStep(0.1)
        self._target_chunk_mb.setValue(1.0)
        self._target_chunk_mb.setToolTip(
            "Desired uncompressed size (MB) for each Zarr chunk when Auto Chunk\n"
            "is enabled. Typically 0.5–4 MB gives good viewer performance."
        )
        chunk_layout.addLayout(_form_row("Target Chunk MB:", self._target_chunk_mb))

        self._manual_chunk_widget = QWidget()
        mcw_layout = QVBoxLayout(self._manual_chunk_widget)
        mcw_layout.setContentsMargins(0, 0, 0, 0)
        mcw_layout.setSpacing(3)
        self._chunk_spins: dict[str, QSpinBox] = {}
        _chunk_dim_tips = {
            "T": "Number of time-points per chunk. Usually 1 for time-lapse data.",
            "C": "Number of channels per chunk. Usually 1 to allow per-channel access.",
            "Z": "Number of z-planes per chunk. Larger values help volumetric rendering.",
            "Y": "Number of rows (pixels) per chunk in the Y direction.",
            "X": "Number of columns (pixels) per chunk in the X direction.",
        }
        for dim, default in [("T", 1), ("C", 1), ("Z", 96), ("Y", 96), ("X", 96)]:
            sp = QSpinBox()
            sp.setRange(1, 4096)
            sp.setValue(default)
            sp.setToolTip(_chunk_dim_tips[dim])
            self._chunk_spins[dim.lower()] = sp
            mcw_layout.addLayout(_form_row(f"Chunk {dim}:", sp))
        chunk_layout.addWidget(self._manual_chunk_widget)
        self._manual_chunk_widget.setVisible(False)

        # Turning auto-chunking on makes the manual sizes inert, so reset them
        # to their defaults rather than carrying a value the writer ignores.
        # The Batch dialog clears the same overrides for the same reason.
        self._auto_chunk.toggled.connect(lambda c: (
            self._target_chunk_mb.setEnabled(c),
            self._manual_chunk_widget.setVisible(not c),
            self._reset_manual_chunks() if c else None,
        ))
        lay.addWidget(chunk_group)

        # OME-Zarr version (the zarr container format, v2/v3, is derived from it
        # and kept internally for compatibility).
        fmt_group = QGroupBox("OME-Zarr Version and Sharding")
        fmt_layout = QVBoxLayout(fmt_group)

        # Item text = OME-Zarr (NGFF) version; item data = required zarr format.
        self._ome_zarr_version = QComboBox()
        self._ome_zarr_version.addItem("0.4", 2)
        self._ome_zarr_version.addItem("0.5", 3)
        self._ome_zarr_version.setToolTip(
            "OME-Zarr (NGFF) specification version for the output.\n"
            "v0.4 uses Zarr format 2 and is compatible with most viewers.\n"
            "v0.5 uses Zarr format 3 with optional sharding support."
        )
        fmt_layout.addLayout(_form_row("OME-Zarr Version:", self._ome_zarr_version))

        self._shard_widget = QWidget()
        shard_widget_layout = QVBoxLayout(self._shard_widget)
        shard_widget_layout.setContentsMargins(0, 0, 0, 0)
        shard_widget_layout.setSpacing(3)
        shard_label = QLabel("Shard Coefficients")
        shard_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #aaa; margin-top: 4px;")
        shard_widget_layout.addWidget(shard_label)
        self._shard_spins: dict[str, QSpinBox] = {}
        _shard_dim_tips = {
            "T": "Number of chunks along T grouped into one shard file.",
            "C": "Number of chunks along C grouped into one shard file.",
            "Z": "Number of chunks along Z grouped into one shard file.",
            "Y": "Number of chunks along Y grouped into one shard file.",
            "X": "Number of chunks along X grouped into one shard file.",
        }
        for dim, default in [("T", 1), ("C", 1), ("Z", 3), ("Y", 3), ("X", 3)]:
            sp = QSpinBox()
            sp.setRange(1, 256)
            sp.setValue(default)
            sp.setToolTip(
                f"{_shard_dim_tips[dim]}\n"
                "Sharding reduces file-system object count at the cost of\n"
                "requiring random-access reads within each shard file."
            )
            self._shard_spins[dim.lower()] = sp
            shard_widget_layout.addLayout(_form_row(f"Shard {dim}:", sp))
        fmt_layout.addWidget(self._shard_widget)
        self._shard_widget.setVisible(False)

        self._ome_zarr_version.currentIndexChanged.connect(self._update_shard_state)
        self._update_shard_state()
        lay.addWidget(fmt_group)

        # Dim ranges
        range_group = QGroupBox("Dimension Ranges (start,stop)")
        range_layout = QVBoxLayout(range_group)
        self._range_edits: dict[str, QLineEdit] = {}
        _range_dim_labels = {
            "T": "time-points", "C": "channels",
            "Z": "z-planes", "Y": "rows", "X": "columns",
        }
        for dim in ("T", "C", "Z", "Y", "X"):
            edit = QLineEdit()
            edit.setPlaceholderText("0,100")
            edit.setToolTip(
                f"Restrict conversion to a slice along {dim} ({_range_dim_labels[dim]}).\n"
                "Format: start,stop (Python slice, stop is exclusive). Leave blank to convert all."
            )
            self._range_edits[dim.lower()] = edit
            range_layout.addLayout(_form_row(f"{dim} range:", edit))
        lay.addWidget(range_group)

        # Concatenation
        concat_group = QGroupBox("Concatenation")
        self._concat_group = concat_group
        concat_layout = QVBoxLayout(concat_group)

        # Shown only in Batch mode.  Concatenation cannot be batched yet, and a
        # silently-ignored setting is worse than a disabled one.
        self._concat_batch_note = QLabel(
            "Not available in Batch mode. A batch is a table with one row per "
            "input file, so it can only describe one-to-one conversions. An "
            "aggregative job spans several files and has no row to live on. "
            "Switch to Run mode to concatenate, or clear these fields and batch "
            "the files individually."
        )
        self._concat_batch_note.setWordWrap(True)
        self._concat_batch_note.setStyleSheet(
            "color: #ffb74d; font-style: italic; font-size: 10px;")
        concat_layout.addWidget(self._concat_batch_note)
        self._concat_edits: dict[str, QLineEdit] = {}
        # Placeholder per axis: a single hardcoded example would show the time
        # tag in every field and read as though the others expected it too.
        _concat_ax_hints = {
            "Time":    "e.g. _t for the time tag",
            "Channel": "e.g. _c for the channel tag",
            "Z":       "e.g. _z for the z tag",
            "Y":       "e.g. _y for the Y tile tag",
            "X":       "e.g. _x for the X tile tag",
        }
        _concat_ax_tips = {
            "Time":    "Filename tag that identifies the time-point index in a file series (e.g. '_t').",
            "Channel": "Filename tag that identifies the channel index in a file series (e.g. '_c').",
            "Z":       "Filename tag that identifies the z-plane index in a file series (e.g. '_z').",
            "Y":       "Filename tag that identifies the Y tile position in a file series.",
            "X":       "Filename tag that identifies the X tile position in a file series.",
        }
        for ax in ("Time", "Channel", "Z", "Y", "X"):
            edit = QLineEdit()
            edit.setPlaceholderText(_concat_ax_hints[ax])
            edit.setToolTip(
                f"{_concat_ax_tips[ax]}\n"
                "Files whose names contain this tag are grouped and concatenated\n"
                "along the corresponding axis into a single OME-Zarr output."
            )
            self._concat_edits[ax.lower()] = edit
            concat_layout.addLayout(_form_row(f"{ax} tag:", edit))
        self._concat_axes = QLineEdit()
        self._concat_axes.setPlaceholderText("e.g. t,c")
        self._concat_axes.setToolTip(
            "Comma-separated list of axes along which to concatenate files.\n"
            "Only axes that have a tag set above will be concatenated.\n"
            "Example: 't,c' concatenates across time and channel."
        )
        concat_layout.addLayout(_form_row("Concat axes:", self._concat_axes))
        self._override_channel_names = QCheckBox("Override Channel Names")
        self._override_channel_names.setToolTip(
            "Replace channel names in the OME-Zarr metadata with names\n"
            "derived from the concatenation tag values (e.g. file-name fragments)."
        )
        concat_layout.addWidget(self._override_channel_names)
        lay.addWidget(concat_group)

        lay.addStretch()

    def _update_shard_state(self):
        """Show shard coefficient controls only when Zarr v3 is selected."""
        self._shard_widget.setVisible(self._ome_zarr_version.currentData() == 3)

    def _build_downscaling_tab(self):
        _, lay = self._scrolled_tab("Downscaling")

        self._downscale_method = QComboBox()
        for m in ("simple", "mean", "median", "min", "max", "mode"):
            self._downscale_method.addItem(m)
        self._downscale_method.setToolTip(
            "Algorithm used to downsample each pyramid level.\n"
            "'simple' (nearest-neighbour) is fastest.\n"
            "'mean' / 'median' are smoother but slower.\n"
            "'min' / 'max' / 'mode' preserve extreme or most-frequent values."
        )
        lay.addLayout(_form_row("Method:", self._downscale_method))

        self._keep_existing_resolutions = QCheckBox("Keep Existing Resolutions")
        self._keep_existing_resolutions.setToolTip(
            "If the input already carries its own multiscale pyramid (e.g. .ims, "
            ".zarr), write its existing resolution levels straight to the output "
            "instead of rebuilding the pyramid."
        )
        lay.addWidget(self._keep_existing_resolutions)

        self._auto_detect_layers = QCheckBox("Auto-detect Layers")
        self._auto_detect_layers.setChecked(True)
        self._auto_detect_layers.setToolTip(
            "Automatically determine the number of pyramid levels so that\n"
            "the smallest level fits within the Min Dim Size threshold.\n"
            "Uncheck to specify the number of levels manually."
        )
        lay.addWidget(self._auto_detect_layers)

        self._layer_controls = QWidget()
        lc_layout = QVBoxLayout(self._layer_controls)
        lc_layout.setContentsMargins(0, 0, 0, 0)
        lc_layout.setSpacing(3)

        self._num_layers = QSpinBox()
        self._num_layers.setRange(1, 20)
        self._num_layers.setValue(4)
        self._num_layers.setToolTip(
            "Fixed number of downscaled pyramid levels to generate.\n"
            "Only active when Auto-detect Layers is unchecked."
        )
        lc_layout.addLayout(_form_row("Num Layers:", self._num_layers))

        lay.addWidget(self._layer_controls)
        self._layer_controls.setVisible(False)
        # Auto-detection makes an explicit layer count inert; reset it so the
        # form never shows a number that will not be used.
        self._auto_detect_layers.toggled.connect(
            lambda c: (
                self._layer_controls.setVisible(not c),
                self._num_layers.setValue(4) if c else None,
            )
        )

        self._min_dim_size = QSpinBox()
        self._min_dim_size.setRange(1, 1024)
        self._min_dim_size.setValue(64)
        self._min_dim_size.setToolTip(
            "Stop generating pyramid levels once the smallest spatial\n"
            "dimension (Y or X) would fall below this size (pixels)."
        )
        lay.addLayout(_form_row("Min Dim Size:", self._min_dim_size))

        # Scale factors
        scale_group = QGroupBox("Scale Factors per Dimension")
        scale_layout = QVBoxLayout(scale_group)
        self._scale_spins: dict[str, QSpinBox] = {}
        defaults = {"t": 1, "c": 1, "z": 2, "y": 2, "x": 2}
        _scale_dim_tips = {
            "T": "Downscale factor along T (time). Usually 1, since time is rarely downscaled.",
            "C": "Downscale factor along C (channels). Usually 1.",
            "Z": "Downscale factor along Z per level. 2 = halve the number of z-planes each level.",
            "Y": "Downscale factor along Y per level. 2 = halve image height each level.",
            "X": "Downscale factor along X per level. 2 = halve image width each level.",
        }
        for dim in ("T", "C", "Z", "Y", "X"):
            sp = QSpinBox()
            sp.setRange(1, 16)
            sp.setValue(defaults[dim.lower()])
            sp.setToolTip(_scale_dim_tips[dim])
            self._scale_spins[dim.lower()] = sp
            scale_layout.addLayout(_form_row(f"Scale {dim}:", sp))
        lay.addWidget(scale_group)

        # Smart downscaling
        self._apply_smart = QCheckBox("Apply Smart Downscaling")
        self._apply_smart.setToolTip(
            "Use anisotropy-aware downscaling: dimensions with coarser physical\n"
            "spacing (e.g. a thick z-step) are downscaled more slowly so the\n"
            "pyramid remains isotropic in physical space."
        )
        lay.addWidget(self._apply_smart)

        self._smart_widget = QWidget()
        sw_layout = QVBoxLayout(self._smart_widget)
        sw_layout.setContentsMargins(0, 0, 0, 0)
        sw_layout.setSpacing(3)
        self._smart_spins: dict[str, QSpinBox] = {}
        _smart_dim_tips = {
            "Z":    "Physical-space scale factor along Z used to compute anisotropy-aware downscaling.",
            "Y":    "Physical-space scale factor along Y used to compute anisotropy-aware downscaling.",
            "X":    "Physical-space scale factor along X used to compute anisotropy-aware downscaling.",
            "Time": "Physical-space scale factor along time used to compute anisotropy-aware downscaling.",
        }
        for dim in ("Z", "Y", "X", "Time"):
            sp = QSpinBox()
            sp.setRange(1, 32)
            sp.setValue(2)
            sp.setSpecialValueText("auto")
            sp.setToolTip(_smart_dim_tips[dim])
            self._smart_spins[dim.lower() if dim != "Time" else "time"] = sp
            sw_layout.addLayout(_form_row(f"Smart {dim}:", sp))
        lay.addWidget(self._smart_widget)
        self._smart_widget.setVisible(False)
        self._apply_smart.toggled.connect(
            lambda c: self._smart_widget.setVisible(c)
        )

        lay.addStretch()

    def _build_metadata_tab(self):
        _, lay = self._scrolled_tab("Metadata")

        self._metadata_reader = QComboBox()
        self._metadata_reader.addItems(["bfio", "bioio"])
        self._metadata_reader.setToolTip(
            "Library used to extract OME metadata (channel names, physical scales, etc.).\n"
            "'bfio' uses Bio-Formats via a Java bridge; 'bioio' is the pure-Python successor."
        )
        lay.addLayout(_form_row("Metadata Reader:", self._metadata_reader))

        self._channel_intensity = QComboBox()
        self._channel_intensity.addItems(["from_datatype", "from_array"])
        self._channel_intensity.setToolTip(
            "Source for the display intensity range stored in OME-Zarr channel metadata.\n"
            "'from_datatype' uses the full range of the pixel type (e.g. 0–65535 for uint16).\n"
            "'from_array' samples actual pixel values to find the min/max."
        )
        lay.addLayout(_form_row("Channel Intensity Limits:", self._channel_intensity))

        self._override_physical = QCheckBox("Override Physical Scale")
        self._override_physical.setToolTip(
            "Replace the physical pixel spacing read from file metadata with\n"
            "the values entered below. Useful when source metadata is missing or incorrect."
        )
        lay.addWidget(self._override_physical)

        self._physical_widget = QWidget()
        pw_layout = QVBoxLayout(self._physical_widget)
        pw_layout.setContentsMargins(0, 0, 0, 0)
        pw_layout.setSpacing(3)
        self._phys_edits: dict[str, QLineEdit] = {}
        self._phys_units: dict[str, QComboBox] = {}
        space_units = ["micrometer", "nanometer", "millimeter", "centimeter", "meter"]
        time_units = ["second", "millisecond", "microsecond", "minute", "hour"]
        _phys_ax_tips = {
            "Time": "Physical time interval between consecutive time-points (e.g. 1.0 for 1 second per frame).",
            "Z":    "Physical distance between consecutive z-planes (e.g. 0.5 for 500 nm z-step).",
            "Y":    "Physical pixel size along Y (e.g. 0.108 for 108 nm lateral pixel).",
            "X":    "Physical pixel size along X (e.g. 0.108 for 108 nm lateral pixel).",
        }
        for ax, units in [("Time", time_units), ("Z", space_units), ("Y", space_units), ("X", space_units)]:
            edit = QLineEdit()
            edit.setFixedWidth(70)
            edit.setToolTip(_phys_ax_tips[ax])
            self._phys_edits[ax.lower()] = edit
            combo = QComboBox()
            combo.addItems(units)
            combo.setToolTip(f"Unit for the {ax} physical scale value.")
            self._phys_units[ax.lower()] = combo
            pw_layout.addLayout(_form_row(f"{ax} scale:", edit, combo))

        lay.addWidget(self._physical_widget)
        self._physical_widget.setVisible(False)
        self._override_physical.toggled.connect(
            lambda c: self._physical_widget.setVisible(c)
        )

        # ── Channel colours ───────────────────────────────────────────────────
        colour_group = QGroupBox("Channel Colours")
        cg_layout = QVBoxLayout(colour_group)
        cg_layout.setSpacing(4)

        colour_note = QLabel(
            "Colours written into the OME-Zarr channel metadata. Leave a "
            "channel unticked to keep the colour stored in the input file, or "
            "an automatic one when the file specifies none. Tick a channel to "
            "replace it with your own colour. Channels beyond the rows below "
            "are always handled automatically."
        )
        colour_note.setWordWrap(True)
        colour_note.setStyleSheet(
            "color: gray; font-style: italic; font-size: 10px;")
        cg_layout.addWidget(colour_note)

        self._channel_colour_rows: list[dict] = []
        self._channel_colour_box = QWidget()
        self._channel_colour_layout = QVBoxLayout(self._channel_colour_box)
        self._channel_colour_layout.setContentsMargins(0, 0, 0, 0)
        self._channel_colour_layout.setSpacing(3)
        cg_layout.addWidget(self._channel_colour_box)

        # The form is built before any file is read, so the channel count is
        # unknown; show the length of the default palette and let the user add
        # more when their data has additional channels.
        for _ in range(len(DEFAULT_CHANNEL_COLORS) + 1):
            self._add_channel_colour_row()

        add_btn = QPushButton("Add channel")
        add_btn.setFixedWidth(110)
        add_btn.setToolTip("Add a row for another channel index.")
        add_btn.clicked.connect(lambda: self._add_channel_colour_row())
        cg_layout.addWidget(add_btn)

        lay.addWidget(colour_group)

        lay.addStretch()

    def _add_channel_colour_row(self):
        """Append one channel colour row: Override toggle plus a swatch.

        Unticked (the default) leaves the channel alone: the reader's own colour
        is used when the file carries one, and the automatic palette otherwise.
        Ticked replaces whatever the source said with the chosen colour.
        """
        index = len(self._channel_colour_rows)

        override = QCheckBox("Override existing")
        override.setChecked(False)
        override.setToolTip(
            "Off: keep the colour stored in the input file, or assign one\n"
            "automatically when the file does not specify one.\n"
            "On: use the colour chosen here, replacing any source colour.")

        swatch = QPushButton()
        swatch.setFixedSize(40, 18)
        swatch.setToolTip("Pick the colour to use for this channel.")
        row = {"index": index, "override": override, "swatch": swatch,
               "hex": auto_channel_color(index)}
        self._channel_colour_rows.append(row)

        def _paint():
            enabled = override.isChecked()
            # Greyed while inactive, so it is clear the swatch is only a preview
            # of the automatic choice and not what will be written.
            border = "#555" if enabled else "#3a3a3a"
            swatch.setStyleSheet(
                f"background-color: #{row['hex']}; border: 1px solid {border};")
            swatch.setEnabled(enabled)

        def _pick():
            chosen = QColorDialog.getColor(
                QColor(f"#{row['hex']}"), self,
                f"Channel {row['index']} colour")
            if chosen.isValid():
                row["hex"] = chosen.name()[1:].upper()
                _paint()

        swatch.clicked.connect(_pick)
        override.toggled.connect(lambda _=None: _paint())
        _paint()

        self._channel_colour_layout.addLayout(
            _form_row(f"Channel {index}:", override, swatch))

    def _channel_colours_to_string(self) -> str:
        """Serialise the non-auto rows to the CLI's ``idx,RRGGBB;...`` format."""
        parts = [f"{row['index']},{row['hex']}"
                 for row in self._channel_colour_rows
                 if row["override"].isChecked()]
        return ";".join(parts)

    def _load_channel_colours(self, value: str):
        """Apply an ``idx,RRGGBB;...`` string back onto the rows."""
        wanted: dict[int, str] = {}
        for pair in (value or "").split(";"):
            pair = pair.strip()
            if not pair or "," not in pair:
                continue
            idx_text, hex_text = pair.split(",", 1)
            try:
                wanted[int(idx_text)] = hex_text.strip().lstrip("#").upper()
            except ValueError:
                continue

        # A saved config may name channels beyond the rows currently shown.
        while wanted and len(self._channel_colour_rows) <= max(wanted):
            self._add_channel_colour_row()

        for row in self._channel_colour_rows:
            colour = wanted.get(row["index"])
            if colour:
                row["hex"] = colour
            else:
                # Preview the automatic choice rather than a stale pick.
                row["hex"] = auto_channel_color(row["index"])
            # Triggers the row's own repaint, keeping enabled/greyed consistent.
            row["override"].setChecked(bool(colour))
            row["override"].toggled.emit(bool(colour))

    def _build_run_tab(self):
        content = QWidget()
        run_layout = QVBoxLayout(content)
        run_layout.setContentsMargins(6, 6, 6, 6)
        run_layout.setSpacing(6)

        # Status + buttons + progress (always visible at top)
        self._run_status = QLabel("Ready")
        self._run_status.setStyleSheet("font-weight: bold; font-size: 11px;")
        run_layout.addWidget(self._run_status)

        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("Start")
        self._start_btn.setFixedHeight(30)
        self._start_btn.setStyleSheet("background: #2a7a3b; color: white; font-weight: bold;")
        self._start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(self._start_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setFixedHeight(30)
        self._stop_btn.setStyleSheet("background: #7a2a2a; color: white; font-weight: bold;")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self._stop_btn)

        refresh_params_btn = QPushButton("Show Current Params")
        refresh_params_btn.setFixedHeight(30)
        refresh_params_btn.setToolTip("Refresh the parameter tree below with the current UI settings")
        refresh_params_btn.clicked.connect(self._on_refresh_params)
        btn_row.addWidget(refresh_params_btn)
        run_layout.addLayout(btn_row)

        # Batching is unavailable for aggregative conversions, because table input is
        # one-to-one only.  Reflect that live rather than failing on click.
        self._concat_axes.textChanged.connect(self._update_batch_availability)


        # Input summary
        self._input_summary = QLabel("No files selected")
        self._input_summary.setStyleSheet("font-size: 10px; color: #aaa;")
        self._input_summary.setWordWrap(True)
        run_layout.addWidget(self._input_summary)

        # Splitter: parameter tree (top) | log (bottom)
        run_split = QSplitter(Qt.Orientation.Vertical)
        run_split.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Parameter tree
        self._param_tree = QTreeWidget()
        self._param_tree.setHeaderLabels(["Parameter", "Value"])
        self._param_tree.setColumnWidth(0, 180)
        self._param_tree.header().setStretchLastSection(True)
        self._param_tree.setAlternatingRowColors(True)
        self._param_tree.setStyleSheet("font-size: 10px;")
        self._param_tree.setMinimumHeight(80)
        run_split.addWidget(self._param_tree)

        # Log
        self._log = LogWidget()
        self._log.setMinimumHeight(60)
        run_split.addWidget(self._log)

        run_split.setStretchFactor(0, 1)
        run_split.setStretchFactor(1, 2)
        run_layout.addWidget(run_split)

        self._run_tab = content

    # ── Execution mode ────────────────────────────────────────────────────────

    def _apply_mode(self):
        """Attach the Run or Batch tab as the final parameter tab."""
        batch = self._mode_bar.currentIndex() == _MODE_BATCH
        was_last = self._tabs.currentIndex() == _LAST_TAB

        while self._tabs.count() > _N_PARAM_TABS:
            self._tabs.removeTab(_N_PARAM_TABS)   # widget survives; we hold a ref

        self._tabs.addTab(self._batch_tab if batch else self._run_tab,
                          "Batch" if batch else "Run")
        if was_last:
            self._tabs.setCurrentIndex(_LAST_TAB)

        # Grey out concatenation in Batch mode so it cannot look supported.
        # The values are left intact, so switching back to Run restores them.
        if hasattr(self, '_concat_group'):
            self._concat_group.setEnabled(not batch)
            self._concat_batch_note.setVisible(batch)
            self._concat_group.setToolTip(
                self._concat_batch_note.text() if batch else "")

    def _on_mode_changed(self, _index: int):
        self._apply_mode()
        self._tabs.setCurrentIndex(_LAST_TAB)

    # ── Batch tab ─────────────────────────────────────────────────────────────

    def _build_batch_tab(self):
        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        note = QLabel(
            "A batch queues several one-to-one conversions and runs them later. "
            "Configure a conversion on the other tabs, select your input files, "
            "then press ‘Add to Batch’ below. Each row stores only the settings "
            "you changed; a blank cell uses the config value. "
            "The table shows those changed settings by default, and the ‘Show’ "
            "boxes add whole parameter groups or every parameter at once. "
            "Select any cells and press ‘Edit Cells’ to change them, including "
            "the input and output paths. "
            "Aggregative (concatenation) conversions are not supported in batch "
            "mode yet, so use Run mode for those."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-style: italic; font-size: 10px;")
        lay.addWidget(note)

        # Batch-wide settings that no row can override.  Shown explicitly, since
        # they never appear as a table column and are otherwise unverifiable.
        self._batch_baseline = QLabel("")
        self._batch_baseline.setWordWrap(True)
        self._batch_baseline.setStyleSheet("font-size: 10px; color: #4fc3f7;")
        lay.addWidget(self._batch_baseline)

        # Row-editing buttons.  Add to Batch leads, since nothing else is usable
        # until the table has rows.
        edit_row = QHBoxLayout()
        self._add_batch_btn = QPushButton("Add to Batch")
        self._add_batch_btn.setStyleSheet("font-weight: bold;")
        self._add_batch_btn.setToolTip(
            "Queue the currently selected input files with the current settings.\n"
            "Change settings and add again to build up a batch of variants."
        )
        self._add_batch_btn.clicked.connect(self._on_add_to_batch)
        edit_row.addWidget(self._add_batch_btn)

        self._edit_cells_btn = QPushButton("Edit Cells")
        self._edit_cells_btn.setToolTip(
            "Edit the selected cells' parameters across every row they touch.")
        self._edit_cells_btn.setEnabled(False)
        self._edit_cells_btn.clicked.connect(self._on_batch_edit_cells)
        edit_row.addWidget(self._edit_cells_btn)

        for label, slot, tip in [
            ("Remove",    self._on_batch_remove,    "Remove the selected row from the batch"),
            ("Duplicate", self._on_batch_duplicate, "Copy the selected row, then edit one field to make a variant"),
            ("Move Up",   lambda: self._on_batch_move(-1), "Move the selected row earlier in the run order"),
            ("Move Down", lambda: self._on_batch_move(1),  "Move the selected row later in the run order"),
            ("Clear",     self._on_batch_clear,     "Discard every row and reset the batch baseline"),
        ]:
            btn = QPushButton(label)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            edit_row.addWidget(btn)
        edit_row.addStretch()
        lay.addLayout(edit_row)

        # View mode for the queue
        self._batch_full_table = QCheckBox("Full table (all parameters)")
        self._batch_full_table.setToolTip(
            "Off: each row stores only what differs from the config file; blank "
            "cells use the config value, keeping the table narrow and readable.\n"
            "On: every parameter is written on every row, so each row is fully "
            "self-describing. Values differing from the config file stay highlighted."
        )
        self._batch_full_table.toggled.connect(self._on_batch_full_toggled)

        # One toggle per category, with "all parameters" last: showing a whole
        # group is the common case, and picking the group is more intuitive than
        # hunting individual parameters.  A column whose value deviates from the
        # config is shown regardless of these, so nothing can be toggled away.
        show_row = QHBoxLayout()
        show_row.setSpacing(6)
        show_row.addWidget(QLabel("Show:"))
        self._batch_tab_toggles: dict[str, QCheckBox] = {}
        for tab in parameter_tabs():
            box = QCheckBox(tab)
            box.setToolTip(
                f"Show every {tab} parameter as a column.\n"
                "Parameters that differ from the config file are always shown.")
            box.toggled.connect(
                lambda checked, name=tab: self._on_batch_tab_toggled(
                    name, checked))
            self._batch_tab_toggles[tab] = box
            show_row.addWidget(box)
        show_row.addWidget(self._batch_full_table)
        show_row.addStretch(1)
        lay.addLayout(show_row)

        # The queue itself
        self._batch_table = QTableWidget(0, 0)
        # Three-row header (tab / group / parameter) mirroring the conversion
        # form's hierarchy; columns are already ordered so each group is one
        # contiguous run, which is what lets the labels span.
        self._batch_header = GroupedHeaderView(self._batch_table)
        self._batch_table.setHorizontalHeader(self._batch_header)
        self._batch_table.setAlternatingRowColors(True)
        self._batch_table.setStyleSheet("font-size: 10px;")
        # Cell-level multi-select drives Edit Cells; row operations still work
        # because _selected_batch_row() reads whichever rows the cells span.
        self._batch_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectItems)
        self._batch_table.setSelectionMode(
            QTableWidget.SelectionMode.ExtendedSelection)
        self._batch_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self._batch_table.itemSelectionChanged.connect(
            self._update_edit_cells_enabled)
        self._batch_table.setToolTip(
            "Queued conversions. A blank cell uses the value from the config "
            "snapshot saved alongside the CSV.\n"
            "Select cells and press Edit Cells to change them in bulk. "
            "Input and output paths can be edited the same way."
        )

        # Queue and log get their own sub-tabs.  In batch mode the Run tab is not
        # even attached, so the batch needs its own log surface rather than
        # writing into one the user cannot see.
        self._batch_subtabs = QTabWidget()
        self._batch_subtabs.addTab(self._batch_table, "Queue")

        log_page = QWidget()
        log_lay = QVBoxLayout(log_page)
        log_lay.setContentsMargins(0, 4, 0, 0)
        log_lay.setSpacing(4)

        self._batch_run_status = QLabel("Ready")
        self._batch_run_status.setStyleSheet("font-weight: bold; font-size: 11px;")
        log_lay.addWidget(self._batch_run_status)

        self._batch_log = LogWidget()
        log_lay.addWidget(self._batch_log)

        self._batch_subtabs.addTab(log_page, "Log")
        lay.addWidget(self._batch_subtabs)

        # Persistence + run
        io_row = QHBoxLayout()
        save_btn = QPushButton("Save Batch")
        save_btn.setToolTip("Validate and write batch.csv plus its config snapshot")
        save_btn.clicked.connect(self._on_batch_save)
        io_row.addWidget(save_btn)

        load_btn = QPushButton("Load Batch")
        load_btn.setToolTip("Open an existing batch.csv (and its config snapshot, if present)")
        load_btn.clicked.connect(self._on_batch_load)
        io_row.addWidget(load_btn)

        self._batch_run_btn = QPushButton("Run Batch")
        self._batch_run_btn.setFixedHeight(30)
        self._batch_run_btn.setStyleSheet("background: #2a7a3b; color: white; font-weight: bold;")
        self._batch_run_btn.setToolTip(
            "Save the batch, then convert every queued row.\n"
            "Equivalent to:  eubi to_zarr <batch.csv>"
        )
        self._batch_run_btn.clicked.connect(self._on_batch_run)
        io_row.addWidget(self._batch_run_btn)

        self._batch_stop_btn = QPushButton("Stop Batch")
        self._batch_stop_btn.setFixedHeight(30)
        self._batch_stop_btn.setStyleSheet("background: #7a2a2a; color: white; font-weight: bold;")
        self._batch_stop_btn.setEnabled(False)
        self._batch_stop_btn.setToolTip(
            "Cancel the running batch and kill its worker processes.\n"
            "Conversions already finished are kept; the current one is aborted."
        )
        self._batch_stop_btn.clicked.connect(self._on_stop)
        io_row.addWidget(self._batch_stop_btn)

        io_row.addStretch()
        lay.addLayout(io_row)

        self._batch_status = QLabel("Batch is empty")
        self._batch_status.setWordWrap(True)
        self._batch_status.setStyleSheet("font-size: 10px; color: #aaa;")
        lay.addWidget(self._batch_status)

        self._batch_tab = content

    # Defaults the manual chunk spins fall back to, matching the values
    # _load_config_to_ui applies for a config that has never set them.
    _MANUAL_CHUNK_DEFAULTS = {"t": 1, "c": 1, "z": 96, "y": 96, "x": 96}

    def _reset_manual_chunks(self):
        """Clear manual per-axis chunk sizes once auto-chunking owns them.

        Without this the form keeps whatever the user typed, `_ui_to_config`
        still reports it, and the Batch baseline captures a size that the writer
        ignores: a value that looks applied but is not.
        """
        for dim, value in self._MANUAL_CHUNK_DEFAULTS.items():
            spin = self._chunk_spins.get(dim)
            if spin is not None:
                spin.setValue(value)

    def _refresh_batch_table(self):
        """Rebuild the queue view from the model."""
        columns = self._batch.columns() if len(self._batch) else list(("input_path", "output_path"))
        # Thin blank gutters between categories: with dozens of columns the
        # header text alone does not make the boundaries readable.
        columns = with_separators(columns)
        self._batch_table.setColumnCount(len(columns))
        self._batch_table.setRowCount(len(self._batch))
        self._batch_table.setHorizontalHeaderLabels(
            ["" if key == SEPARATOR else column_header(key)[2]
             for key in columns])
        self._batch_header.set_hierarchy(
            [("", "", "") if key == SEPARATOR else column_header(key)
             for key in columns])

        for r, row in enumerate(self._batch.rows):
            for c, col in enumerate(columns):
                if col == SEPARATOR:
                    spacer = QTableWidgetItem("")
                    spacer.setFlags(Qt.ItemFlag.NoItemFlags)
                    self._batch_table.setItem(r, c, spacer)
                    continue
                # Render exactly as the CSV will, so the table previews the file
                # rather than Python's repr.
                text = to_cell(self._batch.cell(row, col))
                item = QTableWidgetItem(text)
                if col in ("input_path", "output_path"):
                    item.setToolTip(
                        f"{text}\nSelect and press Edit Cells to change it.")
                else:
                    # Inertness is per cell, not per column: one row switching
                    # auto-chunking off makes the manual sizes meaningful for
                    # that row alone, so the greying is applied cell by cell.
                    inert = self._batch.is_inert(row, col)
                    if inert:
                        reason = self._batch.inert_reason(row, col)
                        item.setForeground(QColor(120, 120, 120))
                        item.setFlags(
                            item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                        item.setToolTip(
                            f"Not used for this row: {reason} makes {col} "
                            f"inactive.")
                    elif self._batch.differs(row, col):
                        # Stands out in full-table mode, where most cells simply
                        # restate the config file.
                        item.setBackground(QColor(74, 110, 60))
                        item.setToolTip(
                            f"Differs from the config file: {col} = {text}")
                    elif text:
                        item.setToolTip(f"Same as the config file: {col} = {text}")
                self._batch_table.setItem(r, c, item)

        self._batch_table.resizeColumnsToContents()
        for index, key in enumerate(columns):
            if key == SEPARATOR:
                self._batch_table.setColumnWidth(index, 10)
        # Rebuilding drops the selection, so re-evaluate rather than leaving
        # Edit Cells enabled for cells that no longer exist.
        self._update_edit_cells_enabled()

        summary = self._batch.baseline_summary()
        self._batch_baseline.setText(
            f"Applies to every row: {summary}" if summary else "")

        n = len(self._batch)
        if n == 0:
            self._batch_status.setText("Batch is empty")
        else:
            n_over = len(self._batch.columns()) - 2
            self._batch_status.setText(
                f"{n} conversion(s) queued, "
                f"{n_over} per-row override column(s). "
                f"Blank cells use the config value."
            )

    def _selected_batch_row(self) -> int:
        """First row touched by the selection, or -1.

        Selection is per-cell, so ``selectedRows()`` (which only reports fully
        selected rows) would return nothing for a single-cell pick, so the row
        buttons read the selected indexes instead.
        """
        indexes = self._batch_table.selectionModel().selectedIndexes()
        return min((i.row() for i in indexes), default=-1)

    def _selected_batch_cells(self) -> tuple[list[int], list[str]]:
        """The rows and parameter columns the current selection spans."""
        indexes = self._batch_table.selectionModel().selectedIndexes()
        if not indexes:
            return [], []
        # Must mirror what _refresh_batch_table rendered, separators included,
        # or every column after the first gutter maps to the wrong parameter.
        columns = with_separators(self._batch.columns())
        rows = sorted({i.row() for i in indexes})
        keys: list[str] = []
        for column in sorted({i.column() for i in indexes}):
            if column < len(columns):
                key = columns[column]
                if key != SEPARATOR and key not in keys:
                    keys.append(key)
        return rows, keys

    def _update_edit_cells_enabled(self):
        """Edit Cells is live only when the selection holds an editable param."""
        rows, keys = self._selected_batch_cells()
        editable = [k for k in keys if uneditable_reason(k) is None]
        self._edit_cells_btn.setEnabled(bool(rows and editable))

    def _on_batch_edit_cells(self):
        rows, keys = self._selected_batch_cells()
        if not rows:
            return

        editable, blocked = [], []
        for key in keys:
            reason = uneditable_reason(key)
            if reason is None:
                editable.append(key)
            else:
                blocked.append(f"{key} ({reason})")

        if blocked:
            # Say why rather than dropping them silently: a column that ignores
            # an edit looks like a bug unless the constraint is stated.
            self._batch_log.append_line(
                "NOTE: not editable per row: " + "; ".join(blocked))
        if not editable:
            self._batch_status.setText(
                "Selected column(s) cannot be changed per row: "
                + "; ".join(blocked))
            self._batch_status.setStyleSheet("font-size: 10px; color: #ffb74d;")
            return

        # The dialog can ask to be re-opened with one more parameter, so the
        # user can reach settings the queue shows no column for yet.
        while True:
            dlg = BatchCellEditor(self._batch, rows, editable, parent=self)
            result = dlg.exec()
            if result == _ADD_PARAMETER and getattr(dlg, "added_key", None):
                editable = [*editable, dlg.added_key]
                continue
            break
        if not result:
            return
        applied = dlg.apply()
        if not applied:
            return

        self._refresh_batch_table()
        self._batch_log.append_line(
            f"Updated {', '.join(applied)} on {len(rows)} row(s).")
        self._batch_status.setText(
            f"Updated {len(applied)} parameter(s) on {len(rows)} row(s).")
        self._batch_status.setStyleSheet("font-size: 10px; color: #aaa;")

    def _batch_ui_config(self) -> dict:
        """UI config as a batch should see it, with concatenation neutralised.

        The Concatenation group is disabled in Batch mode but its values are
        deliberately preserved so switching back to Run does not lose them.  They
        must not leak into a batch, where they cannot be honoured.
        """
        cfg = self._ui_to_config()
        cfg["concatenation"] = {k: "" for k in cfg.get("concatenation", {})}
        return cfg

    def _update_batch_availability(self, *_):
        """Grey out 'Add to Batch' while an aggregative conversion is configured."""
        ok, reason = can_batch(self._batch_ui_config())
        self._add_batch_btn.setEnabled(ok)
        self._add_batch_btn.setToolTip(reason if not ok else (
            "Queue this conversion instead of running it now.\n"
            "Configure, add, repeat, then run them all from the Batch tab."
        ))

    # ── Batch callbacks ───────────────────────────────────────────────────────

    def _on_add_to_batch(self):
        selected = self._browser.selected_paths()
        output_path = self._output_edit.text().strip()

        if not selected:
            self._batch_log.append_line(
                "ERROR: No files selected, so nothing was added to the batch.")
            return
        if not output_path:
            self._batch_log.append_line("ERROR: No output path specified.")
            return

        cfg = self._batch_ui_config()

        ok, reason = can_batch(cfg)
        if not ok:
            self._batch_log.append_line(f"ERROR: {reason}")
            self._tabs.setCurrentIndex(_LAST_TAB)
            return

        # Anchor the batch to the config saved on disk, not to this first row.
        # Diffing the first row against itself would hide every setting the user
        # had already changed before pressing Add.
        if self._batch.base_config is None:
            try:
                persisted = load_config(self._config_path or None)
            except Exception as exc:
                self._batch_log.append_line(
                    f"NOTE: could not read the saved config ({exc}); "
                    "using the current settings as the batch baseline instead."
                )
                persisted = cfg
            # Anchor on the saved config so this row's deliberate changes stay
            # visible, but pin compression / ranges from the live UI.  No row can
            # carry those, so the baseline is the only place they can take effect.
            self._batch.set_baseline(make_baseline(persisted, cfg))

        blocked = self._batch.add(cfg, selected, output_path)

        self._refresh_batch_table()
        self._batch_log.append_line(
            f"Added {len(selected)} conversion(s) to the batch "
            f"({len(self._batch)} queued)."
        )
        self._tabs.setCurrentIndex(_LAST_TAB)  # Batch tab

        if blocked:
            # Shown on the Batch tab itself.  We just switched there, so a Run-tab
            # log line alone would go unread, and silently inheriting these would
            # produce output that does not match what the user configured.
            names = ", ".join(sorted(blocked))
            self._batch_log.append_line(
                f"WARNING: these cannot vary per row: {names}. "
                "The batch baseline applies to every row."
            )
            self._batch_status.setText(
                f"⚠ {len(self._batch)} row(s) queued, but these settings cannot "
                f"differ between rows and will use the batch baseline: {names}. "
                "Put them in a separate batch, or press Save Config first to make "
                "them the baseline."
            )
            self._batch_status.setStyleSheet("font-size: 10px; color: #ffb74d;")
        else:
            self._batch_status.setStyleSheet("font-size: 10px; color: #aaa;")

    def _on_batch_remove(self):
        i = self._selected_batch_row()
        if i < 0:
            self._batch_status.setText("Select a row first.")
            return
        self._batch.remove(i)
        self._refresh_batch_table()

    def _on_batch_duplicate(self):
        i = self._selected_batch_row()
        if i < 0:
            self._batch_status.setText("Select a row first.")
            return
        self._batch.duplicate(i)
        self._refresh_batch_table()
        self._batch_table.selectRow(i + 1)

    def _on_batch_move(self, delta: int):
        i = self._selected_batch_row()
        if i < 0:
            self._batch_status.setText("Select a row first.")
            return
        self._batch_table.selectRow(self._batch.move(i, delta))
        self._refresh_batch_table()

    def _on_batch_clear(self):
        self._batch.clear()
        self._refresh_batch_table()

    def _sync_batch_comparison(self):
        """Point batch highlighting at the currently selected config file."""
        try:
            self._batch.set_compare_config(load_config(self._config_path or None))
        except Exception:
            self._batch.set_compare_config(None)   # falls back to the baseline

    def _on_batch_tab_toggled(self, tab: str, checked: bool):
        """Show or hide a whole parameter category."""
        if checked:
            self._batch.shown_tabs.add(tab)
        else:
            self._batch.shown_tabs.discard(tab)
        self._refresh_batch_table()

    def _on_batch_full_toggled(self, checked: bool):
        # Rows always hold every parameter, so this only changes what is rendered
        # and written, so no data is lost either way.
        self._batch.full = checked
        self._refresh_batch_table()

    def _batch_save_to(self, path: str) -> str | None:
        """Validate then write the batch. Returns the CSV path, or None on failure."""
        problems = self._batch.validate()
        if problems:
            self._batch_status.setText(
                f"Cannot save: {len(problems)} problem(s); see the Log sub-tab."
            )
            self._batch_log.append_line("Batch validation failed:")
            for p in problems:
                self._batch_log.append_line(f"  • {p}")
            return None
        try:
            written = self._batch.save(path)
        except OSError as exc:
            self._batch_status.setText(f"Save failed: {exc}")
            return None
        self._batch_status.setText(
            f"Saved {len(self._batch)} row(s) to {os.path.basename(written)} "
            f"(+ {CONFIG_SNAPSHOT_NAME}) in {os.path.dirname(written)}"
        )
        return written

    def _on_batch_save(self):
        if not len(self._batch):
            self._batch_status.setText("Batch is empty, so there is nothing to save.")
            return
        default_path = os.path.join(DEFAULT_CONFIG_DIR, "batches", DEFAULT_BATCH_NAME)
        os.makedirs(os.path.dirname(default_path), exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Batch", default_path, "CSV files (*.csv);;All files (*)")
        if path:
            self._batch_save_to(path)

    def _on_batch_load(self):
        start_dir = os.path.join(DEFAULT_CONFIG_DIR, "batches")
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Batch", start_dir if os.path.isdir(start_dir) else DEFAULT_CONFIG_DIR,
            "CSV files (*.csv);;All files (*)")
        if not path:
            return
        try:
            self._batch = BatchModel.load(path)
        except Exception as exc:
            self._batch_status.setText(f"Load failed: {exc}")
            return
        self._refresh_batch_table()
        if self._batch.base_config is not None:
            self._load_config_to_ui(self._batch.base_config)
            self._batch_status.setText(
                f"Loaded {len(self._batch)} row(s) from {os.path.basename(path)}; "
                f"config snapshot applied to the parameter tabs."
            )
        else:
            self._batch_status.setText(
                f"Loaded {len(self._batch)} row(s) from {os.path.basename(path)}. "
                f"No {CONFIG_SNAPSHOT_NAME} found, so rows will use the current UI settings."
            )

    def _on_batch_run(self):
        if not len(self._batch):
            self._batch_status.setText("Batch is empty, so there is nothing to run.")
            return

        default_path = os.path.join(DEFAULT_CONFIG_DIR, "batches", DEFAULT_BATCH_NAME)
        os.makedirs(os.path.dirname(default_path), exist_ok=True)
        csv_path = self._batch_save_to(default_path)
        if csv_path is None:
            return

        # The batch baseline is the global config for the run; each CSV row
        # overrides it.  Pass the CSV as inputPath (a bare string) and leave
        # inputPaths empty, since take_filepaths() treats a list as explicit image
        # paths and would never reach its table branch.
        cfg = deepcopy(self._batch.base_config or self._ui_to_config())
        cfg["inputPaths"]     = []
        cfg["inputPath"]      = csv_path
        cfg["outputPath"]     = ""      # each row carries its own output_path
        cfg["includePattern"] = ""
        cfg["excludePattern"] = ""

        self._populate_param_tree(cfg)
        self._active_log = self._batch_log        # batch output has its own screen
        self._batch_log.clear()
        self._batch_log.append_line(
            f"Running batch: {csv_path} ({len(self._batch)} row(s))")

        self._start_btn.setEnabled(False)
        self._batch_run_btn.setEnabled(False)
        self._batch_stop_btn.setEnabled(True)
        self._stop_btn.setEnabled(True)
        self._set_run_status("Running batch...", "#4fc3f7")

        self._worker = ConversionWorker(cfg, self)
        self._worker.log_line.connect(self._batch_log.append_line)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

        self._tabs.setCurrentIndex(_LAST_TAB)
        self._batch_subtabs.setCurrentIndex(1)    # jump to the Log sub-tab

    # ── Config load/save ──────────────────────────────────────────────────────

    def _load_config_to_ui(self, cfg: dict):
        """Populate all UI controls from a camelCase config dict."""
        c = cfg.get("cluster", {})
        self._max_workers.setValue(c.get("maxWorkers", 4))
        self._queue_size.setValue(c.get("queueSize", 4))
        self._max_concurrency.setValue(c.get("maxConcurrency", 4))
        self._max_concurrent_downscale_layers.setValue(c.get("maxConcurrentDownscaleLayers", 3))
        self._max_concurrent_scenes.setValue(c.get("maxConcurrentScenes", 1))
        self._region_size_mb.setValue(float(c.get("regionSizeMb", 256.0)))
        self._memory_per_worker.setValue(float(c.get("memoryPerWorker", 4.0)))
        self._use_local_dask.setChecked(c.get("useLocalDask", False))
        self._use_slurm.setChecked(c.get("useSlurm", False))
        self._slurm_partition.setText(c.get("slurmPartition", ""))
        self._slurm_account.setText(c.get("slurmAccount", ""))
        self._slurm_time.setText(c.get("slurmTime", "24:00:00"))
        self._slurm_sif_path.setText(c.get("slurmSifPath", ""))
        self._slurm_worker_timeout.setValue(int(c.get("slurmWorkerTimeout", 300) or 300))
        self._bf_tile_size_mb.setValue(float(c.get("bfTileSizeMb", 512.0)))
        self._bf_read_concurrency.setValue(c.get("bfReadConcurrency", 4))
        self._jvm_memory.setValue(float(c.get("jvmMemory", 2.0)))

        r = cfg.get("reader", {})
        self._read_all_scenes.setChecked(r.get("readAllScenes", True))
        self._scene_indices.setText(r.get("sceneIndices", ""))
        self._read_all_tiles.setChecked(r.get("readAllTiles", True))
        self._mosaic_tile_indices.setText(r.get("mosaicTileIndices", ""))
        self._read_as_mosaic.setChecked(r.get("readAsMosaic", False))
        self._read_all_views.setChecked(r.get("readAllViews", True))
        self._view_indices.setText(r.get("viewIndices", ""))
        self._concat_views.setChecked(r.get("concatViews", False))
        self._phase_index.setText(str(r.get("phaseIndex", "0")))
        self._read_all_illuminations.setChecked(r.get("readAllIlluminations", True))
        self._illumination_indices.setText(r.get("illuminationIndices", ""))
        self._concat_illuminations.setChecked(r.get("concatIlluminations", False))
        self._rotation_index.setText(str(r.get("rotationIndex", "0")))
        self._sample_index.setText(str(r.get("sampleIndex", "0")))
        self._force_bioformats.setChecked(r.get("forceBioformats", False))

        conv = cfg.get("conversion", {})
        # Prefer the OME-Zarr version; fall back to the (deprecated) zarrFormat
        # for older saved configs (2 -> 0.4, 3 -> 0.5).
        ozv = conv.get("omeZarrVersion")
        if ozv:
            idx = self._ome_zarr_version.findText(str(ozv))
        else:
            idx = self._ome_zarr_version.findData(conv.get("zarrFormat", 2))
        if idx >= 0:
            self._ome_zarr_version.setCurrentIndex(idx)
        dt = conv.get("dataType", "auto") or "auto"
        idx = self._data_type.findText(dt)
        if idx >= 0:
            self._data_type.setCurrentIndex(idx)
        self._verbose.setChecked(conv.get("verbose", False))
        self._overwrite.setChecked(conv.get("overwrite", False))
        self._squeeze.setChecked(conv.get("squeezeDimensions", True))
        self._save_omexml.setChecked(conv.get("saveOmeXml", True))
        self._override_channel_names.setChecked(conv.get("overrideChannelNames", False))
        self._skip_dask.setChecked(conv.get("skipDask", False))
        self._auto_chunk.setChecked(conv.get("autoChunk", True))
        self._target_chunk_mb.setValue(float(conv.get("targetChunkSizeMb", 1.0)))
        for dim, key in [("t", "chunkTime"), ("c", "chunkChannel"), ("z", "chunkZ"), ("y", "chunkY"), ("x", "chunkX")]:
            self._chunk_spins[dim].setValue(conv.get(key, 96 if dim in ("z","y","x") else 1))
        for dim, key in [("t", "shardTime"), ("c", "shardChannel"), ("z", "shardZ"), ("y", "shardY"), ("x", "shardX")]:
            self._shard_spins[dim].setValue(conv.get(key, 3 if dim in ("z","y","x") else 1))
        for dim, key in [("t", "dimRangeTime"), ("c", "dimRangeChannel"), ("z", "dimRangeZ"), ("y", "dimRangeY"), ("x", "dimRangeX")]:
            self._range_edits[dim].setText(conv.get(key, ""))

        comp = conv.get("compression", {})
        codec = comp.get("codec", "blosc")
        idx = self._codec.findText(codec)
        if idx >= 0:
            self._codec.setCurrentIndex(idx)
        self._comp_level.setValue(comp.get("level", 5))
        inner = comp.get("bloscInnerCodec", "lz4")
        idx = self._blosc_inner.findText(inner)
        if idx >= 0:
            self._blosc_inner.setCurrentIndex(idx)
        shuffle = comp.get("bloscShuffle", "shuffle")
        idx = self._blosc_shuffle.findText(str(shuffle))
        if idx >= 0:
            self._blosc_shuffle.setCurrentIndex(idx)
        self._blosc_group.setVisible(codec == "blosc")

        down = cfg.get("downscaling", {})
        method = down.get("downscaleMethod", "simple")
        idx = self._downscale_method.findText(method)
        if idx >= 0:
            self._downscale_method.setCurrentIndex(idx)
        self._keep_existing_resolutions.setChecked(down.get("keepExistingResolutions", False))
        self._auto_detect_layers.setChecked(down.get("autoDetectLayers", True))
        self._num_layers.setValue(down.get("numLayers", 4))
        self._min_dim_size.setValue(down.get("minDimSize", 64))
        for dim, key in [("t", "scaleTime"), ("c", "scaleChannel"), ("z", "scaleZ"), ("y", "scaleY"), ("x", "scaleX")]:
            self._scale_spins[dim].setValue(down.get(key, 2 if dim in ("z","y","x") else 1))
        self._apply_smart.setChecked(down.get("applySmartDownscaling", False))
        for dim, key in [("z", "smartScaleZ"), ("y", "smartScaleY"), ("x", "smartScaleX"), ("time", "smartScaleTime")]:
            val = down.get(key)
            self._smart_spins[dim].setValue(val if val else 1)

        meta = cfg.get("metadata", {})
        idx = self._metadata_reader.findText(meta.get("metadataReader", "bfio"))
        if idx >= 0:
            self._metadata_reader.setCurrentIndex(idx)
        idx = self._channel_intensity.findText(meta.get("channelIntensityLimits", "from_datatype"))
        if idx >= 0:
            self._channel_intensity.setCurrentIndex(idx)
        self._override_physical.setChecked(meta.get("overridePhysicalScale", False))
        for ax in ("time", "z", "y", "x"):
            self._phys_edits[ax].setText(str(meta.get(f"scale{ax.capitalize()}", "")))
            unit_combo = self._phys_units[ax]
            unit_val = meta.get(f"unit{ax.capitalize()}", "")
            u_idx = unit_combo.findText(unit_val)
            if u_idx >= 0:
                unit_combo.setCurrentIndex(u_idx)
        self._load_channel_colours(meta.get("channelColors", ""))

        concat = cfg.get("concatenation", {})
        self._concat_edits["time"].setText(concat.get("timeTag", "") or "")
        self._concat_edits["channel"].setText(concat.get("channelTag", "") or "")
        self._concat_edits["z"].setText(concat.get("zTag", "") or "")
        self._concat_edits["y"].setText(concat.get("yTag", "") or "")
        self._concat_edits["x"].setText(concat.get("xTag", "") or "")
        self._concat_axes.setText(concat.get("concatenationAxes", "") or "")

        if "_configPath" in cfg:
            self._config_path = cfg["_configPath"]
            self._config_path_label.setText(os.path.basename(cfg["_configPath"]))

    def _ui_to_config(self) -> dict:
        """Read all UI controls and build a camelCase config dict."""
        return {
            "cluster": {
                "maxWorkers":          self._max_workers.value(),
                "queueSize":           self._queue_size.value(),
                "maxConcurrency":      self._max_concurrency.value(),
                "maxConcurrentDownscaleLayers": self._max_concurrent_downscale_layers.value(),
                "maxConcurrentScenes": self._max_concurrent_scenes.value(),
                "regionSizeMb":        self._region_size_mb.value(),
                "memoryPerWorker":     self._memory_per_worker.value(),
                "useLocalDask":        self._use_local_dask.isChecked(),
                "useSlurm":            self._use_slurm.isChecked(),
                "slurmPartition":      self._slurm_partition.text().strip(),
                "slurmAccount":        self._slurm_account.text().strip(),
                "slurmTime":           self._slurm_time.text().strip() or "24:00:00",
                "slurmSifPath":        self._slurm_sif_path.text().strip() or None,
                "slurmWorkerTimeout":  self._slurm_worker_timeout.value(),
                "bfTileSizeMb":        self._bf_tile_size_mb.value(),
                "bfReadConcurrency":   self._bf_read_concurrency.value(),
                "jvmMemory":           self._jvm_memory.value(),
            },
            "reader": {
                "readAllScenes":     self._read_all_scenes.isChecked(),
                "sceneIndices":      self._scene_indices.text().strip(),
                "readAllTiles":      self._read_all_tiles.isChecked(),
                "mosaicTileIndices": self._mosaic_tile_indices.text().strip(),
                "readAsMosaic":         self._read_as_mosaic.isChecked(),
                "readAllViews":         self._read_all_views.isChecked(),
                "viewIndices":          self._view_indices.text().strip(),
                "concatViews":          self._concat_views.isChecked(),
                "phaseIndex":           self._phase_index.text().strip(),
                "readAllIlluminations": self._read_all_illuminations.isChecked(),
                "illuminationIndices":  self._illumination_indices.text().strip(),
                "concatIlluminations":  self._concat_illuminations.isChecked(),
                "rotationIndex":        self._rotation_index.text().strip(),
                "sampleIndex":       self._sample_index.text().strip(),
                "forceBioformats":   self._force_bioformats.isChecked(),
            },
            "conversion": {
                "omeZarrVersion":       self._ome_zarr_version.currentText(),
                "dataType":             self._data_type.currentText(),
                "verbose":              self._verbose.isChecked(),
                "overwrite":            self._overwrite.isChecked(),
                "squeezeDimensions":    self._squeeze.isChecked(),
                "saveOmeXml":           self._save_omexml.isChecked(),
                "overrideChannelNames": self._override_channel_names.isChecked(),
                "skipDask":             self._skip_dask.isChecked(),
                "autoChunk":            self._auto_chunk.isChecked(),
                "targetChunkSizeMb":    self._target_chunk_mb.value(),
                "chunkTime":    self._chunk_spins["t"].value(),
                "chunkChannel": self._chunk_spins["c"].value(),
                "chunkZ":       self._chunk_spins["z"].value(),
                "chunkY":       self._chunk_spins["y"].value(),
                "chunkX":       self._chunk_spins["x"].value(),
                "shardTime":    self._shard_spins["t"].value(),
                "shardChannel": self._shard_spins["c"].value(),
                "shardZ":       self._shard_spins["z"].value(),
                "shardY":       self._shard_spins["y"].value(),
                "shardX":       self._shard_spins["x"].value(),
                "dimRangeTime":    self._range_edits["t"].text().strip(),
                "dimRangeChannel": self._range_edits["c"].text().strip(),
                "dimRangeZ":       self._range_edits["z"].text().strip(),
                "dimRangeY":       self._range_edits["y"].text().strip(),
                "dimRangeX":       self._range_edits["x"].text().strip(),
                "compression": {
                    "codec":          self._codec.currentText(),
                    "level":          self._comp_level.value(),
                    "bloscInnerCodec": self._blosc_inner.currentText(),
                    "bloscShuffle":   self._blosc_shuffle.currentText(),
                },
            },
            "downscaling": {
                "downscaleMethod":       self._downscale_method.currentText(),
                "keepExistingResolutions": self._keep_existing_resolutions.isChecked(),
                "autoDetectLayers":      self._auto_detect_layers.isChecked(),
                "numLayers":             self._num_layers.value(),
                "minDimSize":            self._min_dim_size.value(),
                "scaleTime":    self._scale_spins["t"].value(),
                "scaleChannel": self._scale_spins["c"].value(),
                "scaleZ":       self._scale_spins["z"].value(),
                "scaleY":       self._scale_spins["y"].value(),
                "scaleX":       self._scale_spins["x"].value(),
                "applySmartDownscaling": self._apply_smart.isChecked(),
                "smartScaleZ":    (self._smart_spins["z"].value() if self._smart_spins["z"].value() > 1 else None) if self._apply_smart.isChecked() else None,
                "smartScaleY":    (self._smart_spins["y"].value() if self._smart_spins["y"].value() > 1 else None) if self._apply_smart.isChecked() else None,
                "smartScaleX":    (self._smart_spins["x"].value() if self._smart_spins["x"].value() > 1 else None) if self._apply_smart.isChecked() else None,
                "smartScaleTime": (self._smart_spins["time"].value() if self._smart_spins["time"].value() > 1 else None) if self._apply_smart.isChecked() else None,
            },
            "metadata": {
                "metadataReader":         self._metadata_reader.currentText(),
                "channelIntensityLimits": self._channel_intensity.currentText(),
                "overridePhysicalScale":  self._override_physical.isChecked(),
                "scaleTime": self._phys_edits["time"].text().strip(),
                "unitTime":  self._phys_units["time"].currentText(),
                "scaleZ":    self._phys_edits["z"].text().strip(),
                "unitZ":     self._phys_units["z"].currentText(),
                "scaleY":    self._phys_edits["y"].text().strip(),
                "unitY":     self._phys_units["y"].currentText(),
                "scaleX":    self._phys_edits["x"].text().strip(),
                "unitX":     self._phys_units["x"].currentText(),
                "channelColors": self._channel_colours_to_string(),
            },
            "concatenation": {
                "timeTag":             self._concat_edits["time"].text().strip(),
                "channelTag":          self._concat_edits["channel"].text().strip(),
                "zTag":                self._concat_edits["z"].text().strip(),
                "yTag":                self._concat_edits["y"].text().strip(),
                "xTag":                self._concat_edits["x"].text().strip(),
                "concatenationAxes":   self._concat_axes.text().strip(),
            },
        }

    # ── Config management callbacks ───────────────────────────────────────────

    def _on_load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Config", DEFAULT_CONFIG_DIR, "JSON files (*.json);;All files (*)"
        )
        if path:
            try:
                cfg = load_config(path)
                self._load_config_to_ui(cfg)
                self._config_path = path
                self._config_path_label.setText(os.path.basename(path))
                self._sync_batch_comparison()
                self._refresh_batch_table()
            except Exception as exc:
                self._log.append_line(f"ERROR loading config: {exc}")

    def _on_save_config(self):
        # Compute the default save path: current config file, or default dir + "config.json"
        if self._config_path:
            default_path = self._config_path
        else:
            default_path = os.path.join(DEFAULT_CONFIG_DIR, "config.json")
        os.makedirs(os.path.dirname(default_path), exist_ok=True)

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Config",
            default_path,
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return  # user cancelled

        cfg = self._ui_to_config()
        try:
            saved = save_config(cfg, path)
            if "_configPath" in saved:
                self._config_path = saved["_configPath"]
            else:
                self._config_path = path
            self._config_path_label.setText(os.path.basename(self._config_path))
            self._sync_batch_comparison()
            self._refresh_batch_table()
            self._log.append_line(f"Config saved to {os.path.basename(self._config_path)}")
        except Exception as exc:
            self._log.append_line(f"ERROR saving config: {exc}")

    def _on_revert_config(self):
        """Reload every parameter from the config file currently in use.

        Distinct from Reset, which overwrites the file with the built-in
        defaults.  This only discards unsaved UI edits, so it is the quick way
        back to a known-good starting point without re-picking the file.
        """
        try:
            cfg = load_config(self._config_path or None)
        except Exception as exc:
            self._log.append_line(f"ERROR reverting config: {exc}")
            return

        self._load_config_to_ui(cfg)
        self._sync_batch_comparison()
        self._refresh_batch_table()
        self._update_batch_availability()
        self._populate_param_tree(self._ui_to_config())
        name = os.path.basename(self._config_path) if self._config_path else "the default config"
        self._log.append_line(f"Parameters reverted to {name}.")

    def _on_reset_config(self):
        try:
            cfg = reset_config(self._config_path or None)
            self._load_config_to_ui(cfg)
            self._sync_batch_comparison()
            self._refresh_batch_table()
            self._log.append_line("Config reset to installation defaults.")
        except Exception as exc:
            self._log.append_line(f"ERROR resetting config: {exc}")

    # ── Browser / output callbacks ────────────────────────────────────────────

    def _populate_param_tree(self, cfg: dict):
        """Rebuild the parameter tree from the full config dict."""
        self._param_tree.clear()

        def _add_section(title: str, params: dict):
            root = QTreeWidgetItem(self._param_tree, [title, ""])
            root.setExpanded(True)
            for key, val in params.items():
                if isinstance(val, dict):
                    sub = QTreeWidgetItem(root, [key, ""])
                    for k2, v2 in val.items():
                        if v2 not in (None, "", 0, False):
                            QTreeWidgetItem(sub, [f"  {k2}", str(v2)])
                elif val not in (None, "", 0, False):
                    QTreeWidgetItem(root, [key, str(val)])

        # Input / output
        io = QTreeWidgetItem(self._param_tree, ["I/O", ""])
        io.setExpanded(True)
        input_val = cfg.get("inputPaths") or cfg.get("inputPath", "")
        _MAX_LISTED = 10
        if isinstance(input_val, list):
            QTreeWidgetItem(io, ["input", f"{len(input_val)} path(s)"])
            for p in input_val[:_MAX_LISTED]:
                QTreeWidgetItem(io, ["  •", p])
            if len(input_val) > _MAX_LISTED:
                QTreeWidgetItem(io, ["  …", f"and {len(input_val) - _MAX_LISTED} more"])
        else:
            QTreeWidgetItem(io, ["input", str(input_val)])
        QTreeWidgetItem(io, ["output", cfg.get("outputPath", "")])
        if cfg.get("includePattern"):
            QTreeWidgetItem(io, ["include", cfg["includePattern"]])
        if cfg.get("excludePattern"):
            QTreeWidgetItem(io, ["exclude", cfg["excludePattern"]])

        for section in ("cluster", "reader", "concatenation"):
            if section in cfg:
                _add_section(section, cfg[section])

        # Conversion: hide individual chunk/shard sizes when autoChunk is on
        if "conversion" in cfg:
            conv = cfg["conversion"]
            if conv.get("autoChunk"):
                _CHUNK_KEYS = {
                    "chunkTime", "chunkChannel", "chunkZ", "chunkY", "chunkX",
                    "shardTime", "shardChannel", "shardZ", "shardY", "shardX",
                }
                conv = {k: v for k, v in conv.items() if k not in _CHUNK_KEYS}
            _add_section("conversion", conv)

        # Downscaling: hide numLayers when autoDetectLayers is on
        if "downscaling" in cfg:
            ds = cfg["downscaling"]
            if ds.get("autoDetectLayers"):
                ds = {k: v for k, v in ds.items() if k != "numLayers"}
            _add_section("downscaling", ds)

        # Metadata: hide scale/unit fields unless overridePhysicalScale is on
        if "metadata" in cfg:
            _SCALE_UNIT_KEYS = {
                "scaleTime", "unitTime", "scaleZ", "unitZ",
                "scaleY", "unitY", "scaleX", "unitX",
            }
            meta = cfg["metadata"]
            if not meta.get("overridePhysicalScale"):
                meta = {k: v for k, v in meta.items() if k not in _SCALE_UNIT_KEYS}
            _add_section("metadata", meta)

        self._param_tree.resizeColumnToContents(0)

    def _on_tab_changed(self, index: int):
        """Refresh the parameter tree whenever the final tab becomes active."""
        if index == _LAST_TAB:
            self._on_refresh_params()

    def _on_refresh_params(self):
        """Update the parameter tree with current UI state plus current I/O selections."""
        cfg = self._ui_to_config()
        cfg["inputPaths"]     = self._browser.selected_paths()
        cfg["inputPath"]      = self._browser.current_path()
        cfg["outputPath"]     = self._output_edit.text().strip()
        cfg["includePattern"] = self._include_edit.text().strip()
        cfg["excludePattern"] = self._exclude_edit.text().strip()
        self._populate_param_tree(cfg)

    def _apply_filters(self):
        """Push current include/exclude patterns to the input browser."""
        self._browser.set_filters(
            self._include_edit.text(),
            self._exclude_edit.text(),
        )

    def _on_input_path_navigated(self, _path: str):
        """Update summary when user navigates with no files checked."""
        if not self._browser.selected_paths():
            self._on_selection_changed([])

    def _on_selection_changed(self, paths: list[str]):
        n = len(paths)
        if n == 0:
            cur = self._browser.current_path()
            self._input_summary.setText(f"Using directory: {cur}" if cur else "No input selected")
        elif n == 1:
            self._input_summary.setText(f"1 item: {os.path.basename(paths[0])}")
        else:
            self._input_summary.setText(f"{n} items selected")

    # ── Conversion callbacks ──────────────────────────────────────────────────

    def _on_start(self):
        selected = self._browser.selected_paths()
        output_path = self._output_edit.text().strip()

        if not selected:
            self._log.append_line(
                "ERROR: No files selected. Use the input browser to check files, "
                "or apply filters and click \u2018Select All\u2019 to pick all matching files."
            )
            return
        if not output_path:
            self._log.append_line("ERROR: No output path specified.")
            return

        input_paths = selected

        cfg = self._ui_to_config()
        cfg["inputPaths"]      = input_paths
        cfg["inputPath"]       = ""
        cfg["outputPath"]      = output_path
        cfg["includePattern"]  = self._include_edit.text().strip()
        cfg["excludePattern"]  = self._exclude_edit.text().strip()

        self._populate_param_tree(cfg)
        self._active_log = self._log
        self._log.clear()

        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._run_status.setText("Running...")
        self._run_status.setStyleSheet("font-weight: bold; font-size: 11px; color: #4fc3f7;")

        self._worker = ConversionWorker(cfg, self)
        self._worker.log_line.connect(self._log.append_line)

        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

        self._tabs.setCurrentIndex(_LAST_TAB)  # Switch to Run tab

    def _on_stop(self):
        if self._worker:
            self._worker.cancel()
            QTimer.singleShot(3000, self._force_stop)
        self._set_run_status("Stopping...")
        self._stop_btn.setEnabled(False)

    def _force_stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(1000)
        self._reset_run_ui("Stopped")

    def _on_finished(self):
        self._reset_run_ui("Done", success=True)

    def _on_failed(self, tb: str):
        self._active_log.append_line(f"ERROR: {tb}")
        self._reset_run_ui("Failed")

    def _set_run_status(self, text: str, color: str = ""):
        """Update both status labels so either mode's surface stays accurate."""
        style = "font-weight: bold; font-size: 11px;"
        if color:
            style += f" color: {color};"
        for label in (self._run_status, self._batch_run_status):
            label.setText(text)
            label.setStyleSheet(style)

    def _reset_run_ui(self, status: str, success: bool = False):
        self._start_btn.setEnabled(True)
        self._batch_run_btn.setEnabled(True)
        self._batch_stop_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        color = "#4caf50" if success else ("#ff6b6b" if status == "Failed" else "#aaa")
        self._set_run_status(status, color)
        self._worker = None

    # ── Public API ────────────────────────────────────────────────────────────

    def navigate_to(self, path: str):
        """Navigate the sidebar browser to *path*."""
        self._browser.navigate_to(path)
