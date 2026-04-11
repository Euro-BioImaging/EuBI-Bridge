"""
Convert page — full conversion config UI with sidebar browser and run panel.

Layout:
  Left : SidebarBrowser(mode="conversion") — select input files/folders
  Right: QTabWidget (Cluster | Reader | Conversion | Downscaling | Metadata | Run)
         + Config management toolbar above tabs
"""
from __future__ import annotations

import os

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QDoubleSpinBox,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

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


def _labeled_spin(label: str, minimum: int, maximum: int, value: int, step: int = 1) -> tuple[QLabel, QSpinBox]:
    lbl = QLabel(label)
    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    spin.setValue(value)
    spin.setSingleStep(step)
    return lbl, spin


def _form_row(label: str, *widgets) -> QHBoxLayout:
    """Fixed-width label followed by one or more widgets — values align across rows."""
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
        self._build_ui()
        self._load_config_to_ui(load_config())
        # Populate parameter tree now so it's visible before the first run
        self._populate_param_tree(self._ui_to_config())
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
        self._output_edit.setToolTip("Output directory — edit directly or navigate the browser below")
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

        reset_btn = QPushButton("Reset")
        reset_btn.setFixedHeight(24)
        reset_btn.clicked.connect(self._on_reset_config)
        toolbar.addWidget(reset_btn)

        self._config_path_label = QLabel("")
        self._config_path_label.setStyleSheet("font-size: 9px; color: #888;")
        self._config_path_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(self._config_path_label, 1)

        right_layout.addLayout(toolbar)

        # Tabs
        self._tabs = QTabWidget()
        right_layout.addWidget(self._tabs)

        self._build_cluster_tab()
        self._build_reader_tab()
        self._build_conversion_tab()
        self._build_downscaling_tab()
        self._build_metadata_tab()
        self._build_run_tab()

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
        lay.addLayout(_row(QLabel("Max Workers:"), self._max_workers))

        _, self._queue_size = _labeled_spin("Queue Size:", 1, 500, 4)
        lay.addLayout(_row(QLabel("Queue Size:"), self._queue_size))

        _, self._max_concurrency = _labeled_spin("Max Concurrency:", 1, 128, 4)
        lay.addLayout(_row(QLabel("Max Concurrency:"), self._max_concurrency))

        _, self._region_size_mb = _labeled_spin("Region Size MB:", 1, 4096, 256)
        lay.addLayout(_row(QLabel("Region Size MB:"), self._region_size_mb))

        self._memory_per_worker = QLineEdit("1GB")
        lay.addLayout(_form_row("Memory/Worker:", self._memory_per_worker))

        self._use_local_dask = QCheckBox("Use Local Dask")
        lay.addWidget(self._use_local_dask)

        self._use_slurm = QCheckBox("Use SLURM")
        lay.addWidget(self._use_slurm)

        lay.addStretch()

    def _build_reader_tab(self):
        _, lay = self._scrolled_tab("Reader")

        self._read_all_scenes = QCheckBox("Read All Scenes")
        self._read_all_scenes.setChecked(True)
        lay.addWidget(self._read_all_scenes)

        self._scene_indices = QLineEdit()
        self._scene_indices.setPlaceholderText("0,1,2  (blank = all)")
        lay.addLayout(_form_row("Scene Indices:", self._scene_indices))
        self._read_all_scenes.toggled.connect(
            lambda c: self._scene_indices.setEnabled(not c)
        )
        self._scene_indices.setEnabled(False)

        self._read_all_tiles = QCheckBox("Read All Tiles")
        self._read_all_tiles.setChecked(True)
        lay.addWidget(self._read_all_tiles)

        self._mosaic_tile_indices = QLineEdit()
        self._mosaic_tile_indices.setPlaceholderText("0,1  (blank = all)")
        lay.addLayout(_form_row("Mosaic Tile Indices:", self._mosaic_tile_indices))
        self._read_all_tiles.toggled.connect(
            lambda c: self._mosaic_tile_indices.setEnabled(not c)
        )
        self._mosaic_tile_indices.setEnabled(False)

        self._read_as_mosaic = QCheckBox("Read as Mosaic")
        lay.addWidget(self._read_as_mosaic)

        for label, attr in [
            ("View Index:",        "_view_index"),
            ("Phase Index:",       "_phase_index"),
            ("Illumination Index:", "_illumination_index"),
            ("Rotation Index:",    "_rotation_index"),
            ("Sample Index:",      "_sample_index"),
        ]:
            edit = QLineEdit("0")
            setattr(self, attr, edit)
            lay.addLayout(_form_row(label, edit))

        lay.addStretch()

    def _build_conversion_tab(self):
        _, lay = self._scrolled_tab("Conversion")

        self._data_type = QComboBox()
        for t in ("auto", "uint8", "uint16", "uint32", "float32", "float64"):
            self._data_type.addItem(t)
        lay.addLayout(_form_row("Data Type:", self._data_type))

        for attr, label in [
            ("_verbose",    "Verbose"),
            ("_overwrite",  "Overwrite"),
            ("_squeeze",    "Squeeze Dimensions"),
            ("_save_omexml", "Save OME-XML"),
            ("_override_channel_names", "Override Channel Names"),
            ("_skip_dask",  "Skip Dask"),
        ]:
            cb = QCheckBox(label)
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
        comp_layout.addLayout(_form_row("Codec:", self._codec))

        self._comp_level = QSpinBox()
        self._comp_level.setRange(0, 22)
        self._comp_level.setValue(5)
        comp_layout.addLayout(_form_row("Level:", self._comp_level))

        self._blosc_group = QGroupBox("Blosc options")
        blosc_layout = QVBoxLayout(self._blosc_group)

        self._blosc_inner = QComboBox()
        for c in ("lz4", "lz4hc", "zstd", "zlib", "blosclz"):
            self._blosc_inner.addItem(c)
        blosc_layout.addLayout(_form_row("Inner codec:", self._blosc_inner))

        self._blosc_shuffle = QComboBox()
        for s in ("noshuffle", "shuffle", "bitshuffle"):
            self._blosc_shuffle.addItem(s)
        self._blosc_shuffle.setCurrentIndex(1)
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
        chunk_layout.addWidget(self._auto_chunk)

        self._target_chunk_mb = QDoubleSpinBox()
        self._target_chunk_mb.setRange(0.1, 2048.0)
        self._target_chunk_mb.setDecimals(2)
        self._target_chunk_mb.setSingleStep(0.1)
        self._target_chunk_mb.setValue(1.0)
        chunk_layout.addLayout(_form_row("Target Chunk MB:", self._target_chunk_mb))

        self._manual_chunk_widget = QWidget()
        mcw_layout = QVBoxLayout(self._manual_chunk_widget)
        mcw_layout.setContentsMargins(0, 0, 0, 0)
        mcw_layout.setSpacing(3)
        self._chunk_spins: dict[str, QSpinBox] = {}
        for dim, default in [("T", 1), ("C", 1), ("Z", 96), ("Y", 96), ("X", 96)]:
            sp = QSpinBox()
            sp.setRange(1, 4096)
            sp.setValue(default)
            self._chunk_spins[dim.lower()] = sp
            mcw_layout.addLayout(_form_row(f"Chunk {dim}:", sp))
        chunk_layout.addWidget(self._manual_chunk_widget)
        self._manual_chunk_widget.setVisible(False)

        self._auto_chunk.toggled.connect(lambda c: (
            self._target_chunk_mb.setEnabled(c),
            self._manual_chunk_widget.setVisible(not c),
        ))
        lay.addWidget(chunk_group)

        # Zarr Format (+ v3-only shard coefficients)
        fmt_group = QGroupBox("Zarr Format and Sharding")
        fmt_layout = QVBoxLayout(fmt_group)

        self._zarr_format = QComboBox()
        self._zarr_format.addItem("v2", 2)
        self._zarr_format.addItem("v3", 3)
        fmt_layout.addLayout(_form_row("Format version:", self._zarr_format))

        self._shard_widget = QWidget()
        shard_widget_layout = QVBoxLayout(self._shard_widget)
        shard_widget_layout.setContentsMargins(0, 0, 0, 0)
        shard_widget_layout.setSpacing(3)
        shard_label = QLabel("Shard Coefficients")
        shard_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #aaa; margin-top: 4px;")
        shard_widget_layout.addWidget(shard_label)
        self._shard_spins: dict[str, QSpinBox] = {}
        for dim, default in [("T", 1), ("C", 1), ("Z", 3), ("Y", 3), ("X", 3)]:
            sp = QSpinBox()
            sp.setRange(1, 256)
            sp.setValue(default)
            self._shard_spins[dim.lower()] = sp
            shard_widget_layout.addLayout(_form_row(f"Shard {dim}:", sp))
        fmt_layout.addWidget(self._shard_widget)
        self._shard_widget.setVisible(False)

        self._zarr_format.currentIndexChanged.connect(self._update_shard_state)
        self._update_shard_state()
        lay.addWidget(fmt_group)

        # Dim ranges
        range_group = QGroupBox("Dimension Ranges (start,stop)")
        range_layout = QVBoxLayout(range_group)
        self._range_edits: dict[str, QLineEdit] = {}
        for dim in ("T", "C", "Z", "Y", "X"):
            edit = QLineEdit()
            edit.setPlaceholderText("0,100")
            self._range_edits[dim.lower()] = edit
            range_layout.addLayout(_form_row(f"{dim} range:", edit))
        lay.addWidget(range_group)

        # Concatenation
        concat_group = QGroupBox("Concatenation")
        concat_layout = QVBoxLayout(concat_group)
        self._concat_edits: dict[str, QLineEdit] = {}
        for ax in ("Time", "Channel", "Z", "Y", "X"):
            edit = QLineEdit()
            edit.setPlaceholderText(f"e.g. _t for time tag")
            self._concat_edits[ax.lower()] = edit
            concat_layout.addLayout(_form_row(f"{ax} tag:", edit))
        self._concat_axes = QLineEdit()
        self._concat_axes.setPlaceholderText("e.g. t,c")
        concat_layout.addLayout(_form_row("Concat axes:", self._concat_axes))
        lay.addWidget(concat_group)

        lay.addStretch()

    def _update_shard_state(self):
        """Show shard coefficient controls only when Zarr v3 is selected."""
        self._shard_widget.setVisible(self._zarr_format.currentData() == 3)

    def _build_downscaling_tab(self):
        _, lay = self._scrolled_tab("Downscaling")

        self._downscale_method = QComboBox()
        for m in ("simple", "mean", "median", "min", "max", "mode"):
            self._downscale_method.addItem(m)
        lay.addLayout(_form_row("Method:", self._downscale_method))

        self._auto_detect_layers = QCheckBox("Auto-detect Layers")
        self._auto_detect_layers.setChecked(True)
        lay.addWidget(self._auto_detect_layers)

        self._layer_controls = QWidget()
        lc_layout = QVBoxLayout(self._layer_controls)
        lc_layout.setContentsMargins(0, 0, 0, 0)
        lc_layout.setSpacing(3)

        self._num_layers = QSpinBox()
        self._num_layers.setRange(1, 20)
        self._num_layers.setValue(4)
        lc_layout.addLayout(_form_row("Num Layers:", self._num_layers))

        self._min_dim_size = QSpinBox()
        self._min_dim_size.setRange(1, 1024)
        self._min_dim_size.setValue(64)
        lc_layout.addLayout(_form_row("Min Dim Size:", self._min_dim_size))

        lay.addWidget(self._layer_controls)
        self._layer_controls.setVisible(False)
        self._auto_detect_layers.toggled.connect(
            lambda c: self._layer_controls.setVisible(not c)
        )

        # Scale factors
        scale_group = QGroupBox("Scale Factors per Dimension")
        scale_layout = QVBoxLayout(scale_group)
        self._scale_spins: dict[str, QSpinBox] = {}
        defaults = {"t": 1, "c": 1, "z": 2, "y": 2, "x": 2}
        for dim in ("T", "C", "Z", "Y", "X"):
            sp = QSpinBox()
            sp.setRange(1, 16)
            sp.setValue(defaults[dim.lower()])
            self._scale_spins[dim.lower()] = sp
            scale_layout.addLayout(_form_row(f"Scale {dim}:", sp))
        lay.addWidget(scale_group)

        # Smart downscaling
        self._apply_smart = QCheckBox("Apply Smart Downscaling")
        lay.addWidget(self._apply_smart)

        self._smart_widget = QWidget()
        sw_layout = QVBoxLayout(self._smart_widget)
        sw_layout.setContentsMargins(0, 0, 0, 0)
        sw_layout.setSpacing(3)
        self._smart_spins: dict[str, QSpinBox] = {}
        for dim in ("Z", "Y", "X", "Time"):
            sp = QSpinBox()
            sp.setRange(1, 32)
            sp.setValue(2)
            sp.setSpecialValueText("auto")
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
        lay.addLayout(_form_row("Metadata Reader:", self._metadata_reader))

        self._channel_intensity = QComboBox()
        self._channel_intensity.addItems(["from_datatype", "from_array"])
        lay.addLayout(_form_row("Channel Intensity Limits:", self._channel_intensity))

        self._override_physical = QCheckBox("Override Physical Scale")
        lay.addWidget(self._override_physical)

        self._physical_widget = QWidget()
        pw_layout = QVBoxLayout(self._physical_widget)
        pw_layout.setContentsMargins(0, 0, 0, 0)
        pw_layout.setSpacing(3)
        self._phys_edits: dict[str, QLineEdit] = {}
        self._phys_units: dict[str, QComboBox] = {}
        space_units = ["micrometer", "nanometer", "millimeter", "centimeter", "meter"]
        time_units = ["second", "millisecond", "microsecond", "minute", "hour"]
        for ax, units in [("Time", time_units), ("Z", space_units), ("Y", space_units), ("X", space_units)]:
            edit = QLineEdit()
            edit.setFixedWidth(70)
            self._phys_edits[ax.lower()] = edit
            combo = QComboBox()
            combo.addItems(units)
            self._phys_units[ax.lower()] = combo
            pw_layout.addLayout(_form_row(f"{ax} scale:", edit, combo))

        lay.addWidget(self._physical_widget)
        self._physical_widget.setVisible(False)
        self._override_physical.toggled.connect(
            lambda c: self._physical_widget.setVisible(c)
        )

        lay.addStretch()

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

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        run_layout.addWidget(self._progress_bar)

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

        self._tabs.addTab(content, "Run")

    # ── Config load/save ──────────────────────────────────────────────────────

    def _load_config_to_ui(self, cfg: dict):
        """Populate all UI controls from a camelCase config dict."""
        c = cfg.get("cluster", {})
        self._max_workers.setValue(c.get("maxWorkers", 4))
        self._queue_size.setValue(c.get("queueSize", 4))
        self._max_concurrency.setValue(c.get("maxConcurrency", 4))
        self._region_size_mb.setValue(c.get("regionSizeMb", 256))
        self._memory_per_worker.setText(str(c.get("memoryPerWorker", "1GB")))
        self._use_local_dask.setChecked(c.get("useLocalDask", False))
        self._use_slurm.setChecked(c.get("useSlurm", False))

        r = cfg.get("reader", {})
        self._read_all_scenes.setChecked(r.get("readAllScenes", True))
        self._scene_indices.setText(r.get("sceneIndices", ""))
        self._read_all_tiles.setChecked(r.get("readAllTiles", True))
        self._mosaic_tile_indices.setText(r.get("mosaicTileIndices", ""))
        self._read_as_mosaic.setChecked(r.get("readAsMosaic", False))
        self._view_index.setText(str(r.get("viewIndex", "0")))
        self._phase_index.setText(str(r.get("phaseIndex", "0")))
        self._illumination_index.setText(str(r.get("illuminationIndex", "0")))
        self._rotation_index.setText(str(r.get("rotationIndex", "0")))
        self._sample_index.setText(str(r.get("sampleIndex", "0")))

        conv = cfg.get("conversion", {})
        fmt = conv.get("zarrFormat", 2)
        idx = self._zarr_format.findData(fmt)
        if idx >= 0:
            self._zarr_format.setCurrentIndex(idx)
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

        if "_configPath" in cfg:
            self._config_path = cfg["_configPath"]
            self._config_path_label.setText(os.path.basename(cfg["_configPath"]))

    def _ui_to_config(self) -> dict:
        """Read all UI controls and build a camelCase config dict."""
        return {
            "cluster": {
                "maxWorkers":      self._max_workers.value(),
                "queueSize":       self._queue_size.value(),
                "maxConcurrency":  self._max_concurrency.value(),
                "regionSizeMb":    self._region_size_mb.value(),
                "memoryPerWorker": self._memory_per_worker.text().strip(),
                "useLocalDask":    self._use_local_dask.isChecked(),
                "useSlurm":        self._use_slurm.isChecked(),
            },
            "reader": {
                "readAllScenes":     self._read_all_scenes.isChecked(),
                "sceneIndices":      self._scene_indices.text().strip(),
                "readAllTiles":      self._read_all_tiles.isChecked(),
                "mosaicTileIndices": self._mosaic_tile_indices.text().strip(),
                "readAsMosaic":      self._read_as_mosaic.isChecked(),
                "viewIndex":         self._view_index.text().strip(),
                "phaseIndex":        self._phase_index.text().strip(),
                "illuminationIndex": self._illumination_index.text().strip(),
                "rotationIndex":     self._rotation_index.text().strip(),
                "sampleIndex":       self._sample_index.text().strip(),
            },
            "conversion": {
                "zarrFormat":           self._zarr_format.currentData(),
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
            self._log.append_line(f"Config saved to {os.path.basename(self._config_path)}")
        except Exception as exc:
            self._log.append_line(f"ERROR saving config: {exc}")

    def _on_reset_config(self):
        try:
            cfg = reset_config(self._config_path or None)
            self._load_config_to_ui(cfg)
            self._log.append_line("Config reset to defaults.")
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

        for section in ("cluster", "reader", "conversion", "downscaling", "metadata", "concatenation"):
            if section in cfg:
                _add_section(section, cfg[section])

        self._param_tree.resizeColumnToContents(0)

    def _on_tab_changed(self, index: int):
        """Refresh parameter tree whenever the Run tab (index 5) becomes active."""
        if index == 5:
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
        self._log.clear()
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._run_status.setText("Running...")
        self._run_status.setStyleSheet("font-weight: bold; font-size: 11px; color: #4fc3f7;")

        self._worker = ConversionWorker(cfg, self)
        self._worker.log_line.connect(self._log.append_line)
        self._worker.progress.connect(self._progress_bar.setValue)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

        self._tabs.setCurrentIndex(5)  # Switch to Run tab

    def _on_stop(self):
        if self._worker:
            self._worker.cancel()
            QTimer.singleShot(3000, self._force_stop)
        self._run_status.setText("Stopping...")
        self._stop_btn.setEnabled(False)

    def _force_stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(1000)
        self._reset_run_ui("Stopped")

    def _on_finished(self):
        self._reset_run_ui("Done", success=True)
        self._progress_bar.setValue(100)

    def _on_failed(self, tb: str):
        self._log.append_line(f"ERROR: {tb}")
        self._reset_run_ui("Failed")

    def _reset_run_ui(self, status: str, success: bool = False):
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._run_status.setText(status)
        color = "#4caf50" if success else ("#ff6b6b" if status == "Failed" else "#aaa")
        self._run_status.setStyleSheet(f"font-weight: bold; font-size: 11px; color: {color};")
        self._worker = None

    # ── Public API ────────────────────────────────────────────────────────────

    def navigate_to(self, path: str):
        """Navigate the sidebar browser to *path*."""
        self._browser.navigate_to(path)
