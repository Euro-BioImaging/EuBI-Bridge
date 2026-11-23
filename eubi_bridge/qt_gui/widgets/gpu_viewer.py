"""eubi's GPU viewer: micro-viewer (Workplan WP14) in the Inspect page's Viewer tab.

``GpuZarrViewer`` has the members the Inspect page uses of the classic ``ZarrViewer``
(signals ``frame_ready``, ``dataset_loaded``, ``status_changed``; ``channel_panel``;
``load_dataset``, ``path``, ``axes``, ``shape``, ``channels``, ``level_idx``,
``level_paths``, ``orientation``, ``z``, ``fov_center``, ``image_widget``), and
``status_text()`` for the page's status line. micro-viewer draws on the GPU from a tile
cache, so frames never pass through Python (``frame_ready`` carries no image).

What it shows on top of the classic viewer:

- planes at any angle through a volume (Alt + drag turns the plane; XY, XZ, YZ look
  along the axes), and the volume seen whole in 3-D: maximum intensity, the volume
  composited, or surfaces, with shading and a density;
- linked views: the plane, its YZ and XZ companions and the 3-D view in quarters;
- each channel's histogram (micro-viewer's panel under eubi's channel panel), with the
  contrast limits, the surface level and the opacity curve dragged on it; contrast
  changed there flows back into eubi's panel, so "Save Channels" writes what is shown;
- moving and turning the image by hand (Ctrl + drag, Ctrl + wheel) and saving that
  transform into the image (metadata only);
- image positions: a plate opens whole with every field of each well at its position
  (one field on request under Planes), a well or a bioformats2raw collection (tiles,
  views, positions) with each image at its position, an image of a collection or a
  field at its own position (NGFF 0.4 / 0.5: the image's translation; 0.6: its owner's
  scene); the store's contents (wells, fields, series, label images) as a tree; label
  images drawn over their image.

Several images as layers of one view are left out until micro-viewer's layers are
stable. Errors never escape a slot (PyQt6 ends the process when one does): they go to
``status_changed``.
"""
from __future__ import annotations

import functools
import math
import os

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QLabel,
                             QMessageBox, QSizePolicy, QSlider, QSpinBox, QSplitter,
                             QToolBar, QVBoxLayout, QWidget)

import micro_viewer
from micro_viewer import ERRORS
from micro_viewer.contents import ContentsPanel, open_node
from micro_viewer.panel import ChannelPanel as HistogramPanel
from micro_viewer.qt import ViewerWidget

from eubi_bridge.qt_gui.widgets.channel_panel import ChannelPanel

#: what the projection shows: the label in the menu, micro-viewer's name
_MODES = [("maximum intensity", "mip"), ("volume", "volume"), ("surfaces", "iso")]
_ORIENTATIONS = {"XY": ([1, 0, 0], [0, 1, 0]), "XZ": ([1, 0, 0], [0, 0, 1]),
                 "YZ": ([0, 0, 1], [0, 1, 0])}

KEYS = """Mouse: drag pans, the wheel zooms at the pointer, F fits the image.
Page Up / Page Down: the next plane (with Shift: the next time point).

Volumes:
  Alt + drag turns the plane through the volume; 1, 2, 3 look along z, y, x.
  I switches smooth (trilinear) sampling.
  M shows the 3-D view instead of the plane; P shows it and cycles
  maximum intensity, volume and surfaces; Shift + P switches shading;
  - and + make the volume thinner or denser.
  L switches linked views (the plane, YZ, XZ and the 3-D view).

Moving the image (to save its placement):
  Ctrl + drag moves it, Ctrl + wheel turns it (Shift: finer),
  Ctrl + arrow keys nudge it, Ctrl + Alt + drag tilts a volume,
  Ctrl + 0 puts it back, Ctrl + S saves the transform into the image.

Turning the view: Shift + wheel; [ and ] by 90 degrees; 0 back.

Label images (from the Contents tree): double-click selects the id under
the pointer, Esc clears it, O switches outlines."""


def _safe(slot):
    """A slot that reports its error instead of raising (see the module's notes)."""
    @functools.wraps(slot)
    def run(self, *args, **kwargs):
        try:
            return slot(self, *args, **kwargs)
        except ERRORS as exc:  # noqa: BLE001 (a Rust panic too)
            self.status_changed.emit(f"Viewer: {exc}")
    return run


def _hex(rgb) -> str:
    return "#" + "".join(f"{round(max(0.0, min(1.0, c)) * 255):02x}" for c in rgb)


def _rgb(text: str):
    t = str(text).lstrip("#")
    if len(t) < 6:
        return (1.0, 1.0, 1.0)
    return tuple(int(t[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


class GpuZarrViewer(QWidget):
    """The Inspect page's viewer on micro-viewer (see the module's notes)."""

    frame_ready = pyqtSignal(object, float, int)
    dataset_loaded = pyqtSignal(object)
    status_changed = pyqtSignal(str)

    def __init__(self, parent=None, **_classic_options):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._path: str | None = None
        self._shown: str | None = None          # what the view opened (a series, a field)
        self._pyr = None
        self._image: dict | None = None
        self._level_paths: list[str] = []
        self._kind = ""
        self._pick = ""
        self._note = ""
        self._rows: dict = {}
        self.view = ViewerWidget()
        self.view.setMinimumSize(200, 200)
        self.channel_panel = ChannelPanel()
        self.channel_panel.channels_changed.connect(self._apply_channels)
        self.channel_panel.auto_requested.connect(self._auto)
        self.histogram_panel = HistogramPanel(self.view)
        self.contents = ContentsPanel()
        self.contents.activated.connect(self._open_node)
        # left: planes, contents, channels, histogram
        self._planes = QGroupBox("Planes")
        self._planes_layout = QVBoxLayout(self._planes)
        self._planes_layout.setContentsMargins(6, 18, 6, 4)
        self._contents_box = QGroupBox("Contents")
        cl = QVBoxLayout(self._contents_box)
        cl.setContentsMargins(4, 18, 4, 4)
        cl.addWidget(self.contents)
        self._contents_box.hide()
        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self._planes)
        tl.addWidget(self._contents_box, 1)
        side = QSplitter(Qt.Orientation.Vertical)
        side.addWidget(top)
        side.addWidget(self.channel_panel)
        side.addWidget(self.histogram_panel)
        side.setStretchFactor(0, 1)
        side.setStretchFactor(1, 2)
        side.setStretchFactor(2, 2)
        # right: the toolbar over the view
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        rl.addWidget(self._toolbar())
        rl.addWidget(self.view, 1)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(side)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 4)
        split.setSizes([320, 1100])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(split)
        # what follows the view
        self.view.viewChanged.connect(self._view_changed)
        self.view.picked.connect(self._picked)
        self.view.saved.connect(self._saved)
        self.view.failed.connect(lambda text: self.status_changed.emit(f"Viewer: {text}"))
        self._later = QTimer(self)               # frame_ready, a little behind the view
        self._later.setSingleShot(True)
        self._later.setInterval(80)
        self._later.timeout.connect(self._frame)
        self._tick = QTimer(self)                # the level sharpens without events
        self._tick.timeout.connect(self._frame)
        self._tick.start(500)

    # ------------------------------------------------------------------ the toolbar
    def _toolbar(self) -> QToolBar:
        bar = QToolBar()
        bar.setStyleSheet("QToolBar { border: 0; }")

        def action(text, slot, tip, checkable=False):
            act = QAction(text, self)
            act.setToolTip(tip)
            act.setCheckable(checkable)
            act.triggered.connect(slot)
            bar.addAction(act)
            return act

        group = QActionGroup(self)
        self.orient = {}
        for name, tip in (("XY", "look along z (1)"), ("XZ", "look along y (2)"),
                          ("YZ", "look along x (3)")):
            act = action(name, lambda _=False, n=name: self._orient(n), tip, True)
            group.addAction(act)
            self.orient[name] = act
        self.orient["XY"].setChecked(True)
        action("Fit", lambda: self._apply(lambda v: v.fit()), "show the whole image (F)")
        bar.addSeparator()
        self.linked = action("Linked", self._linked,
                             "the plane, YZ, XZ and the 3-D view in quarters (L)", True)
        self.three_d = action("3D", self._three_d,
                              "the volume seen whole instead of the plane (M)", True)
        self.mode = QComboBox()
        self.mode.addItems([label for label, _ in _MODES])
        self.mode.setToolTip("what the 3-D view shows (P)")
        self.mode.currentIndexChanged.connect(self._mode)
        bar.addWidget(self.mode)
        bar.addWidget(QLabel(" density "))
        self.density = QSlider(Qt.Orientation.Horizontal)
        self.density.setRange(0, 100)            # log scale, 0.001 .. 1
        self.density.setFixedWidth(90)
        self.density.setToolTip("the volume's opacity per voxel at full brightness (- / +)")
        self.density.valueChanged.connect(self._density)
        bar.addWidget(self.density)
        self.shading = QCheckBox("shading")
        self.shading.setToolTip("light the volume and surfaces from their gradients "
                                "(Shift + P)")
        self.shading.toggled.connect(lambda on: self._apply(lambda v: setattr(v, "shading", on)))
        bar.addWidget(self.shading)
        self.smooth = QCheckBox("smooth")
        self.smooth.setToolTip("trilinear sampling between voxels (I)")
        self.smooth.toggled.connect(self._smooth)
        bar.addWidget(self.smooth)
        bar.addSeparator()
        action("Save transform", self._save_transform,
               "save the image's placement (moved with Ctrl + drag, turned with Ctrl + "
               "wheel) into the image: metadata only (Ctrl + S)")
        action("Put back", lambda: self._apply(lambda v: v.reset_transform(layer=0)),
               "put the image back where its metadata places it (Ctrl + 0)")
        bar.addSeparator()
        action("Keys", lambda: QMessageBox.information(self, "Viewer keys", KEYS),
               "the mouse and keys of the viewer")
        return bar

    # ------------------------------------------------------------------ what the page reads
    @property
    def path(self):
        return self._path

    @property
    def pyr(self):
        return self._pyr

    @property
    def is_plate(self) -> bool:
        return self._kind == "plate"

    @property
    def axes(self) -> str:
        """The axes as one lowercase string ("czyx"), as the classic viewer gives them."""
        return "".join(self._image["axes"]).lower() if self._image else ""

    @property
    def shape(self) -> list:
        return list(self._image["levels"][0]["shape"]) if self._image else []

    @property
    def channels(self) -> list:
        """The channel dicts as eubi's panel holds them (what "Save Channels" writes)."""
        return self.channel_panel.channel_data()

    @property
    def level_idx(self) -> int:
        try:
            return self.view.viewer.level if self._image else 0
        except ERRORS:  # noqa: BLE001
            return 0

    @property
    def level_paths(self) -> list[str]:
        return self._level_paths

    def _orientation_name(self) -> str:
        try:
            o = [[round(c, 6) for c in row] for row in self.view.viewer.orientation]
        except ERRORS:  # noqa: BLE001
            return "XY"
        for name, (r, d) in _ORIENTATIONS.items():
            if o[0] == r and o[1] == d:
                return name
        return "turned"

    @property
    def orientation(self) -> str:
        """ "XY", "XZ" or "YZ" (a turned plane counts as XY here; see status_text)."""
        name = self._orientation_name()
        return name if name in _ORIENTATIONS else "XY"

    def _plane_index(self, axis: str) -> int:
        try:
            for name, i, _ in (self.view.viewer.position if self._image else []):
                if name == axis:
                    return round(i)
        except ERRORS:  # noqa: BLE001
            pass
        return 0

    @property
    def z(self) -> int:
        return self._plane_index("z")

    @property
    def t(self) -> int:
        return self._plane_index("t")

    @property
    def fov_center(self):
        """(y, x) in level-0 pixels at the view's centre."""
        if not self._image:
            return (0.0, 0.0)
        try:
            v = self.view.viewer
            w, h = v.size
            p = v.pick(w / 2, h / 2)
        except ERRORS:  # noqa: BLE001
            return (0.0, 0.0)
        first = p["layers"][0] if p["layers"] else {}
        if first.get("pixel") is not None:
            return (float(first["pixel"][1]), float(first["pixel"][0]))
        return (p["world"][1], p["world"][0])

    @property
    def image_widget(self):
        return self

    def set_frame(self, *_args):
        """The classic image widget takes frames from Python; micro-viewer draws its own."""

    def status_text(self) -> str:
        """The page's status line: level, plane, what the view shows, what is under the
        pointer."""
        if not self._image:
            return self._note
        try:
            v = self.view.viewer
            level = v.level
            n = len(self._image["levels"])
            parts = [f"Level {level} of {n}"]
            name = self._orientation_name()
            if v.mip:
                parts.append(f"3D: {dict((b, a) for a, b in _MODES).get(v.projection)}")
            else:
                parts.append(f"{name} plane" + (f"  z={self.z}" if name == "XY" else ""))
            if self.t:
                parts.append(f"t={self.t}")
            if v.linked:
                parts.append("linked views")
            if self.is_plate:
                info = v.plate(layer=0)
                parts.append("plate: every field" if info["field"] is None
                             else f"plate: field {info['field']}")
            elif v.layers[0].get("group"):
                members = v.members(layer=0)
                parts.append(f"{len(members['names'])} images at their positions"
                             + (" (no positions: on a grid)" if members["gridded"] else ""))
            m = v.transform_matrix(0)
            k = len(m)
            moved = any(abs(m[r][c] - (1.0 if r == c else 0.0)) > 1e-9
                        for r in range(k) for c in range(k + 1))
            if moved:
                parts.append("moved (Save transform keeps it)")
        except ERRORS as exc:  # noqa: BLE001
            return f"Viewer: {exc}"
        text = "   ".join(parts)
        if self._pick:
            text += "   |   " + self._pick
        if self._note:
            text += "   |   " + self._note
        return text

    # ------------------------------------------------------------------ loading
    def load_dataset(self, zarr_path: str, _keep_tree: bool = False):
        """Open the store at `zarr_path`: an image, a plate (whole), a well or a
        bioformats2raw collection (whole: each image at its position; with a
        micro-viewer older than that, a collection's first series), and list its contents
        (`_keep_tree`: an image of the store shown, whose tree stays)."""
        from zarr_plane_server import pyramid_cache  # type: ignore (on the page's path)
        try:
            node = micro_viewer.tree(zarr_path)
        except ERRORS:  # noqa: BLE001 (micro-viewer then says why when it opens)
            node = {"kind": "image", "children": []}
        kind = node.get("kind", "image")
        shown = zarr_path
        whole = hasattr(micro_viewer.Viewer, "members")   # micro-viewer with positions
        if kind == "collection" and node.get("children") and not whole:
            shown = node["children"][0]["path"]
        meta_path = shown
        if kind == "collection" and node.get("children"):
            meta_path = node["children"][0]["path"]
        if kind == "plate":
            wells = [w for w in node.get("children", []) if w.get("children")]
            if wells:
                meta_path = wells[0]["children"][0]["path"]
        try:
            v = self.view.viewer                 # the core first: its errors come back here
            v.open(shown)
            v.fit()
            image = v.image
            if image is None:
                raise ValueError(f"nothing to show in {shown}")
            members = v.members(layer=0) if whole else None
            if members and members["paths"] and meta_path == shown:
                # a well opened whole: the metadata tab shows its first field
                meta_path = os.path.join(shown, members["paths"][0])
        except ERRORS as exc:  # noqa: BLE001 (the page shows it)
            self.view._changed()
            self.status_changed.emit(f"Error loading: {exc}")
            return
        self.view._changed()
        try:
            pyr = pyramid_cache.get(meta_path)
        except Exception as exc:  # noqa: BLE001
            self.status_changed.emit(f"Viewer: the metadata tab cannot read {meta_path}: {exc}")
            pyr = None
        self._path, self._shown, self._pyr, self._kind = zarr_path, shown, pyr, kind
        self._image = image
        paths = [lv["path"] for lv in image["levels"]]
        if pyr is not None and (kind == "plate" or v.layers[0].get("group")):
            # the levels the fields have (a plate or a group adds coarser ones)
            paths = [p for p in paths if p in getattr(pyr, "layers", {})]
        self._level_paths = paths
        self._note = ""
        self._fill_channels()
        self._build_planes()
        # the 3-D controls only for a volume
        volume = bool(v.layers[0]["volume"])
        for w in (self.three_d, self.linked, self.mode, self.density, self.shading, self.smooth,
                  self.orient["XZ"], self.orient["YZ"]):
            w.setEnabled(volume)
        if not volume and (v.mip or v.linked):
            v.mip = False
            v.linked = False
        if not _keep_tree:
            has_children = bool(node.get("children")) or kind in ("plate", "collection")
            if has_children:
                self.contents.show_tree(zarr_path)
            self._contents_box.setVisible(has_children)
        self.histogram_panel.refresh()
        self._view_changed()
        self.status_changed.emit(zarr_path)
        if pyr is not None:
            self.dataset_loaded.emit(pyr)
        self._frame()

    def _fill_channels(self):
        v = self.view.viewer
        chans = v.layers[0]["channels"]
        self.channel_panel.set_channels([
            {"index": i, "label": c["label"], "color": _hex(c["color"]),
             "visible": c["visible"], "intensityMin": c["lo"], "intensityMax": c["hi"]}
            for i, c in enumerate(chans)])

    def _build_planes(self):
        while self._planes_layout.count():
            item = self._planes_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._rows = {}
        v = self.view.viewer
        rows = [(n, s) for n, _, s in v.position]
        plate = v.plate(layer=0) if v.layers and v.layers[0]["plate"] else None
        if plate and plate["fields"] > 1:
            rows.append(("field", plate["fields"]))
        for name, size in rows:
            row = QWidget()
            lay = QHBoxLayout(row)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(QLabel(f"{name.upper() if name != 'field' else 'Field'}:"))
            if name == "field":
                spin = QSpinBox()
                # -1: every field of each well at its position
                spin.setRange(-1, size - 1)
                spin.setSpecialValueText("all")
                spin.setValue(-1 if plate["field"] is None else plate["field"])
                spin.setToolTip("the field shown in every well (all: every field at its "
                                "position)")
                spin.valueChanged.connect(self._field)
                lay.addWidget(spin)
                lay.addStretch(1)
                self._rows[name] = (spin, None)
            else:
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(0, max(0, size - 1))
                slider.valueChanged.connect(lambda val, n=name: self._move(n, val))
                label = QLabel()
                label.setMinimumWidth(60)
                lay.addWidget(slider, 1)
                lay.addWidget(label)
                self._rows[name] = (slider, label)
            self._planes_layout.addWidget(row)
        self._planes.setVisible(bool(self._rows))
        self._sync_planes()

    def _sync_planes(self):
        try:
            v = self.view.viewer
            if not v.layers or v.current != 0:
                return
            pos = v.position
        except ERRORS:  # noqa: BLE001
            return
        for name, index, size in pos:
            if name not in self._rows:
                continue
            slider, label = self._rows[name]
            slider.blockSignals(True)
            slider.setValue(round(index))
            slider.blockSignals(False)
            label.setText(f"{round(index)} / {size - 1}")

    # ------------------------------------------------------------------ slots
    def _apply(self, step):
        try:
            step(self.view.viewer)
        except ERRORS as exc:  # noqa: BLE001
            self.status_changed.emit(f"Viewer: {exc}")
            return
        self.view._changed()

    def _orient(self, name):
        self._apply(lambda v: v.set_orientation(name.lower()))

    def _linked(self, on):
        self._apply(lambda v: setattr(v, "linked", on))

    def _three_d(self, on):
        self._apply(lambda v: setattr(v, "mip", on))

    def _mode(self, index):
        def step(v):
            v.projection = _MODES[index][1]
            v.mip = True
        self._apply(step)

    def _density(self, value):
        self._apply(lambda v: setattr(v, "density", 10 ** (value / 100 * 3 - 3)))

    def _smooth(self, on):
        def step(v):
            if v.layers:
                v.set_interpolation("linear" if on else "nearest", layer=0)
        self._apply(step)

    @_safe
    def _move(self, name: str, value: int):
        v = self.view.viewer
        v.set_position(name, float(value), layer=0)
        self.view._changed()

    @_safe
    def _field(self, value: int):
        self.view.viewer.set_field(None if int(value) < 0 else int(value), layer=0)
        self.view._changed()

    @_safe
    def _apply_channels(self, channels: list):
        v = self.view.viewer
        if not v.layers:
            return
        current = v.layers[0]["channels"]
        for ch in channels:
            i = ch.get("index", 0)
            if i >= len(current):
                continue
            cur = current[i]
            lo, hi = ch.get("intensityMin"), ch.get("intensityMax")
            v.set_channel(i, color=_rgb(ch.get("color", "#ffffff")),
                          lo=cur["lo"] if lo is None else float(lo),
                          hi=cur["hi"] if hi is None else float(hi),
                          visible=bool(ch.get("visible", True)),
                          label=str(ch.get("label", cur["label"])), layer=0)
        self.view._changed()

    @_safe
    def _auto(self, index: int):
        v = self.view.viewer
        v.auto_contrast(index, layer=0)
        c = v.layers[0]["channels"][index]
        self.channel_panel.set_channel_range(index, c["lo"], c["hi"])
        self.view._changed()

    @_safe
    def _view_changed(self):
        """The toolbar and planes follow the view (keys change it too); contrast changed
        on the histogram flows into eubi's panel."""
        v = self.view.viewer
        if not v.layers:
            return
        for widget, value in ((self.linked, v.linked), (self.three_d, v.mip)):
            widget.blockSignals(True)
            widget.setChecked(value)
            widget.blockSignals(False)
        modes = [name for _, name in _MODES]
        for w, setter in (
                (self.mode, lambda: self.mode.setCurrentIndex(modes.index(v.projection))),
                (self.density, lambda: self.density.setValue(
                    round((math.log10(v.density) + 3) / 3 * 100))),
                (self.shading, lambda: self.shading.setChecked(v.shading)),
                (self.smooth, lambda: self.smooth.setChecked(
                    v.layers[0]["interpolation"] == "linear"))):
            w.blockSignals(True)
            setter()
            w.blockSignals(False)
        name = self._orientation_name()
        for key, act in self.orient.items():
            act.blockSignals(True)
            act.setChecked(key == name)
            act.blockSignals(False)
        self._sync_planes()
        self._sync_eubi_panel()
        self._later.start()

    def _sync_eubi_panel(self):
        """Contrast changed on the histogram: into eubi's panel (only what differs:
        eubi's set_range always signals, and that would come back here)."""
        rows = self.channel_panel.channel_data()
        for i, c in enumerate(self.view.viewer.layers[0]["channels"]):
            if i >= len(rows):
                break
            lo, hi = rows[i].get("intensityMin"), rows[i].get("intensityMax")
            if lo is None or hi is None or abs(lo - c["lo"]) > 1e-9 or abs(hi - c["hi"]) > 1e-9:
                self.channel_panel.set_channel_range(i, c["lo"], c["hi"])

    @_safe
    def _picked(self, p):
        if p.get("pixel") is None:
            self._pick = ""
        else:
            names = [c["label"] for c in self.view.viewer.channels]
            vals = "  ".join(
                f"label id {e['id']}" if e.get("id") is not None
                else f"{names[e['channel']]}: {e['value']:g}" if e["value"] is not None
                else f"{names[e['channel']]}: -" for e in p["values"])
            where = (f"x={p['voxel'][0]} y={p['voxel'][1]} z={p['voxel'][2]}"
                     if p.get("voxel") else f"x={p['pixel'][0]} y={p['pixel'][1]}")
            if p.get("well"):
                where = f"well {p['well'][0]} ({p['well'][1]})  " + where
            elif p.get("member"):
                where = f"{p['member'][0]}  " + where
            self._pick = f"{where}  {vals}"
        self._later.start()

    @_safe
    def _open_node(self, node):
        v = self.view.viewer
        if node["kind"] in ("image", "collection", "plate") or (
                node["kind"] == "well" and not self.is_plate):
            # another image of the store: the page's metadata follows it (a well
            # opens whole, each field at its position)
            whole = hasattr(micro_viewer.Viewer, "members")
            path = (node.get("first_field") if node["kind"] == "well" and not whole
                    else node["path"])
            self._load_inner(path or node["path"])
            return
        self._note = open_node(v, node)
        self.view._changed()

    def _load_inner(self, path):
        """Open an image of the open store (a field, a series, the plate again); the
        store's tree stays."""
        self.load_dataset(path, _keep_tree=True)

    @_safe
    def _save_transform(self):
        if self.is_plate:
            raise ValueError("a plate is its wells on a grid, which no image stores: open "
                             "a field (in Contents) to save its transform")
        if self.view.viewer.layers[0].get("group"):
            raise ValueError("a well or a collection is its images at their positions, "
                             "which no one image stores: open one of them (in Contents) "
                             "to save its transform")
        keep = self.view.viewer.current
        self.view.viewer.current = 0
        try:
            self.view.save_transform()
        finally:
            self.view.viewer.current = keep

    @_safe
    def _saved(self, text):
        """A transform was saved: the metadata tab reads the image again."""
        self._note = text
        self.status_changed.emit(text)
        from zarr_plane_server import pyramid_cache  # type: ignore
        with pyramid_cache._lock:
            pyramid_cache._cache.pop(self._shown, None)
        try:
            self._pyr = pyramid_cache.get(self._shown)
        except Exception:  # noqa: BLE001
            return
        self.dataset_loaded.emit(self._pyr)

    @_safe
    def _frame(self):
        if self._image:
            self.frame_ready.emit(None, 0.0, 0)

    def closeEvent(self, event):
        self._tick.stop()
        self._later.stop()
        super().closeEvent(event)


def describe() -> str:
    """micro-viewer's version and where it was imported from (for bug reports)."""
    return f"micro-viewer {micro_viewer.__version__} ({os.path.dirname(micro_viewer.__file__)})"
