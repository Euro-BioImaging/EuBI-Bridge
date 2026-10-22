"""micro-reader inside eubi-bridge's older readers and the aggregative path.

``MicroSource`` / ``micro_array``: micro-reader's pixels of one (series, tile, view,
illumination) image as an array-like that ``DynamicArray`` wraps (as it wraps
``_BioFormatsSource``), from a pool of open files (``_using``). ``MicroReader``:
micro-reader as a reader of its own behind ``PFFImageMeta`` (``metadata_reader='micro'``),
with its metadata as OME (``micro_omemeta``) and the extras NGFF has no field for
(``micro_extras``).

eubi-bridge 0.1.x could also check micro-reader's pixels against the standard reader's
(``micro_check_pixels``: ``swap_in`` / ``verify``); that check is retired (kept for
reference in ``legacy_0_1/``).
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Optional

import numpy as np

from eubi_bridge.conversion.pyramid_path import SERIES
from eubi_bridge.utils.logging_config import get_logger

logger = get_logger(__name__)

PIXEL_READERS = ("standard", "micro")
#: a dimension a file does not name (plain multi-page TIFF pages) is z, as
#: Bio-Formats reads it
RENAME = {"sequence": "z"}
#: image indices eubi-bridge selects by (its series / tile / view /
#: illumination / rotation / phase options); a file with any other index
#: (block, emission, excitation, lifetime) is left to the standard reader
SELECTABLE = {SERIES, "tile", "rotation", "view", "illumination", "phase", "resolution"}
STANDARD = "tczyx"
#: files kept open per process, least recently used closed first: an
#: aggregative input of thousands of files must not hold them all open (each
#: holds a handle per reading thread; "Too many open files" at ~500 of 2347).
#: 128 keeps a slab of planes open while a memory-bounded write visits each y/x
#: tile of it (dyna write_levels: slabs of e.g. 96 planes); at 32, every file was
#: opened again for every tile (2347 files x 4 tiles: 6.7 min instead of ~1.3).
MAX_OPEN_FILES = 128
#: the micro-reader releases this eubi-bridge reads (pyproject: micro-reader>=0.2.0,<0.3):
#: 0.2.x declares the coordinate systems each output is written with (9.18); 0.1.x
#: declares none, and older releases call the series index "scene"
MICRO_READER_RANGE = ((0, 2), (0, 3))
MICRO_READER_FIX = 'pip install -U "micro-reader>=0.2,<0.3"'


class MicroReaderVersionError(ImportError):
    """The installed micro-reader is outside the releases this eubi-bridge reads
    (MICRO_READER_RANGE). Without this, its files would go to the older readers
    without a word (a forced or ``--no-deps`` install that ``pip check`` reports)."""


_micro_reader_checked = False


def require_micro_reader() -> None:
    """Raise MicroReaderVersionError, naming the fix, when the installed micro-reader
    is outside MICRO_READER_RANGE; checked once per process. A micro-reader without a
    version (a source tree on the path) is taken as it is."""
    global _micro_reader_checked
    if _micro_reader_checked:
        return
    import micro_reader
    version = getattr(micro_reader, "__version__", None)
    if version is None:
        try:
            from importlib.metadata import version as dist_version
            version = dist_version("micro-reader")
        except Exception:                           # noqa: BLE001 - no version known
            version = None
    if version is not None:
        try:
            got = tuple(int(part) for part in str(version).split(".")[:2])
        except ValueError:
            got = None
        low, high = MICRO_READER_RANGE
        if got is not None and not (low <= got < high):
            if got < (0, 1):
                why = (" (it names the series index 'scene'; micro-reader 0.1.0 renamed "
                       f"it '{SERIES}', the name this eubi-bridge reads)")
            elif got < low:
                why = (" (it declares no coordinate systems; micro-reader 0.2.0 does, "
                       "and this eubi-bridge writes each output's system)")
            else:
                why = ""
            raise MicroReaderVersionError(
                f"eubi-bridge needs micro-reader >={low[0]}.{low[1]}.0,<{high[0]}.{high[1]}, "
                f"but micro-reader {version} is installed{why}. Fix: {MICRO_READER_FIX}")
    _micro_reader_checked = True


class _Open:
    """An open micro-reader file, its axes images, and how many reads use it."""

    def __init__(self, file):
        self.file, self.views, self.users = file, {}, 0


#: (path, tiles, multifile) -> _Open, oldest first
_files: "OrderedDict[tuple, _Open]" = OrderedDict()
_files_lock = threading.Lock()


def _evict() -> None:
    """Close the least recently used files beyond MAX_OPEN_FILES that no read
    is using (a file in use is never closed under a reader)."""
    while len(_files) > MAX_OPEN_FILES:
        idle = next((k for k, e in _files.items() if e.users == 0), None)
        if idle is None:
            return
        _files.pop(idle).file.close()


@contextmanager
def _using(path: str, tiles: str, multifile: bool = True):
    """The open file for (*path*, *tiles*, *multifile*), opened (again) if
    needed, held open while the block runs.  *multifile*: micro-reader's
    option -- False opens a file of a multi-file OME set alone (the aggregative
    path joins the files itself), True the whole set."""
    import micro_reader
    require_micro_reader()
    key = (path, tiles, bool(multifile))
    with _files_lock:
        entry = _files.get(key)
        if entry is not None:
            _files.move_to_end(key)
            entry.users += 1
            _evict()
    if entry is None:
        # opened outside the lock, so several files open at once (an aggregative
        # input's files are opened on a thread pool); a file two threads opened at
        # once is kept once
        opened = micro_reader.open(path, tiles=tiles, rename=RENAME, multifile=bool(multifile))
        with _files_lock:
            entry = _files.get(key)
            if entry is None:
                entry = _files[key] = _Open(opened)
                opened = None
            else:
                _files.move_to_end(key)
            entry.users += 1
            _evict()
        if opened is not None:
            opened.close()
    try:
        yield entry
    finally:
        with _files_lock:
            entry.users -= 1
            _evict()


def set_read_concurrency(n: int) -> None:
    """micro-reader's decode threads in this process (``decode_threads``):
    one pool shared by all region reads, so this caps the CPU spent reading
    without changing how many regions the writer keeps in flight. Its memory
    is set per file, once the file is open (:func:`set_read_memory`)."""
    import micro_reader
    if micro_reader.decode_threads() != int(n):
        micro_reader.set_decode_threads(int(n))


_read_memory_lock = threading.Lock()
_read_memory = (0, 0)


def set_read_memory(decode_mb: float, cache_mb: float) -> None:
    """micro-reader's decoding limit and its caches' limit in this process
    (``core/memory.reader_share``). Process-wide: scenes and files read at once
    in this process get the largest any of them needs."""
    global _read_memory
    import micro_reader
    with _read_memory_lock:
        decode = max(_read_memory[0], int(decode_mb * 2**20))
        cache = max(_read_memory[1], int(cache_mb * 2**20))
        if (decode, cache) != _read_memory:
            micro_reader.set_memory_limit(decode)
            if hasattr(micro_reader, "set_cache_limit"):     # micro-reader with shared caches
                micro_reader.set_cache_limit(cache)
            _read_memory = (decode, cache)


def close_all() -> None:
    """Close every micro-reader file this process opened."""
    with _files_lock:
        entries = list(_files.values())
        _files.clear()
    for entry in entries:
        entry.file.close()


class MicroSource:
    """One micro-reader image as a T C Z Y X array (RGB samples as channels),
    read lazily: any rectangle, exactly.  It holds only
    where the image is (picklable, deep-copyable); each read borrows the open
    file from the process's bounded cache, reopening it if it was closed."""

    def __init__(self, path: str, tiles: str, index: int, shape, dtype, level: int = 0,
                 read_unit=None, multifile: bool = True):
        self.path, self.tiles, self.index = path, tiles, int(index)
        #: micro-reader's ``multifile`` the file was opened with (``_using``)
        self.multifile = bool(multifile)
        #: resolution level of the image (0: full resolution)
        self.level = int(level)
        self.shape = tuple(int(n) for n in shape)
        self.dtype = np.dtype(dtype)
        self.ndim = len(self.shape)
        # any rectangle can be read: a whole plane is advertised as one chunk
        self.chunks = (1, 1, 1, self.shape[3], self.shape[4])
        #: what reading part of it really decodes, as micro-reader reports it for this
        #: T C Z Y X view: None when any rectangle is read exactly (uncompressed),
        #: else the smallest block the backend decodes at once (a compressed strip or
        #: tile, an IMS chunk). dyna's writer plans its y/x tiles from it.
        self.read_unit = None if read_unit is None else tuple(int(n) for n in read_unit)

    def __getitem__(self, key):
        with _using(self.path, self.tiles, getattr(self, "multifile", True)) as entry:
            view_key = (self.index, self.level)
            view = entry.views.get(view_key)
            if view is None:
                image = entry.file.images[self.index]
                if self.level:
                    image = image.resolutions[self.level]
                view = entry.views[view_key] = image.as_axes(STANDARD, samples="channels")
            return view[key]

    def __repr__(self):
        return (f"<MicroSource {self.path!r} image {self.index} level {self.level} "
                f"{self.shape} {self.dtype}>")


def micro_array(path: str, scene: int, tile: Optional[int], *, as_mosaic: bool = False,
                view: int = 0, illumination: int = 0, rotation: int = 0, phase: int = 0,
                multifile: bool = True):
    """The micro-reader image eubi-bridge means by (series index, tile, view,
    illumination, rotation, phase), as a ``DynamicArray`` -- or (None,
    reason) when micro-reader cannot read the file or has no such image.

    eubi-bridge's rules: *scene* is the rank among the file's series;
    *as_mosaic* stitches tiles, otherwise *tile* is the mosaic tile index
    (CZI ``M``; None: tile 0), and on a file without tiles tile 0 is the
    whole image (``set_tile`` is a no-op there).  *multifile*: False reads a
    file of a multi-file OME set alone (the aggregative path), True the set.
    """
    import micro_reader
    from dyna_zarr import from_array
    tiles = "mosaic" if as_mosaic else "separate"
    require_micro_reader()          # a wrong micro-reader is not a fallback
    try:
        with _using(path, tiles, multifile) as entry:
            return _find(entry.file, path, tiles, scene, tile, view=view,
                         illumination=illumination, rotation=rotation, phase=phase,
                         multifile=multifile)
    except micro_reader.UnsupportedFormatError as exc:
        if tiles == "mosaic":                        # tiled, but not composable
            return None, f"micro-reader: {exc}"
        return None, f"micro-reader cannot read it: {exc}"
    except Exception as exc:                        # noqa: BLE001 - fall back, say why
        return None, f"micro-reader cannot read it: {type(exc).__name__}: {exc}"


def _find(f, path: str, tiles: str, scene: int, tile: Optional[int], *, view: int,
          illumination: int, rotation: int, phase: int, multifile: bool = True):
    """``micro_array``'s lookup in the open file *f*."""
    from dyna_zarr import from_array
    scenes = sorted({k[SERIES] for k in f.indices})
    if not 0 <= scene < len(scenes):
        return None, f"series index {scene}: micro-reader has {len(scenes)} series"
    want = {SERIES: scenes[scene]}
    levels = f.index_range
    if "tile" in levels:
        want["tile"] = int(tile or 0)
    elif tile:
        return None, f"tile {tile}: micro-reader sees no tiles in this file"
    for level, value in (("rotation", rotation), ("view", view),
                         ("illumination", illumination), ("phase", phase)):
        if level in levels:
            values = sorted({k[level] for k in f.indices if level in k})
            if not 0 <= int(value) < len(values):
                return None, f"{level} {value}: micro-reader sees {len(values)}"
            want[level] = values[int(value)]            # eubi-bridge counts from 0
        elif value:
            return None, f"{level} {value}: micro-reader sees one {level}"
    found = f.select(**want)
    if len(found) != 1:
        return None, f"micro-reader has {len(found)} images indexed {want}"
    image = found[0]
    view_ = image.as_axes(STANDARD, samples="channels")
    source = MicroSource(path, tiles, f.images.index(image), view_.shape, view_.dtype,
                         read_unit=view_.read_unit, multifile=multifile)
    return from_array(source), ""


# -- micro-reader as a reader of its own (no standard reader opened) ---------------

def _resolve_index(value, total: int):
    """(first index, how many exposed), as eubi-bridge's CZI reader resolves
    view_index / illumination_index: 'all', '0,2' or a single index."""
    if value == "all":
        return 0, total
    if isinstance(value, (list, tuple)):
        return int(value[0]), len(value)
    if isinstance(value, str) and "," in value:
        parts = [int(x) for x in value.split(",")]
        return parts[0], len(parts)
    return int(value or 0), 1


class MicroReader:
    """eubi-bridge's reader interface (``reader_interface.ImageReader``) over
    a micro-reader file: series, tiles, views and illuminations from its
    image indices, pixels as ``MicroSource`` arrays.  No standard reader is
    opened.  Semantics follow eubi-bridge's CZI reader (scene = the series' rank,
    tile = position within the series, view / illumination exposure, series_path).

    ``MicroReader.open`` returns None when micro-reader cannot read the file,
    or when the file has tiles / views / rotations outside a CZI (eubi-bridge's
    standard readers treat those differently): the standard reader is used.
    """

    def __init__(self, path: str, tiles: str, keys: list, *, as_mosaic: bool,
                 view_index=0, illumination_index=0, phase_index=0, rotation_index=0):
        self._path, self._tiles, self._keys = path, tiles, keys
        self.as_mosaic = as_mosaic
        self._scene_values = sorted({k[SERIES] for k in keys})
        self._phase, self._rotation = int(phase_index or 0), int(rotation_index or 0)
        self._view_index, self._illumination_index = view_index, illumination_index
        #: a CZI may have views / illuminations (SceneLoader's probe)
        self.may_have_views = path.lower().endswith(".czi")
        # PFFImageMeta reads img.dims for non-RGB data (no S: T C Z Y X as is)
        self.img = SimpleNamespace(dims=SimpleNamespace())
        self.series, self.tile, self.view, self.illumination = 0, 0, 0, 0
        self.set_scene(0)

    @classmethod
    def open(cls, path: str, *, as_mosaic: bool = False, view_index=0, illumination_index=0,
             phase_index=0, rotation_index=0, **_ignored):
        import micro_reader
        tiles = "mosaic" if as_mosaic else "separate"
        require_micro_reader()          # a wrong micro-reader is not a fallback
        try:
            with _using(path, tiles) as entry:
                keys = [dict(k) for k in entry.file.indices]
        except micro_reader.UnsupportedFormatError as exc:
            logger.info(f"pixel_reader=micro: {path}: standard reader ({exc})")
            return None
        except Exception as exc:                    # noqa: BLE001 - fall back, say why
            logger.warning(f"pixel_reader=micro: {path}: standard reader "
                           f"({type(exc).__name__}: {exc})")
            return None
        levels = set().union(*keys) - {SERIES, "resolution"}
        if (levels - SELECTABLE) or (levels and not path.lower().endswith(".czi")):
            logger.info(f"pixel_reader=micro: {path}: standard reader (micro-reader sees "
                        f"{sorted(levels)}, which eubi-bridge reads differently here)")
            return None
        return cls(path, tiles, keys, as_mosaic=as_mosaic, view_index=view_index,
                   illumination_index=illumination_index, phase_index=phase_index,
                   rotation_index=rotation_index)

    # -- what eubi-bridge asks -------------------------------------------------

    @property
    def path(self) -> str:
        return self._path

    @property
    def series_path(self) -> str:
        parts = [self._path, f"_{self.series}"]
        if self._n_views > 1:
            parts.append(f"_view{self.view}")
        if self._n_illuminations > 1:
            parts.append(f"_illu{self.illumination}")
        if self.as_mosaic is False and self.tile:
            parts.append(f"_tile{self.tile}")
        return "".join(parts)

    @property
    def n_scenes(self) -> int:
        return len(self._scene_values)

    @property
    def n_tiles(self) -> int:
        return max(1, len(self._tile_values))

    @property
    def n_views(self) -> int:
        return self._n_views

    @property
    def n_illuminations(self) -> int:
        return self._n_illuminations

    def set_scene(self, scene_index: int) -> None:
        if not 0 <= scene_index < self.n_scenes:
            raise IndexError(f"Scene index {scene_index} out of range [0, {self.n_scenes})")
        self.series = int(scene_index)
        scene = self._scene_values[self.series]
        here = [k for k in self._keys if k[SERIES] == scene]
        self._tile_values = sorted({k["tile"] for k in here if "tile" in k})
        views = sorted({k["view"] for k in here if "view" in k}) or [0]
        self.view, self._n_views = _resolve_index(self._view_index, len(views))
        self.tile = 0
        self._illumination_values = sorted({k["illumination"] for k in here
                                            if "illumination" in k})
        self._phase_values = sorted({k["phase"] for k in here if "phase" in k})
        self.illumination, self._n_illuminations = _resolve_index(
            self._illumination_index, max(1, len(self._illumination_values)))

    def set_tile(self, tile_index: int) -> None:
        if not 0 <= tile_index < self.n_tiles:
            raise IndexError(f"Tile index {tile_index} out of range [0, {self.n_tiles})")
        self.tile = int(tile_index)

    def set_view(self, view_index: int) -> None:
        self.view = int(view_index)

    def set_illumination(self, illumination_index: int) -> None:
        self.illumination = int(illumination_index)

    @property
    def sample_layout(self):
        """``(channels, samples)`` for RGB pixels, else None (samples are read
        as channels, C x S, as eubi-bridge's sample_channels does)."""
        image = self._image()
        if "s" not in image.axes:
            return None
        c = image.shape[image.axes.index("c")] if "c" in image.axes else 1
        return c, image.shape[image.axes.index("s")]

    def get_image_dask_data(self, *args, **kwargs):
        """The current scene / tile / view / illumination as T C Z Y X (RGB
        samples as channels): a ``DynamicArray`` over a ``MicroSource``."""
        return self.get_resolution_level_dask_data(0)

    @property
    def n_resolution_levels(self) -> int:
        """Resolution levels the file stores for the current selection (1:
        full resolution only; CZI pyramids belong to whole scenes, not tiles)."""
        return len(self._image().resolutions)

    def get_resolution_level_dask_data(self, level: int):
        """Resolution *level* of the current selection, as T C Z Y X (as
        eubi-bridge's Imaris reader offers its levels)."""
        from dyna_zarr import from_array
        image, index = self._image(with_index=True)
        view = image.resolutions[level].as_axes(STANDARD, samples="channels")
        return from_array(MicroSource(self._path, self._tiles, index,
                                      view.shape, view.dtype, level=level,
                                      read_unit=view.read_unit))

    # -- the image the selection means -----------------------------------------

    def _key(self) -> dict:
        key = {SERIES: self._scene_values[self.series]}
        if self._tile_values:
            key["tile"] = self._tile_values[self.tile]
        sample = next(k for k in self._keys if k[SERIES] == key[SERIES])
        if "view" in sample:
            key["view"] = self.view
        if "rotation" in sample:
            key["rotation"] = self._rotation
        # illuminations / phases: eubi-bridge counts from 0, the index holds
        # the file's own values
        if getattr(self, "_illumination_values", None):
            key["illumination"] = self._illumination_values[self.illumination]
        if getattr(self, "_phase_values", None):
            key["phase"] = self._phase_values[self._phase]
        return key

    def _image(self, with_index: bool = False):
        key = self._key()
        with _using(self._path, self._tiles) as entry:
            # every index of a file with a pyramid also says "resolution": 0
            found = [i for i, k in enumerate(entry.file.indices)
                     if {n: v for n, v in k.items() if n != "resolution"} == key]
            if len(found) != 1:
                raise ValueError(f"{self._path}: micro-reader has {len(found)} images indexed {key}")
            image = entry.file.images[found[0]]
        return (image, found[0]) if with_index else image


def _with_levels(path: str) -> list:
    """Every image of a CZI with its stored resolution levels, as Bio-Formats
    lists them when its own pyramid collapse does not apply (S=2_2x2: 2 scenes
    x 2 levels = 4 images).  [] for other formats."""
    if not path.lower().endswith(".czi"):
        return []
    try:
        with _using(path, "mosaic") as entry:
            return [level for image in entry.file.images for level in image.resolutions]
    except Exception:                               # noqa: BLE001 - just no count
        return []


def metadata_mismatch(reader: "MicroReader", omemeta) -> str:
    """'' when micro-reader's images agree with the file's OME metadata (bfio),
    else why not.  No pixels are read; without metadata: ''.

    Bio-Formats lists a CZI's views as separate images, so the count compared
    is micro-reader's images in those terms -- one per scene x rotation x
    view, tiles stitched -- not its scenes (Bugra: 25 scenes x 2 views pass
    against 50 images).  A CZI may also pass with its resolution levels counted
    (Bio-Formats lists stored levels as images when its collapse fails).  The images are then matched without relying on
    order: each needs its own metadata image with the same T and Z, Y and X
    (whole images, not tiles), dtype, and C (C x S or C for RGB; any C when
    illuminations split the channels)."""
    images = getattr(omemeta, "images", None)
    if not images:
        return ""
    from eubi_bridge.core.data_manager import _ome_dtype
    with _using(reader._path, reader._tiles) as entry:
        keys, imgs = entry.file.indices, entry.file.images
        combos, illuminations = {}, {}
        for key, image in zip(keys, imgs):          # first tile stands for its image
            combo = (key[SERIES], key.get("rotation", 0), key.get("view", 0))
            combos.setdefault(combo, image)
            illuminations.setdefault(combo, set()).add(key.get("illumination", 0))
        tiled = any("tile" in key for key in keys)
        if len(combos) != len(images) and len(_with_levels(reader._path)) != len(images):
            return (f"{len(combos)} images ({reader.n_scenes} scenes x rotations x views) "
                    f"vs {len(images)} in the metadata")
        unused = list(range(len(images)))
        for (scene, rotation, view), image in combos.items():
            t, c, z, y, x = image.as_axes(STANDARD, samples="channels").shape
            dtype = np.dtype(image.dtype)
            samples = image.shape[image.axes.index("s")] if "s" in image.axes else 1
            # Bio-Formats folds illuminations into the channels
            split = len(illuminations[(scene, rotation, view)]) > 1
            channels = None if split else {c, c // samples}

            def fits(pix):
                return ((pix.size_t, pix.size_z) == (t, z)
                        and (tiled or (pix.size_y, pix.size_x) == (y, x))
                        and (channels is None or pix.size_c in channels)
                        and _ome_dtype(pix.type) == dtype)

            match = next((j for j in unused if fits(images[j].pixels)), None)
            if match is None:
                return (f"no image in the metadata matches scene {scene}"
                        f"{f' view {view}' if view else ''} (T{t} C{c} Z{z} Y{y} X{x} {dtype})")
            unused.remove(match)
    return ""


# -- metadata from micro-reader (readers.metadata_reader="micro") ----------------

#: numpy dtype -> OME PixelType (OME has no half float: float16 is "float")
_OME_PIXEL_TYPES = {"int8": "int8", "int16": "int16", "int32": "int32", "uint8": "uint8",
                    "uint16": "uint16", "uint32": "uint32", "float16": "float",
                    "float32": "float", "float64": "double", "complex64": "complex",
                    "complex128": "double-complex", "bool": "bit"}


def _ome_unit(enum, quantity, fallback: str):
    """(value, OME unit) for a micro-reader Quantity: its own unit when OME
    names it, else converted to *fallback*."""
    name = quantity.unit.upper()
    if name in enum.__members__:
        return quantity.value, enum[name]
    converted = quantity.to(fallback)
    return converted.value, enum[fallback.upper()]


def micro_omemeta(reader: "MicroReader"):
    """An OME object from micro-reader's metadata (micro-reader
    docs/METADATA.md), one image per scene as ``reader`` counts them -- what
    PFFImageMeta reads from Bio-Formats otherwise: sizes, type, pixel sizes
    and units, channels (name, colour, wavelengths).  What the file does not
    say stays unset (eubi-bridge's defaults apply).  Channels follow the
    native c axis; RGB samples are expanded later (sample_channels), as for
    Bio-Formats' metadata.  The reader is left at scene 0."""
    from ome_types.model import OME, Channel, Image, MetadataOnly, Pixels, UnitsLength, UnitsTime
    from ome_types.model.simple_types import Color

    images = []
    for s in range(reader.n_scenes):
        reader.set_scene(s)
        image = reader._image()
        size = {a: image.shape[image.axes.index(a)] if a in image.axes else 1 for a in STANDARD}
        samples = image.shape[image.axes.index("s")] if "s" in image.axes else 1
        meta = image.metadata
        pixels = dict(id=f"Pixels:{s}", dimension_order="XYZCT",
                      type=_OME_PIXEL_TYPES[np.dtype(image.dtype).name],
                      size_x=size["x"], size_y=size["y"], size_z=size["z"],
                      size_c=size["c"] * samples, size_t=size["t"],
                      metadata_only=MetadataOnly())
        for axis in "xyz":
            q = meta.pixel_sizes.get(axis)
            if q is not None:
                value, unit = _ome_unit(UnitsLength, q, "micrometer")
                pixels[f"physical_size_{axis}"] = value
                pixels[f"physical_size_{axis}_unit"] = unit
        q = meta.pixel_sizes.get("t")
        if q is not None:
            pixels["time_increment"], pixels["time_increment_unit"] = \
                _ome_unit(UnitsTime, q, "second")
        channels = []
        for c, ch in enumerate(meta.channels):
            kwargs = dict(id=f"Channel:{s}:{c}", samples_per_pixel=samples)
            if ch.name:
                kwargs["name"] = ch.name
            if ch.color:
                kwargs["color"] = Color(ch.color)
            for kind in ("emission", "excitation"):
                q = getattr(ch, f"{kind}_wavelength")
                if q is not None:
                    value, unit = _ome_unit(UnitsLength, q.to("nanometer"), "nanometer")
                    kwargs[f"{kind}_wavelength"], kwargs[f"{kind}_wavelength_unit"] = value, unit
            channels.append(Channel(**kwargs))
        pixels["channels"] = channels
        images.append(Image(id=f"Image:{s}", name=image.name, pixels=Pixels(**pixels)))
    reader.set_scene(0)
    return OME(images=images)


def micro_extras(reader: "MicroReader", unitdict: dict) -> tuple:
    """For the reader's current selection: (origin, acquisition) -- the
    image's position as eubi-bridge's ``origindict`` (in the x / y scale
    units, written as the NGFF translation: D3), and what OME-Zarr has no
    field for, for the ``eubi_bridge`` attrs block (D4): channel wavelengths
    and values along non-standard axes.  Empty where the file says nothing."""
    image = reader._image()
    meta = image.metadata
    origin = {}
    for axis, q in meta.position.items():
        unit = (unitdict or {}).get(axis) or "micrometer"
        try:
            origin[axis] = float(q.to(unit).value)
        except KeyError:                    # a unit micro-reader does not convert
            origin = {}
            break
    acquisition = {}
    waves = []
    for ch in meta.channels:
        entry = {k: {"value": q.to("nanometer").value, "unit": "nanometer"}
                 for k, q in (("emission_wavelength", ch.emission_wavelength),
                              ("excitation_wavelength", ch.excitation_wavelength))
                 if q is not None}
        waves.append(entry)
    if any(waves):
        acquisition["channel_wavelengths"] = waves
    if meta.index_values:
        acquisition["index_values"] = {name: {"value": q.value, "unit": q.unit}
                                       for name, q in meta.index_values.items()}
    return origin, acquisition
