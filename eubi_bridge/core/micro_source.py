"""micro-reader as eubi-bridge's pixel reader (``readers.pixel_reader="micro"``).

A first, opt-in version: metadata, scene counting, channels and the writer
are unchanged; only the pixels of each (scene, tile, view, illumination)
snapshot come from micro-reader instead of the standard reader, through
``MicroSource`` -- an array-like that ``DynamicArray`` wraps, as it wraps
``_BioFormatsSource``.

Safety net: every micro-reader array is checked against the standard
reader's array for the same snapshot -- the same T C Z Y X shape, and the
same pixels in a small window of the first and last plane.  Anything that
does not map or does not match keeps the standard array and logs why.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Optional

import numpy as np

from eubi_bridge.utils.logging_config import get_logger

logger = get_logger(__name__)

PIXEL_READERS = ("standard", "micro")
#: a dimension a file does not name (plain multi-page TIFF pages) is z, as
#: Bio-Formats reads it
RENAME = {"sequence": "z"}
#: image indices eubi-bridge selects by (its scene / tile / view /
#: illumination / rotation / phase options); a file with any other index
#: (block, emission, excitation, lifetime) is left to the standard reader
SELECTABLE = {"scene", "tile", "rotation", "view", "illumination", "phase", "resolution"}
STANDARD = "tczyx"
#: side of the window compared per plane by ``verify``
CHECK_SIDE = 64

#: files kept open per process, least recently used closed first: an
#: aggregative input of thousands of files must not hold them all open (each
#: holds a handle per reading thread; "Too many open files" at ~500 of 2347)
MAX_OPEN_FILES = 32


class _Open:
    """An open micro-reader file, its axes images, and how many reads use it."""

    def __init__(self, file):
        self.file, self.views, self.users = file, {}, 0


#: (path, tiles) -> _Open, oldest first
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
def _using(path: str, tiles: str):
    """The open file for (*path*, *tiles*), opened (again) if needed, held
    open while the block runs."""
    import micro_reader
    key = (path, tiles)
    with _files_lock:
        entry = _files.get(key)
        if entry is None:
            entry = _files[key] = _Open(micro_reader.open(path, tiles=tiles, rename=RENAME))
        else:
            _files.move_to_end(key)
        entry.users += 1
        _evict()
    try:
        yield entry
    finally:
        with _files_lock:
            entry.users -= 1
            _evict()


def set_read_concurrency(n: int, memory: Optional[str] = None) -> None:
    """micro-reader's decode threads in this process (``micro_read_concurrency``):
    one pool shared by all region reads, so this caps the CPU spent reading
    without changing how many regions the writer keeps in flight.  *memory*
    (``memory_per_worker``, e.g. "3GB") is micro-reader's decoding budget:
    a compressed block larger than memory / n is streamed, or refused with a
    clear error where its codec cannot stream, instead of running the worker
    out of memory."""
    import micro_reader
    if micro_reader.decode_threads() != int(n):
        micro_reader.set_decode_threads(int(n))
    if memory:
        micro_reader.set_memory_limit(memory)


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

    def __init__(self, path: str, tiles: str, index: int, shape, dtype, level: int = 0):
        self.path, self.tiles, self.index = path, tiles, int(index)
        #: resolution level of the image (0: full resolution)
        self.level = int(level)
        self.shape = tuple(int(n) for n in shape)
        self.dtype = np.dtype(dtype)
        self.ndim = len(self.shape)
        # any rectangle can be read: a whole plane is advertised as one chunk
        self.chunks = (1, 1, 1, self.shape[3], self.shape[4])

    def __getitem__(self, key):
        with _using(self.path, self.tiles) as entry:
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
                view: int = 0, illumination: int = 0, rotation: int = 0, phase: int = 0):
    """The micro-reader image eubi-bridge means by (scene index, tile, view,
    illumination, rotation, phase), as a ``DynamicArray`` -- or (None,
    reason) when micro-reader cannot read the file or has no such image.

    eubi-bridge's rules: *scene* is the rank among the file's scenes;
    *as_mosaic* stitches tiles, otherwise *tile* is the mosaic tile index
    (CZI ``M``; None: tile 0), and on a file without tiles tile 0 is the
    whole image (``set_tile`` is a no-op there).
    """
    import micro_reader
    from eubi_bridge.external.dyna_zarr.dynamic_array import DynamicArray
    tiles = "stitched" if as_mosaic else "separate"
    try:
        with _using(path, tiles) as entry:
            return _find(entry.file, path, tiles, scene, tile, view=view,
                         illumination=illumination, rotation=rotation, phase=phase)
    except micro_reader.UnsupportedFormatError as exc:
        if tiles == "stitched":                      # tiled, but not composable
            return None, f"micro-reader: {exc}"
        return None, f"micro-reader cannot read it: {exc}"
    except Exception as exc:                        # noqa: BLE001 - fall back, say why
        return None, f"micro-reader cannot read it: {type(exc).__name__}: {exc}"


def _find(f, path: str, tiles: str, scene: int, tile: Optional[int], *, view: int,
          illumination: int, rotation: int, phase: int):
    """``micro_array``'s lookup in the open file *f*."""
    from eubi_bridge.external.dyna_zarr.dynamic_array import DynamicArray
    scenes = sorted({k["scene"] for k in f.indices})
    if not 0 <= scene < len(scenes):
        return None, f"scene index {scene}: micro-reader has {len(scenes)} scenes"
    want = {"scene": scenes[scene]}
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
    source = MicroSource(path, tiles, f.images.index(image), view_.shape, view_.dtype)
    return DynamicArray(source), ""


def _window(arr, t, c, z, ny, nx) -> np.ndarray:
    """A CHECK_SIDE window in the middle of one plane, asked for with slices
    only (eubi-bridge's own lazy arrays mishandle integer indices)."""
    y0, x0 = max(0, ny // 2 - CHECK_SIDE // 2), max(0, nx // 2 - CHECK_SIDE // 2)
    key = (slice(t, t + 1), slice(c, c + 1), slice(z, z + 1),
           slice(y0, min(ny, y0 + CHECK_SIDE)), slice(x0, min(nx, x0 + CHECK_SIDE)))
    part = arr[key]
    if hasattr(part, "compute"):
        part = part.compute()
    return np.asarray(part)


def verify(micro, standard, pixels: bool = True) -> str:
    """'' when *micro* matches *standard* -- shape, dtype and, with *pixels*,
    the pixels in a window of the first and last plane; otherwise why not.
    Shape and dtype cost nothing; the pixel windows are read from both."""
    if tuple(micro.shape) != tuple(standard.shape):
        return f"shape {tuple(micro.shape)} vs standard {tuple(standard.shape)}"
    if np.dtype(micro.dtype) != np.dtype(standard.dtype):
        return f"dtype {np.dtype(micro.dtype)} vs standard {np.dtype(standard.dtype)}"
    if not pixels:
        return ""
    nt, nc, nz, ny, nx = micro.shape
    for t, c, z in {(0, 0, 0), (nt - 1, nc - 1, nz - 1)}:
        a, b = _window(micro, t, c, z, ny, nx), _window(standard, t, c, z, ny, nx)
        if a.shape != b.shape or not np.array_equal(a, b):
            return f"pixels differ at t{t} c{c} z{z}"
    return ""


def layout(path: str, arr) -> tuple:
    """What files whose arrays map alike share: file type, shape, dtype."""
    return os.path.splitext(path)[1].lower(), tuple(arr.shape), str(np.dtype(arr.dtype))


def swap_in(loader_img, path: str, scene: int, tile: Optional[int], *, as_mosaic=False,
            view=0, illumination=0, rotation=0, phase=0, verify_pixels: bool = True,
            verified_layouts: Optional[set] = None):
    """micro-reader's array for this snapshot when it maps and matches the
    standard one (``loader_img.arraydata``), else None (and a log line).

    *verified_layouts*: for many files of one kind (an aggregative input),
    pixels are compared once per ``layout`` -- the first file that passes
    the full check adds its layout to the set -- and every other file only
    by shape and dtype.  The pixel check reads a whole plane through the
    standard reader for each file, ~20 ms (45 s of a 2347-file run)."""
    standard = loader_img.arraydata
    if standard is None:
        return None
    arr, reason = micro_array(path, scene, tile, as_mosaic=as_mosaic, view=view,
                              illumination=illumination, rotation=rotation, phase=phase)
    if arr is None:
        logger.warning(f"pixel_reader=micro: {path} scene {scene} tile {tile}: standard "
                       f"reader kept ({reason})")
        return None
    if verify_pixels:
        key = layout(path, arr)
        pixels = verified_layouts is None or key not in verified_layouts
        try:
            problem = verify(arr, standard, pixels=pixels)
        except Exception as exc:                    # noqa: BLE001 - fall back, say why
            problem = f"check failed: {type(exc).__name__}: {exc}"
        if problem:
            logger.warning(f"pixel_reader=micro: {path} scene {scene} tile {tile}: standard "
                           f"reader kept ({problem})")
            return None
        if pixels and verified_layouts is not None:
            verified_layouts.add(key)
    logger.info(f"pixel_reader=micro: {path} scene {scene} tile {tile} view {view} "
                f"illumination {illumination}: micro-reader {tuple(arr.shape)} {arr.dtype}")
    return arr


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
    a micro-reader file: scenes, tiles, views and illuminations from its
    image indices, pixels as ``MicroSource`` arrays.  No standard reader is
    opened.  Semantics follow eubi-bridge's CZI reader (scene = rank, tile =
    position within the scene, view / illumination exposure, series_path).

    ``MicroReader.open`` returns None when micro-reader cannot read the file,
    or when the file has tiles / views / rotations outside a CZI (eubi-bridge's
    standard readers treat those differently): the standard reader is used.
    """

    def __init__(self, path: str, tiles: str, keys: list, *, as_mosaic: bool,
                 view_index=0, illumination_index=0, phase_index=0, rotation_index=0):
        self._path, self._tiles, self._keys = path, tiles, keys
        self.as_mosaic = as_mosaic
        self._scene_values = sorted({k["scene"] for k in keys})
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
        tiles = "stitched" if as_mosaic else "separate"
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
        levels = set().union(*keys) - {"scene", "resolution"}
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
        here = [k for k in self._keys if k["scene"] == scene]
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
        from eubi_bridge.external.dyna_zarr.dynamic_array import DynamicArray
        image, index = self._image(with_index=True)
        view = image.resolutions[level].as_axes(STANDARD, samples="channels")
        return DynamicArray(MicroSource(self._path, self._tiles, index,
                                        view.shape, view.dtype, level=level))

    # -- the image the selection means -----------------------------------------

    def _key(self) -> dict:
        key = {"scene": self._scene_values[self.series]}
        if self._tile_values:
            key["tile"] = self._tile_values[self.tile]
        sample = next(k for k in self._keys if k["scene"] == key["scene"])
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
        with _using(path, "stitched") as entry:
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
            combo = (key["scene"], key.get("rotation", 0), key.get("view", 0))
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
