"""The older readers (Bio-Formats, bioio, CZI, IMS, H5) as micro-reader-like images.

``LegacyFile`` opens a file the way eubi-bridge always has (``SceneLoader._open``: the
reader routing, the OME metadata with its resolution-series collapse, the damaged-file
cap) and presents it like a micro-reader ``File``: ``indices``, ``image(index)``,
``close()``. Each ``LegacyImage`` carries the reader's own lazy array (``array``: a dask
array or a ``DynamicArray``, used as it is by ``Pyramid.from_image``) and the metadata
ozp reads: pixel sizes and units, channels, the stage position of a CZI tile.

So the conversion path of eubi-bridge 0.2.0 (``pyramid_path.py``) converts these files
with the same steps as micro-reader files; nothing goes through ``ArrayManager``.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Optional

from eubi_bridge.conversion.pyramid_path import SERIES
from eubi_bridge.utils.logging_config import get_logger

logger = get_logger(__name__)

#: length units eubi-bridge's readers report, in metres (for Quantity.to)
_LENGTH = {"meter": 1.0, "millimeter": 1e-3, "micrometer": 1e-6, "nanometer": 1e-9,
           "angstrom": 1e-10, "picometer": 1e-12, "centimeter": 1e-2, "kilometer": 1e3}
#: time units, in seconds
_TIME = {"second": 1.0, "millisecond": 1e-3, "microsecond": 1e-6, "nanosecond": 1e-9,
         "minute": 60.0, "hour": 3600.0}


class Quantity:
    """A value with an NGFF / UDUNITS unit name, convertible within length or time."""

    def __init__(self, value: float, unit: str):
        self.value, self.unit = float(value), str(unit)

    def to(self, unit: str) -> "Quantity":
        for table in (_LENGTH, _TIME):
            if self.unit in table and unit in table:
                return Quantity(self.value * table[self.unit] / table[unit], unit)
        raise KeyError(f"no conversion from {self.unit} to {unit}")

    def __repr__(self) -> str:
        return f"Quantity({self.value}, {self.unit!r})"


def _run(coroutine):
    """The readers' async API, from the worker thread the conversion opens files in."""
    return asyncio.run(coroutine)


def _color(color) -> Optional[str]:
    """An OME colour as six hex digits (ChannelParser._normalize_color): the readers
    hand over 12 digits for a 6-digit colour (expand_hex_shorthand)."""
    if not color:
        return None
    text = str(color).lstrip("#")
    return text[::2] if len(text) == 12 else text[:6]


def _channels(img, acquisition: dict) -> list:
    """The reader's own channels (names, colours), as 0.1.x wrote them: an
    unnamed channel becomes ``Channel {i}`` in ozp's ``from_image``, a colourless one
    white; a file listing fewer channels than it has gets eubi-bridge's generated ones
    (the reader's ``get_channels``). Wavelengths only from micro-reader's metadata."""
    described = img.get_channels() or []
    waves = list((acquisition or {}).get("channel_wavelengths") or [])
    out = []
    for i, ch in enumerate(described):
        wave = waves[i] if i < len(waves) else {}
        out.append(SimpleNamespace(
            name=ch.get("label") or None, color=_color(ch.get("color")),
            emission_wavelength=_wave(wave.get("emission_wavelength")),
            excitation_wavelength=_wave(wave.get("excitation_wavelength"))))
    return out


def _wave(entry) -> Optional[Quantity]:
    return Quantity(entry["value"], entry["unit"]) if entry else None


class _SerialisedReads:
    """A resource-backed dask array read one region at a time. bioio-nd2's array and
    Bio-Formats-backed dask arrays read through one shared handle that is not
    thread-safe: read from several threads at once they return wrong pixels without an
    error, or crash (output_store._is_resource_backed; 0.1.x's own writer locked them,
    ozp's dask writer does not)."""

    def __init__(self, array):
        self._array = array
        self.shape, self.dtype, self.ndim = tuple(array.shape), array.dtype, array.ndim

    def __getitem__(self, key):
        import numpy as np
        return np.asarray(self._array[key].compute(scheduler="synchronous"))


def _thread_safe(array):
    """`array` as it is, or, when it is resource-backed, a dask array over the same
    chunks whose reads run one at a time (dask's per-array read lock); the writes stay
    parallel."""
    from eubi_bridge.core.output_store import _is_resource_backed
    if not _is_resource_backed(array):
        return array
    import dask.array as da
    import numpy as np
    return da.from_array(_SerialisedReads(array), chunks=array.chunks, lock=True,
                         meta=np.empty((0,) * array.ndim, dtype=array.dtype))


class LegacyImage:
    """One image (series, CZI tile) of a file the older readers read, micro-reader-like."""

    def __init__(self, *, array, axes: str, index: dict, name: str, pixel_sizes: dict,
                 channels: list, position: dict, index_values: dict):
        array = _thread_safe(array)
        self.array = array
        self.axes = str(axes).lower()
        self.shape = tuple(int(n) for n in array.shape)
        self.dtype = array.dtype
        self.index = dict(index)
        self.name = name
        self.resolutions = [self]
        #: samples are already channels here (the readers fold them, and name them
        #: with eubi-bridge's RGB rule): no "s" for pyramid_path's RGB naming
        self.source = self
        self.metadata = SimpleNamespace(pixel_sizes=pixel_sizes, channels=channels,
                                        position=position, index_values=index_values)

    def __getitem__(self, key):
        import numpy as np
        part = self.array[key]
        return np.asarray(part.compute() if hasattr(part, "compute") else part)


class LegacyFile:
    """A file opened with eubi-bridge's older readers, presented like a micro-reader
    ``File``: ``indices`` (series, and tile for an unstitched CZI mosaic, named as
    micro-reader 0.1.0 names them, so one loop reads both), ``image(index)`` (built on
    demand, one at a time: the readers switch scenes on one object), ``close()``."""

    def __init__(self, job):
        from eubi_bridge.core.data_manager import SceneLoader
        readers, meta = job.readers, job.metadata
        self.path = str(job.input_path)
        # a CZI's phase and rotation (0.1.x never passed them on in a unary conversion:
        # always 0); the other readers take no such option
        czi_kwargs = ({"phase_index": int(readers.phase_index or 0),
                       "rotation_index": int(readers.rotation_index or 0)}
                      if self.path.lower().endswith(".czi") else {})
        self._loader = SceneLoader(
            self.path, meta.metadata_reader, False, reader_kwargs=czi_kwargs,
            force_bioformats=readers.force_bioformats, as_mosaic=readers.as_mosaic,
            keep_existing_resolutions=job.downscale.keep_existing_resolutions,
            pixel_reader=readers.pixel_reader)
        _run(self._loader._open())
        self._img = self._loader._img
        self.n_scenes = int(self._loader.n_scenes)
        self.n_scenes_in_file = int(self._loader.n_scenes_in_file)
        #: tiles only when they were asked for (as SceneLoader.load): positions too
        self._tiles_asked = readers.mosaic_tile_index is not None
        self._keep_levels = bool(job.downscale.keep_existing_resolutions)
        self._tiles = {}
        self._tile_origins = {}

    # -- what the file has ---------------------------------------------------
    def _n_tiles(self, scene: int) -> int:
        """Tiles of `scene`, counted per scene: only an unstitched CZI mosaic has more
        than one."""
        if scene not in self._tiles:
            if self._loader._as_mosaic or not self.path.lower().endswith(".czi"):
                self._tiles[scene] = 1
            else:
                _run(self._img.set_scene_isolated(scene))
                self._tiles[scene] = max(1, int(self._img.n_tiles))
        return self._tiles[scene]

    @property
    def indices(self) -> list:
        tiled = any(self._n_tiles(s) > 1 for s in range(self.n_scenes))
        out = []
        for s in range(self.n_scenes):
            if tiled:
                out += [{SERIES: s, "tile": t} for t in range(self._n_tiles(s))]
            else:
                out.append({SERIES: s})
        return out

    # -- one image -----------------------------------------------------------
    def image(self, index: dict) -> LegacyImage:
        """The image at `index` ({name: value}, or its items), read the way
        SceneLoader._snapshot reads it."""
        index = dict(index)
        scene, tile = int(index[SERIES]), index.get("tile")
        img = self._img
        _run(img.set_scene_isolated(scene))
        if tile is not None:
            _run(img.set_tile(int(tile)))
        array = img.arraydata
        axes = img.get_axes()
        scales = {ax: v for ax, v in img.get_scaledict().items() if ax in axes and ax != "c"}
        units = {ax: u for ax, u in img.get_unitdict().items() if ax in axes and ax != "c"}
        pixel_sizes = {ax: Quantity(v, units.get(ax) or ("second" if ax == "t" else "micrometer"))
                       for ax, v in scales.items() if v is not None}
        origin, acquisition = self._micro_extras(units)
        position = {ax: Quantity(v, units.get(ax) or "micrometer") for ax, v in origin.items()}
        if not position and tile is not None and self._tiles_asked:
            position = self._czi_tile_position(scene, int(tile), scales, units)
        omemeta = getattr(img, "omemeta", None)
        images = getattr(omemeta, "images", None) or []
        name = images[scene].name if scene < len(images) and images[scene].name else None
        index_values = {k: Quantity(v["value"], v["unit"])
                        for k, v in (acquisition.get("index_values") or {}).items()}
        image = LegacyImage(array=array, axes=axes, index=index, name=name,
                            pixel_sizes=pixel_sizes, channels=_channels(img, acquisition),
                            position=position, index_values=index_values)
        if self._keep_levels:
            # the reader's stored levels (IMS), finest first after the image itself
            stored = _run(img.get_pyramid())
            paths = list(stored.meta.resolution_paths) if stored is not None else []
            if len(paths) > 1:
                image.resolutions = [image] + [
                    SimpleNamespace(array=_thread_safe(stored.layers[p]),
                                    shape=tuple(stored.layers[p].shape))
                    for p in paths[1:]]
        return image

    def _micro_extras(self, units: dict) -> tuple:
        """micro-reader's position and wavelengths, when it read the metadata AND the
        pixels (SceneLoader._attach_micro_extras): ({axis: value}, acquisition)."""
        from eubi_bridge.core.micro_source import MicroReader, micro_extras
        reader = getattr(self._img, "reader", None)
        if not getattr(self._img, "_micro_meta", False) or not isinstance(reader, MicroReader):
            return {}, {}
        try:
            return micro_extras(reader, units)
        except Exception as exc:                    # noqa: BLE001 - metadata extras only
            logger.warning(f"metadata_reader=micro: {self.path}: no position / "
                           f"wavelengths ({type(exc).__name__}: {exc})")
            return {}, {}

    def _czi_tile_position(self, scene: int, tile: int, scales: dict, units: dict) -> dict:
        """A CZI tile's place from aicspylibczi's boxes (pixels x pixel size), as
        0.1.x's ArrayManager.load_scenes set it."""
        from eubi_bridge.core.data_manager import czi_mosaic_tile_origins
        if scene not in self._tile_origins:
            self._tile_origins[scene] = czi_mosaic_tile_origins(self.path, scene=scene)
        box = self._tile_origins[scene].get(tile)
        if not box:
            return {}
        return {ax: Quantity(box[ax] * float(scales.get(ax) or 1.0),
                             units.get(ax) or "micrometer") for ax in ("y", "x")}

    def close(self) -> None:
        self._img = None
        self._loader._img = None


class NgffFile:
    """An OME-Zarr input, read by ome_zarr_pyramid itself (roadmap step 2, part 3) and
    presented like a micro-reader ``File``: one image, its level 0 (the array as it is),
    pixel sizes, units, channel names and colours, and its translation as the stage
    position, so the output keeps it (0.1.5 dropped it)."""

    def __init__(self, job):
        from ome_zarr_pyramid import IO

        from eubi_bridge.core.output_store import _ozp_target_args
        self.path = str(job.input_path)
        target, storage_options = _ozp_target_args(self.path)
        self._pyr = IO().read_pyramid(target, storage_options=storage_options)
        #: the input Pyramid itself, written as it is with keep_existing_resolutions
        self.source_pyramid = self._pyr
        self.n_scenes = self.n_scenes_in_file = 1
        self.indices = [{SERIES: 0}]

    def image(self, index) -> LegacyImage:
        meta = self._pyr.meta
        base = meta.resolution_paths[0]
        axes = meta.axis_order
        units = dict(zip(axes, meta.unit_list))
        scales = dict(zip(axes, meta.get_scale(base)))
        pixel_sizes = {ax: Quantity(scales[ax], units.get(ax) or
                                    ("second" if ax == "t" else "micrometer"))
                       for ax in axes if ax != "c"}
        translation = dict(zip(axes, meta.get_translation(base) or [0.0] * len(axes)))
        position = {ax: Quantity(translation[ax], units.get(ax) or "micrometer")
                    for ax in "zyx" if ax in axes and translation.get(ax)}
        omero = (meta.metadata or {}).get("omero") or {}
        channels = [SimpleNamespace(name=ch.get("label") or None, color=_color(ch.get("color")),
                                    emission_wavelength=None, excitation_wavelength=None)
                    for ch in omero.get("channels") or []]
        return LegacyImage(array=self._pyr.layers[base], axes=axes, index=dict(index),
                           name=meta.multiscales.get("name"), pixel_sizes=pixel_sizes,
                           channels=channels,
                           position=position, index_values={})

    def close(self) -> None:
        self._pyr = None
