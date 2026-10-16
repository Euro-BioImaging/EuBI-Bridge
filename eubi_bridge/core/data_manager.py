"""The older readers of EuBI-Bridge.

``ImageReader`` (ABC) and its format-specific loaders (PFFImageMeta, TIFFImageMeta,
H5ImageMeta, IMSImageMeta): async I/O, one scene / tile at a time. OME-Zarr inputs are
read by ome_zarr_pyramid itself (``legacy_image.NgffFile``).
``SceneLoader`` picks the loader for a file and opens it; the conversion reads the
images from it through ``conversion/legacy_image.py`` (``LegacyFile``) when micro-reader
does not take the file.

eubi-bridge 0.1.x converted through ``ArrayManager`` / ``ArrayState`` / ``BatchManager``
here; that layer is retired (kept for reference in ``legacy_0_1/``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import asyncio
import copy
import json
from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional, Union

import numpy as np
from ome_types.model import (OME, Channel, Image, Pixels,
                             Pixels_DimensionOrder, PixelType, UnitsLength,
                             UnitsTime)

from eubi_bridge.core.czi_reader import CZIReader
from eubi_bridge.core.readers import (
    read_metadata_via_bfio, read_metadata_via_bioio_bioformats,
    read_metadata_via_extension, read_single_image)
from dyna_zarr import DynamicArray, from_array
from ome_zarr_pyramid import Pyramid
from eubi_bridge.utils.array_utils import normalise_basic_index
from eubi_bridge.utils.logging_config import get_logger
from eubi_bridge.utils.path_utils import is_zarr_group

logger = get_logger(__name__)

#: eubi-bridge's axes, and the pixel size / unit an axis gets when the file has none
default_axes = 'tczyx'
scale_map = {'t': 1, 'c': 1, 'z': 1, 'y': 1, 'x': 1}
unit_map = {'t': 'second', 'c': 'Channel', 'z': 'micrometer', 'y': 'micrometer',
            'x': 'micrometer'}


# ---------------------------------------------------------------------------
# bfio tiled reading helpers (bioformats fallback path only)
# ---------------------------------------------------------------------------

def _is_bioformats_backed(reader: object) -> bool:
    """Return True when *reader* is backed by the ``bioio_bioformats`` plugin.

    This is the fallback reader for formats without a native bioio extension
    (e.g. ``.mrc``, ``.dm4``).  Native plugins (CZI, ND2, LIF, TIFF…) return
    False and keep their existing ``get_image_dask_data()`` path.
    """
    try:
        return "bioio_bioformats" in type(reader.img).__module__
    except AttributeError:
        return False


# Thread-local BioReader cache — one open reader per PATH per dask thread.
#
# Bio-Formats' setId() (called inside BioReader.__init__) costs 7-10 s because it
# parses the entire file index.  For a multi-series file (e.g. LIF with 10 series)
# this cost was paid once per series per thread.
#
# Fix: cache at the PATH level.  The first access for a given (path, thread) pays
# setId() once.  All subsequent accesses for ANY series of that file call setSeries()
# on the already-open Java ImageReader — that is essentially free.
import threading as _threading

if TYPE_CHECKING:
    import dask.array as da
_tl_readers: _threading.local = _threading.local()


def _get_cached_reader(path: str, series: int):
    """Return a cached BioReader for this thread.

    setId() is called only once per path per thread.  Series switching is free
    via setSeries() on the underlying Java reader.

    bfio validates reads against its Python-level br.Z/Y/X, which are set at
    open time and reflect the ORIGINAL series.  Switching to a series with
    LARGER dimensions would cause z1 > br.Z assertion errors.  When that
    happens we close and re-open so Python metadata is refreshed.  Switching
    to a series with SMALLER or equal dimensions is always safe (stale br.Z is
    conservative).
    """
    from bfio import BioReader
    if not hasattr(_tl_readers, 'cache'):
        _tl_readers.cache = {}
    if path not in _tl_readers.cache:
        _tl_readers.cache[path] = BioReader(path, level=series)
        return _tl_readers.cache[path]

    br = _tl_readers.cache[path]
    try:
        rdr = br._backend._rdr
        if int(rdr.getSeries()) != series:
            rdr.setSeries(series)
            # After switching, check whether new series is larger than the
            # stale Python metadata.  Java dimensions are always up-to-date.
            new_z = int(rdr.getSizeZ())
            new_y = int(rdr.getSizeY())
            new_x = int(rdr.getSizeX())
            if new_z > br.Z or new_y > br.Y or new_x > br.X:
                # Re-open so bfio's Python layer sees the correct dimensions.
                try:
                    br.close()
                except Exception:
                    pass
                _tl_readers.cache[path] = BioReader(path, level=series)
    except Exception:
        pass
    return _tl_readers.cache[path]


def _bioformats_resolution_count(path: str) -> Optional[int]:
    """Return Bio-Formats' CoreMetadata.resolutionCount for series 0, or None."""
    try:
        br = _get_cached_reader(path, 0)
        core = br._backend._rdr.getCoreMetadataList()
        return int(core[0].resolutionCount)
    except Exception:
        return None


def _collapse_resolution_series(omemeta: OME, path: str) -> OME:
    """Collapse Bio-Formats-flattened resolution-pyramid <Image> entries.

    bioio_bioformats reports each internal resolution level of a pyramidal
    image as a separate <Image> (flattenedResolutions=True). When
    CoreMetadata.resolutionCount for series 0 equals the number of reported
    <Image> elements, every one of them is a resolution level of a single
    image -- keep only the full-resolution (first) one.
    """
    if len(omemeta.images) <= 1:
        return omemeta
    res_count = _bioformats_resolution_count(path)
    if res_count is not None and res_count == len(omemeta.images):
        omemeta.images = omemeta.images[:1]
    return omemeta


def _ome_dtype(pixel_type) -> np.dtype:
    """Convert an OME PixelType enum (or its string value) to a numpy dtype."""
    name = (pixel_type.value if hasattr(pixel_type, 'value') else str(pixel_type)).lower()
    _MAP = {
        'uint8': np.uint8, 'uint16': np.uint16, 'uint32': np.uint32,
        'int8': np.int8, 'int16': np.int16, 'int32': np.int32,
        'float': np.float32, 'double': np.float64,
        'bit': np.bool_, 'complex': np.complex64, 'double-complex': np.complex128,
    }
    return np.dtype(_MAP.get(name, np.float32))


def _ome_pixeltype_from_dtype(dtype) -> PixelType:
    """Convert a numpy dtype to an OME PixelType enum. Inverse of ``_ome_dtype``."""
    _MAP = {
        np.dtype(np.uint8): PixelType.UINT8,
        np.dtype(np.uint16): PixelType.UINT16,
        np.dtype(np.uint32): PixelType.UINT32,
        np.dtype(np.int8): PixelType.INT8,
        np.dtype(np.int16): PixelType.INT16,
        np.dtype(np.int32): PixelType.INT32,
        np.dtype(np.float32): PixelType.FLOAT,
        np.dtype(np.float64): PixelType.DOUBLE,
        np.dtype(np.bool_): PixelType.BIT,
        np.dtype(np.complex64): PixelType.COMPLEXFLOAT,
        np.dtype(np.complex128): PixelType.COMPLEXDOUBLE,
    }
    return _MAP.get(np.dtype(dtype), PixelType.UINT16)


def _build_ims_omemeta(path: str) -> OME:
    """Build a synthetic single-``<Image>`` OME object from an Imaris (.ims) file.

    Reads ``/DataSetInfo/Image`` (size/extent → physical pixel sizes),
    ``/DataSetInfo/TimeInfo`` (timepoint count/increment), and
    ``/DataSetInfo/Channel {c}`` (channel names) directly via h5py, plus the
    dtype/shape of the ResolutionLevel 0 data.
    """
    import h5py
    from eubi_bridge.core.ims_reader import _ims_attr_str

    with h5py.File(path, 'r') as f:
        img_attrs = f['DataSetInfo/Image'].attrs

        size_x = int(_ims_attr_str(img_attrs, 'X', '1'))
        size_y = int(_ims_attr_str(img_attrs, 'Y', '1'))
        size_z = int(_ims_attr_str(img_attrs, 'Z', '1'))

        def _extent(key, default):
            val = _ims_attr_str(img_attrs, key, None)
            return float(val) if val is not None else default

        ext_min0, ext_max0 = _extent('ExtMin0', 0.0), _extent('ExtMax0', float(size_x))
        ext_min1, ext_max1 = _extent('ExtMin1', 0.0), _extent('ExtMax1', float(size_y))
        ext_min2, ext_max2 = _extent('ExtMin2', 0.0), _extent('ExtMax2', float(size_z))

        physical_size_x = (ext_max0 - ext_min0) / size_x if size_x else 1.0
        physical_size_y = (ext_max1 - ext_min1) / size_y if size_y else 1.0
        physical_size_z = (ext_max2 - ext_min2) / size_z if size_z else 1.0

        res0 = f['DataSet/ResolutionLevel 0']
        n_timepoints = sum(1 for k in res0.keys() if k.startswith('TimePoint'))
        n_channels = sum(1 for k in res0['TimePoint 0'].keys() if k.startswith('Channel'))

        time_increment = 1.0
        if 'TimeInfo' in f.get('DataSetInfo', {}):
            time_attrs = f['DataSetInfo/TimeInfo'].attrs
            if n_timepoints > 1:
                try:
                    from datetime import datetime
                    fmt = '%Y-%m-%d %H:%M:%S.%f'
                    t1 = datetime.strptime(_ims_attr_str(time_attrs, 'TimePoint1').strip(), fmt)
                    t2 = datetime.strptime(_ims_attr_str(time_attrs, 'TimePoint2').strip(), fmt)
                    time_increment = (t2 - t1).total_seconds() or 1.0
                except Exception:
                    time_increment = 1.0

        channels = []
        for c in range(n_channels):
            name = None
            channel_key = f'DataSetInfo/Channel {c}'
            if channel_key in f:
                name = _ims_attr_str(f[channel_key].attrs, 'Name', None)
            channels.append(Channel(id=f"Channel:{c}", name=name or f"Channel {c}"))

        data_dtype = f[f'/DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data'].dtype

    pixels = Pixels(
        dimension_order=Pixels_DimensionOrder.XYZCT,
        type=_ome_pixeltype_from_dtype(data_dtype),
        size_x=size_x, size_y=size_y, size_z=size_z,
        size_c=n_channels, size_t=n_timepoints,
        physical_size_x=physical_size_x, physical_size_x_unit=UnitsLength.MICROMETER,
        physical_size_y=physical_size_y, physical_size_y_unit=UnitsLength.MICROMETER,
        physical_size_z=physical_size_z, physical_size_z_unit=UnitsLength.MICROMETER,
        time_increment=time_increment, time_increment_unit=UnitsTime.SECOND,
        channels=channels,
    )
    return OME(images=[Image(id="Image:0", name="Series_0", pixels=pixels)])


# How many Bio-Formats reads may run at once in one process.  Each needs Java
# heap for its byte[] plus Bio-Formats' working copy, so the per-read size below
# is derived from this: together the concurrent reads stay within half the heap.
_JAVA_READ_SLOTS = 4
_java_read_slots = _threading.BoundedSemaphore(_JAVA_READ_SLOTS)
# Java arrays are indexed by int, so no single read may exceed 2 GB.
_JAVA_MAX_ARRAY_BYTES = 2 ** 31 - 1
_java_read_budget: Optional[int] = None


def _java_read_budget_bytes() -> int:
    """Largest single Bio-Formats read, in bytes, for this process's JVM heap."""
    global _java_read_budget
    if _java_read_budget is None:
        try:
            import jpype
            heap = int(jpype.JClass("java.lang.Runtime").getRuntime().maxMemory())
        except Exception:                                   # noqa: BLE001
            heap = 1024 ** 3        # the JVM's usual default when unknown
        # Two copies per read (Java's buffer, Bio-Formats' working copy), and
        # _JAVA_READ_SLOTS reads at once, sharing half the heap.
        per_read = heap // (2 * 2 * _JAVA_READ_SLOTS)
        _java_read_budget = int(max(1024 ** 2, min(per_read, _JAVA_MAX_ARRAY_BYTES)))
    return _java_read_budget


class _BioFormatsSource:
    """A Bio-Formats image read lazily, one requested rectangle at a time.

    Wrapped in a ``DynamicArray``, so every writer region becomes a single read
    of exactly the pixels it needs.  The fixed tile grid this replaces made a
    region read every whole tile it touched: strip-shaped regions across
    square tiles re-read each tile several times over, which exhausted both
    RAM and the Java heap on large planes.  Bio-Formats reads sub-plane
    rectangles natively, so nothing larger than the region is ever loaded.
    """

    def __init__(self, path: str, series: int, shape, dtype):
        self.path = path
        self.series = int(series)
        self.shape = tuple(int(v) for v in shape)        # (T, C, Z, Y, X)
        self.dtype = np.dtype(dtype)
        self.ndim = len(self.shape)
        # Any rectangle can be read, so a whole plane is advertised as one chunk
        # and region alignment follows the output chunks alone.
        self.chunks = (1, 1, 1, self.shape[3], self.shape[4])

    def __getitem__(self, key):
        spans, squeeze, steps = normalise_basic_index(key, self.shape)
        (t0, t1), (c0, c1), (z0, z1), (y0, y1), (x0, x1) = spans
        out = np.empty(tuple(stop - start for start, stop in spans), dtype=self.dtype)
        if out.size:
            # Rows per Bio-Formats call: a plane rectangle too large for the
            # heap (or for one Java array) is read as several row strips.
            row_bytes = max(1, (x1 - x0) * self.dtype.itemsize)
            rows = max(1, _java_read_budget_bytes() // row_bytes)
            reader = _get_cached_reader(self.path, self.series)
            for ti, t in enumerate(range(t0, t1)):
                for ci, c in enumerate(range(c0, c1)):
                    for zi, z in enumerate(range(z0, z1)):
                        for y in range(y0, y1, rows):
                            y_end = min(y + rows, y1)
                            with _java_read_slots:
                                plane = reader[y:y_end, x0:x1, z:z + 1, c:c + 1, t:t + 1]
                            # Written straight into the output: collecting planes
                            # and stacking them held every plane twice.
                            out[ti, ci, zi, y - y0:y_end - y0] = (
                                np.asarray(plane).reshape(y_end - y, x1 - x0))
        if any(step != 1 for step in steps):
            out = out[tuple(slice(None, None, step) for step in steps)]
        if squeeze:
            out = out.squeeze(axis=squeeze)
        return out


def _build_bioformats_array(path: str, series: int, shape, dtype) -> DynamicArray:
    """Lazy (T, C, Z, Y, X) array over a Bio-Formats image.

    *shape* and *dtype* come from the already-loaded OME-XML, so building the
    array opens no reader and scene loading stays fully lazy.
    """
    logger.info("Bio-Formats lazy read: shape=(T=%d,C=%d,Z=%d,Y=%d,X=%d) dtype=%s; "
                "each writer region is read as one rectangle", *shape, dtype)
    return from_array(_BioFormatsSource(path, series, shape, dtype))


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def abbreviate_units(measure: str) -> str:
    """Return the abbreviated form of a unit string, e.g. 'micrometer' → 'µm'."""
    if measure is None:
        return None
    abbreviations = {
        "millimeter": "mm", "centimeter": "cm", "decimeter": "dm",
        "meter": "m", "decameter": "dam", "hectometer": "hm",
        "kilometer": "km", "micrometer": "µm", "nanometer": "nm",
        "picometer": "pm",
        # angstrom: official symbol Å; Bio-Formats / older OME also use plain "A"
        "angstrom": "Å", "angström": "Å",
        "second": "s", "millisecond": "ms",
        "microsecond": "µs", "nanosecond": "ns", "minute": "min", "hour": "h",
    }
    if measure in abbreviations.values():
        return measure
    return abbreviations.get(measure.lower(), "Unknown")


def expand_units(measure: str) -> str:
    """Return the expanded form of an abbreviated unit, e.g. 'µm' → 'micrometer'."""
    if measure is None:
        return None
    expansions = {
        "mm": "millimeter", "cm": "centimeter", "dm": "decimeter",
        "m": "meter", "dam": "decameter", "hm": "hectometer",
        "km": "kilometer", "µm": "micrometer", "nm": "nanometer",
        "pm": "picometer",
        # angstrom variants: Å (U+00C5), å (U+00E5 lowercase), A (older OME abbreviation)
        "Å": "angstrom", "å": "angstrom", "A": "angstrom",
        "s": "second", "ms": "millisecond",
        "µs": "microsecond", "ns": "nanosecond", "min": "minute", "h": "hour",
    }
    if measure in expansions.values():
        return measure
    # Try exact match first (preserves case-sensitive symbols like Å), then lowercase fallback
    return expansions.get(measure, expansions.get(measure.lower(), "Unknown"))


def expand_hex_shorthand(hex_color: str) -> str:
    """Expand a shorthand hex colour, e.g. '#abc' → '#aabbcc'."""
    if not hex_color.startswith('#'):
        raise ValueError("Hex color must start with '#'")
    shorthand = hex_color[1:]
    if not all(c in '0123456789ABCDEFabcdef' for c in shorthand):
        raise ValueError("Invalid hex digits")
    return '#' + ''.join([c * 2 for c in shorthand])


def create_ome_xml(
        image_shape: tuple,
        axis_order: str,
        pixel_size_x: float = None,
        pixel_size_y: float = None,
        pixel_size_z: float = None,
        pixel_size_t: float = None,
        unit_x: str = "MICROMETER",
        unit_y: str = None,
        unit_z: str = None,
        unit_t: str = None,
        dtype: str = "uint8",
        image_name: str = "Default Image",
        channel_names: list = None,
) -> OME:
    """Build an OME metadata object from image shape and physical metadata."""
    fullaxes = 'xyczt'
    if len(axis_order) != len(image_shape):
        raise ValueError("Length of axis_order must match length of image_shape")
    axis_order = axis_order.upper()

    pixel_size_basemap = {
        'time_increment': pixel_size_t,
        'physical_size_z': pixel_size_z,
        'physical_size_y': pixel_size_y,
        'physical_size_x': pixel_size_x,
    }
    pixel_size_map = {}
    for ax in 'tzyx':
        if ax == 't':
            if ax in axis_order.lower():
                pixel_size_map['time_increment'] = pixel_size_t or 1
        else:
            if ax in axis_order.lower():
                pixel_size_map[f'physical_size_{ax}'] = pixel_size_basemap[f'physical_size_{ax}'] or 1

    unit_basemap = {
        'time_increment_unit': unit_t,
        'physical_size_z_unit': unit_z,
        'physical_size_y_unit': unit_y,
        'physical_size_x_unit': unit_x,
    }
    unit_map_ = {}
    for ax in 'tzyx':
        if ax == 't':
            if ax in axis_order.lower():
                unit_map_['time_increment_unit'] = unit_t or 'second'
        else:
            if ax in axis_order.lower():
                unit_map_[f'physical_size_{ax}_unit'] = unit_basemap[f'physical_size_{ax}_unit'] or 'MICROMETER'
    unit_map_ = {k: abbreviate_units(v) for k, v in unit_map_.items() if v is not None}

    dtype_map = {
        "uint8": PixelType.UINT8, "uint16": PixelType.UINT16,
        "uint32": PixelType.UINT32, "int8": PixelType.INT8,
        "int16": PixelType.INT16, "int32": PixelType.INT32,
        "float32": PixelType.FLOAT, "float64": PixelType.DOUBLE,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}")

    size_map_ = dict(zip(axis_order.lower(), image_shape))
    size_map = {f'size_{ax}': size_map_.get(ax, 1) for ax in fullaxes}

    if channel_names is None or len(channel_names) != size_map['size_c']:
        channels = [Channel(id=f"Channel:{i}", samples_per_pixel=1)
                    for i in range(size_map['size_c'])]
    else:
        channels = [Channel(id=f"Channel:{i}", samples_per_pixel=1, name=channel_names[i])
                    for i in range(size_map['size_c'])]

    pixels = Pixels(
        dimension_order=Pixels_DimensionOrder(fullaxes.upper()),
        **size_map, type=dtype_map[dtype],
        **pixel_size_map, **unit_map_, channels=channels,
    )
    return OME(images=[Image(id="Image:0", name=image_name, pixels=pixels)])


def parse_series(series: Union[Iterable, int]):
    if series is None:
        series = 0
    if np.isscalar(series):
        series = [series]
    return series


# ---------------------------------------------------------------------------
# ImageReader — abstract base for all format-specific loaders
# ---------------------------------------------------------------------------

class ImageReader(ABC):
    """Contract for format-specific image loaders.

    Concrete implementations: PFFImageMeta, TIFFImageMeta, H5ImageMeta,
    IMSImageMeta.  All I/O is async; metadata getters are synchronous once
    the dataset has been loaded via ``read_dataset()``.
    """

    # ── mandatory async I/O ──────────────────────────────────────────────

    @abstractmethod
    async def read_dataset(self) -> None:
        """Load metadata and array data concurrently."""

    @abstractmethod
    async def get_arraydata(self) -> Any:
        """Return the dask/zarr array for the current scene."""

    @abstractmethod
    async def get_pyramid(self, version: str = '0.4') -> Pyramid:
        """Return (or build) a Pyramid from the loaded data."""

    # ── optional scene / tile switching (default: no-op) ────────────────

    async def set_scene(self, scene_index: int) -> None:
        pass

    async def set_tile(self, tile_index: int) -> None:
        pass

    # ── synchronous metadata getters (available after read_dataset) ──────

    @abstractmethod
    def get_axes(self) -> str: ...

    @abstractmethod
    def get_scaledict(self) -> dict: ...

    @abstractmethod
    def get_unitdict(self) -> dict: ...

    @abstractmethod
    def get_channels(self) -> Optional[list]: ...

    def get_scales(self) -> list:
        sd = self.get_scaledict()
        return [sd[ax] for ax in self.get_axes() if ax != 'c' and ax in sd]

    def get_units(self) -> list:
        ud = self.get_unitdict()
        return [ud[ax] for ax in self.get_axes() if ax != 'c' and ax in ud]

    # ── scene / tile counts (default: single scene, single tile) ─────────

    @property
    def n_scenes(self) -> int:
        return 1

    @property
    def n_tiles(self) -> int:
        return 1

    # ── view / illumination support (default: 1 each, no-op setters) ─────
    # Override in format-specific subclasses that expose these dimensions.

    @property
    def n_views(self) -> int:
        return 1

    @property
    def n_illuminations(self) -> int:
        return 1

    def set_view(self, view_index: int) -> None:
        pass

    def set_illumination(self, illumination_index: int) -> None:
        pass


# ---------------------------------------------------------------------------
# PFFImageMeta — generic format reader (Bio-Formats / bioio)
# ---------------------------------------------------------------------------

class PFFImageMeta(ImageReader):
    essential_omexml_fields = {
        "physical_size_x", "physical_size_x_unit",
        "physical_size_y", "physical_size_y_unit",
        "physical_size_z", "physical_size_z_unit",
        "time_increment", "time_increment_unit",
        "size_x", "size_y", "size_z", "size_t", "size_c",
    }

    def __init__(self, path, meta_reader="bioio", aszarr=False,
                 reader_tile_size_mb: float = 256.0,
                 force_bioformats: bool = False,
                 as_mosaic: bool = False,
                 reader_kwargs: Optional[dict] = None):
        self.root = path
        self._series = 0
        self.omemeta = None
        self.pyr = None
        self._tile = 0
        self._aszarr = aszarr
        self.arraydata = None
        self.reader = None
        self._meta_reader = meta_reader
        self._as_mosaic = as_mosaic
        self._n_scenes = None
        self._n_tiles = None
        self._tile_mb = float(reader_tile_size_mb)
        self._force_bioformats = bool(force_bioformats)
        self._bfio_tiling = False   # set True when bfio tiled path is active
        self._reader_kwargs: dict = reader_kwargs or {}

    def __getstate__(self): return self.__dict__.copy()
    def __setstate__(self, state): self.__dict__.update(state)

    async def _micro_omemeta(self):
        """OME metadata from micro-reader (metadata_reader='micro'), or None
        when micro-reader does not take this file (then Bio-Formats is used,
        and the standard pixel reader with it)."""
        from eubi_bridge.core.micro_source import MicroReader, micro_omemeta
        kwargs = {k: v for k, v in self._reader_kwargs.items() if k != 'pixel_reader'}
        reader = await asyncio.to_thread(MicroReader.open, self.root,
                                         as_mosaic=self._as_mosaic, **kwargs)
        if reader is None:
            return None
        try:
            return await asyncio.to_thread(micro_omemeta, reader)
        except Exception as exc:                    # noqa: BLE001 - fall back, say why
            logger.warning(f"metadata_reader=micro: {self.root}: Bio-Formats used "
                           f"({type(exc).__name__}: {exc})")
            return None

    async def read_omemeta(self):
        #: metadata from micro-reader (metadata_reader='micro'): then micro-reader
        #: also reads the pixels -- one parse, so the two cannot disagree
        self._micro_meta = False
        if self._meta_reader == 'micro':
            omemeta = await self._micro_omemeta()
            if omemeta is not None:
                self._micro_meta = True
                self.omemeta = omemeta
                self._n_scenes = len(omemeta.images)
                return
            # micro-reader does not take this file: Bio-Formats metadata, and
            # the standard pixel reader with it -- if installed (not in lite)
            from eubi_bridge.utils.capabilities import (has_bioformats,
                                                        micro_reader_declined)
            if not has_bioformats():
                raise micro_reader_declined(self.root)
            self._reader_kwargs = {k: v for k, v in self._reader_kwargs.items()
                                   if k != 'pixel_reader'}
            logger.info(f"metadata_reader=micro: {self.root}: Bio-Formats metadata "
                        f"and the standard reader used")
        if self.root.lower().endswith(('ome', 'xml')):
            from ome_types import OME
            omemeta = OME().from_xml(self.root)
        elif self.root.lower().endswith('.czi') and self._meta_reader == 'czi_native':
            # Provisional native CZI metadata via pylibCZIrw (no JVM; enumerates
            # all scenes from the directory).  Opt-in only
            # (metadata_reader='czi_native'); the default CZI metadata path is
            # still Bio-Formats.  NB: pylibCZIrw reads the directory, so for a
            # corrupt/partial CZI it can report more scenes than are readable.
            from eubi_bridge.core.czi_reader import build_czi_omemeta
            omemeta = await asyncio.to_thread(build_czi_omemeta, self.root)
        else:
            # 'czi_native' on a non-CZI input, and 'micro' on a file micro-reader
            # does not take, fall back to the default reader.
            meta_reader = ('bfio' if self._meta_reader in ('czi_native', 'micro')
                           else self._meta_reader)
            if meta_reader == 'bioio':
                try:
                    omemeta = await read_metadata_via_extension(self.root, series=self._series)
                except Exception as e:
                    logger.debug(f"bioio extension reader failed for {self.root}: {e}. Falling back.")
                    omemeta = await read_metadata_via_bioio_bioformats(self.root, series=self._series)
            elif meta_reader == 'bfio':
                try:
                    omemeta = await read_metadata_via_bfio(self.root)
                except Exception as e:
                    logger.debug(f"bfio reader failed for {self.root}: {e}. Falling back.")
                    omemeta = await read_metadata_via_bioio_bioformats(self.root, series=self._series)
            else:
                raise ValueError(f"Unsupported metadata reader: {self._meta_reader}")
        self.omemeta = _collapse_resolution_series(omemeta, self.root)
        self._n_scenes = len(self.omemeta.images)

    async def get_arraydata(self):
        if getattr(self.reader, 'sample_layout', None) is not None:
            # Multi-sample (RGB) pixels: the reader returns every sample as a
            # channel, C x S (see sample_channels); get_pixels matches it.
            return self.reader.get_image_dask_data()
        pix = self.pixels
        dims = self.reader.img.dims
        if not hasattr(dims, 'S'):
            dask_data = self.reader.get_image_dask_data(dimensions_to_read='TCZYX')
        elif hasattr(dims, 'S') and not hasattr(dims, 'C'):
            logger.warning("'S' found but no 'C' — treating 'S' as channel dimension.")
            dask_data = self.reader.get_image_dask_data(dimensions_to_read='TSZYX')
        elif hasattr(dims, 'S') and hasattr(dims, 'C'):
            if dims.C != pix.size_c and dims.S == pix.size_c:
                logger.warning("'S' matches channel count better than 'C' — using 'S'.")
                dask_data = self.reader.get_image_dask_data(dimensions_to_read='TSZYX')
            else:
                dask_data = self.reader.get_image_dask_data(dimensions_to_read='TCZYX')
        else:
            dask_data = self.reader.get_image_dask_data(dimensions_to_read='TCZYX')
        return dask_data

    async def set_scene(self, scene_index: int) -> None:
        self._series = scene_index
        if self.reader is not None:
            self.reader.set_scene(self._series)
            if not self._bfio_tiling:
                self.arraydata = await self.get_arraydata()
        elif self._bfio_tiling:
            # force_bioformats or bfio fallback: rebuild dask graph for new series.
            # Use OME-XML shape to avoid a blocking BioReader probe.
            shape, dtype = self._bfio_shape_from_meta()
            self.arraydata = _build_bioformats_array(
                self.root, self._series, shape, dtype)

    async def set_scene_isolated(self, scene_index: int) -> None:
        """Switch scenes using a FRESH underlying reader pinned to this scene.

        Native bioio readers (CZI/ND2/LIF…) resolve the scene index *lazily*, at
        dask compute time, from a single mutable reader (``_current_scene_index``).
        ``SceneLoader`` snapshots every scene's lazy array up front and only
        computes them later (at write time), so reusing one reader across scenes
        makes every snapshot read whichever scene the reader was last left on —
        silently corrupting all but the last scene, and crashing outright when the
        per-scene stitched geometry differs.  Giving each scene its own reader
        instance that is never mutated again keeps the snapshots independent.

        Single-scene formats and the Bio-Formats tiled path already bake the
        series index into their graphs, so they fall back to the plain
        ``set_scene``.
        """
        if (self.n_scenes <= 1 or self._bfio_tiling
                or self._force_bioformats or self.reader is None):
            await self.set_scene(scene_index)
            return
        # Native multi-scene bioio path: rebuild a fresh reader for this scene so
        # its lazy dask graph closes over an immutable scene index.
        self._series = scene_index
        self.reader = await read_single_image(
            self.root, aszarr=self._aszarr, as_mosaic=self._as_mosaic,
            **self._reader_kwargs)
        await self.set_scene(scene_index)
        await self.set_tile(self._tile)
        self.arraydata = await self.get_arraydata()

    async def set_tile(self, tile_index: int) -> None:
        self._tile = 0
        if self.reader is not None and hasattr(self.reader, 'set_tile') and self.reader.n_tiles > 1:
            self._tile = tile_index
            self.reader.set_tile(self._tile)
            if not self._bfio_tiling:
                self.arraydata = await self.get_arraydata()

    def set_view(self, view_index: int) -> None:
        if self.reader is not None and hasattr(self.reader, 'set_view'):
            self.reader.set_view(view_index)
            if not self._bfio_tiling:
                # arraydata will be refreshed by the next _snapshot call
                pass

    def set_illumination(self, illumination_index: int) -> None:
        if self.reader is not None and hasattr(self.reader, 'set_illumination'):
            self.reader.set_illumination(illumination_index)

    @property
    def n_views(self) -> int:
        if self.reader is not None and hasattr(self.reader, 'n_views'):
            return self.reader.n_views
        return 1

    @property
    def n_illuminations(self) -> int:
        if self.reader is not None and hasattr(self.reader, 'n_illuminations'):
            return self.reader.n_illuminations
        return 1

    @property
    def n_tiles(self) -> int:
        if self.reader is not None and hasattr(self.reader, 'n_tiles'):
            try:
                return self.reader.n_tiles
            except Exception as e:
                logger.warning(f"Failed to get n_tiles: {e}. Using cached value.")
        return self._n_tiles or 1

    @property
    def n_scenes(self) -> int:
        if self.reader is None or self.root.lower().endswith('.lsm'):
            # OME-XML (read_omemeta) is the single source of truth for series count.
            return self._n_scenes or 1
        return self.reader.n_scenes

    def get_pixels(self):
        if self.omemeta is None:
            raise ValueError(f"OME metadata not loaded for {self.root}")
        if self._series >= len(self.omemeta.images):
            raise ValueError(f"Series {self._series} out of range for {self.root}")
        try:
            pixels = self.omemeta.images[self._series].pixels
            missing = self.essential_omexml_fields - pixels.model_fields_set
            pixels.model_fields_set.update(missing)
        except (AttributeError, IndexError) as e:
            raise ValueError(f"Failed to get pixels from {self.root} series {self._series}: {e}") from e
        layout = getattr(self.reader, 'sample_layout', None)
        if layout is not None and not getattr(self, '_bfio_tiling', False):
            # The array holds every sample as a channel (C x S); make the
            # channel list say so.  Idempotent.
            from eubi_bridge.core.sample_channels import expand_channel_metadata
            expand_channel_metadata(pixels, *layout)
        return pixels

    @property
    def pixels(self):
        return self.get_pixels()

    def _bfio_shape_from_meta(self):
        """Return (shape, dtype) from already-loaded OME-XML — no BioReader open."""
        pix = self.pixels
        shape = (pix.size_t, pix.size_c, pix.size_z, pix.size_y, pix.size_x)
        dtype = _ome_dtype(pix.type)
        return shape, dtype

    async def read_img(self, **kwargs):
        if self._force_bioformats:
            # User explicitly wants bfio tiled path — skip bioio entirely.
            self._bfio_tiling = True
            shape, dtype = self._bfio_shape_from_meta()
            self.arraydata = _build_bioformats_array(
                self.root, self._series, shape, dtype)
            return

        self.reader = await read_single_image(self.root, aszarr=self._aszarr, as_mosaic=self._as_mosaic, **{**self._reader_kwargs, **kwargs})
        await self._check_micro_reader(**kwargs)

        if _is_bioformats_backed(self.reader):
            # A format without a native reader: read it exactly as force_bioformats
            # does -- bfio pixels per series, the scene count from the (collapsed)
            # OME metadata. bioio's Bio-Formats reader is dropped: kept, it made every
            # scene read series 0's pixels (set_scene does not rebuild the array while
            # a reader is set; eubi-bridge 0.1.2-0.1.5), and its scene count included
            # resolution levels (a false "PARTIAL READ - DAMAGED FILE" warning).
            self.reader = None
            self._bfio_tiling = True
            shape, dtype = self._bfio_shape_from_meta()
            self.arraydata = _build_bioformats_array(
                self.root, self._series, shape, dtype)
        else:
            # Native bioio plugin (CZI, ND2, LIF, TIFF…) — unchanged path.
            self._bfio_tiling = False
            await self.set_scene(self._series)
            await self.set_tile(self._tile)
            self.arraydata = await self.get_arraydata()

    def _micro_stored_pyramid(self, version: str):
        """The file's own resolution levels as a Pyramid, when micro-reader is
        the reader and the file stores more than one (keep_existing_resolutions;
        as IMSImageMeta does for Imaris); else None.  Each level's pixel size
        is level 0's scaled by the shape ratio."""
        from eubi_bridge.core.micro_source import MicroReader
        reader = self.reader
        if not isinstance(reader, MicroReader) or reader.n_resolution_levels <= 1:
            return None
        arrays = [reader.get_resolution_level_dask_data(r)
                  for r in range(reader.n_resolution_levels)]
        base_shape = arrays[0].shape
        pix = self.pixels
        # a size the metadata lacks counts as 1.0 (eubi-bridge's default scale)
        base_scale = [1.0 if v is None else float(v) for v in (
            pix.time_increment, 1.0, pix.physical_size_z,
            pix.physical_size_y, pix.physical_size_x)]
        scales = [[s * (b / a) for s, b, a in zip(base_scale, base_shape, arr.shape)]
                  for arr in arrays]
        logger.info(f"pixel_reader=micro: {self.root}: keeping the file's "
                    f"{len(arrays)} resolution levels")
        return Pyramid().from_arrays(
            arrays=arrays, axis_order='tczyx',
            unit_list=self.get_units(), scales=scales,
            version=version, name="Series_0",
        )

    async def _check_micro_reader(self, **kwargs) -> None:
        """With micro-reader as the reader (pixel_reader='micro'), check it
        against the metadata -- no pixels read, no standard reader opened --
        and use the standard reader for this file if anything disagrees."""
        from eubi_bridge.core.micro_source import MicroReader, metadata_mismatch
        if not isinstance(self.reader, MicroReader):
            return
        if getattr(self, '_micro_meta', False):     # metadata from the same parse
            self.reader.set_scene(min(self._series, self.reader.n_scenes - 1))
            return
        try:
            problem = metadata_mismatch(self.reader, self.omemeta)
        except Exception as exc:                    # noqa: BLE001 - fall back, say why
            problem = f"check failed: {type(exc).__name__}: {exc}"
        if not problem:
            self.reader.set_scene(min(self._series, self.reader.n_scenes - 1))
            return
        logger.warning(f"pixel_reader=micro: {self.root}: standard reader used ({problem})")
        self._reader_kwargs = {k: v for k, v in self._reader_kwargs.items()
                               if k != 'pixel_reader'}
        self.reader = await read_single_image(
            self.root, aszarr=self._aszarr, as_mosaic=self._as_mosaic,
            **{**self._reader_kwargs, **kwargs})

    async def get_pyramid(self, version='0.4') -> Pyramid:
        if getattr(self, '_keep_existing_resolutions', False):
            stored = self._micro_stored_pyramid(version)
            if stored is not None:
                return stored
        # Use the already-built tiled array when read_img() set it (bioformats path).
        # Calling get_arraydata() again would re-issue a full-plane openBytes and
        # hit the 2 GB limit for large files.
        array = self.arraydata if self.arraydata is not None else await self.get_arraydata()
        return Pyramid().from_array(
            array=array, axis_order=self.get_axes(),
            unit_list=self.get_units(), scale=self.get_scales(),
            version=version, name="Series_0",
        )

    async def read_dataset(self) -> None:
        # Metadata must be loaded before the image reader, because get_arraydata
        # reconciles dims against the OME pixels (self.pixels).  Run them in
        # order rather than concurrently to avoid a race where get_arraydata
        # reads omemeta before it is populated.
        await self.read_omemeta()
        await self.read_img()
        await self.set_scene(self._series)
        await self.set_tile(self._tile)

    def get_axes(self) -> str:
        return 'tczyx'

    def get_scaledict(self) -> dict:
        return {
            't': self.pixels.time_increment,
            'z': self.pixels.physical_size_z,
            'y': self.pixels.physical_size_y,
            'x': self.pixels.physical_size_x,
        }

    def get_unitdict(self) -> dict:
        return {
            't': self.pixels.time_increment_unit.name.lower(),
            'z': self.pixels.physical_size_z_unit.name.lower(),
            'y': self.pixels.physical_size_y_unit.name.lower(),
            'x': self.pixels.physical_size_x_unit.name.lower(),
        }

    def get_channels(self) -> Optional[list]:
        pixels = self.get_pixels()
        if not hasattr(pixels, 'channels') or len(pixels.channels) == 0:
            return None
        if len(pixels.channels) < pixels.size_c:
            return ChannelIterator(num_channels=pixels.size_c)._channels
        return [
            dict(label=ch.name, color=expand_hex_shorthand(ch.color.as_hex().upper()))
            for ch in pixels.channels
        ]

    def snapshot(self) -> dict:
        return {
            k: copy.deepcopy(v)
            for k, v in self.__dict__.items()
            if k not in ("reader", "omemeta", "pyr")
        }

    def restore(self, snapshot: dict) -> None:
        for k, v in snapshot.items():
            setattr(self, k, copy.deepcopy(v))


# ---------------------------------------------------------------------------
# TIFFImageMeta
# ---------------------------------------------------------------------------

class TIFFImageMeta(PFFImageMeta):
    essential_omexml_fields = PFFImageMeta.essential_omexml_fields

    def __init__(self, path, meta_reader="bioio", aszarr=False):
        if not path.endswith(('.tif', '.tiff', '.lsm')):
            raise ValueError(f"Not a TIFF file: {path}")
        super().__init__(path, meta_reader, aszarr)

    async def read_omemeta(self):
        import tifffile
        tif = tifffile.TiffFile(self.root)
        self.tiffzarrstore = tif.aszarr()
        self._zarrmeta = self.tiffzarrstore._data[self._series]
        self._meta = tif.series[self._series]
        await super().read_omemeta()

    def get_axes(self) -> str:
        if self._aszarr:
            return 'tczyx'
        axes = self._meta.axes.lower()
        default_axes_cut = default_axes[-len(axes):]
        newaxes = []
        for i, ax in enumerate(axes):
            if ax == 's':
                ax = 'c'
            newaxes.append(ax if ax in default_axes else default_axes_cut[i])
        return ''.join(newaxes)

    def get_scaledict(self) -> dict:
        if self._aszarr:
            parent_sd = super().get_scaledict()
            native_axes = self._meta.axes.lower()
            return {
                ax: parent_sd.get(ax, scale_map.get(ax, 1.0)) if ax in native_axes
                else scale_map.get(ax, 1.0)
                for ax in 'tczyx'
            }
        sd = super().get_scaledict()
        return {ax: sd[ax] for ax in self.get_axes() if ax in sd}

    def get_scales(self) -> list:
        sd = self.get_scaledict()
        return [sd[ax] for ax in self.get_axes() if ax != 'c']

    def get_unitdict(self) -> dict:
        if self._aszarr:
            parent_ud = super().get_unitdict()
            native_axes = self._meta.axes.lower()
            return {
                ax: parent_ud.get(ax, unit_map.get(ax)) if ax in native_axes
                else unit_map.get(ax)
                for ax in 'tczyx'
            }
        ud = super().get_unitdict()
        return {ax: ud[ax] for ax in self.get_axes() if ax != 'c' and ax in ud}


# ---------------------------------------------------------------------------
# H5ImageMeta
# ---------------------------------------------------------------------------

class H5ImageMeta(PFFImageMeta):
    essential_omexml_fields = PFFImageMeta.essential_omexml_fields

    def __init__(self, path, meta_reader="bioio", **kwargs):
        if not path.lower().endswith('.h5'):
            raise ValueError(f"Not an HDF5 file: {path}")
        super().__init__(path, meta_reader, aszarr=False, **kwargs)

    async def read_omemeta(self, **kwargs):
        import h5py
        f = h5py.File(self.root)
        dset_name = list(f.keys())[self._series]
        ds = f[dset_name]
        self._ds = ds
        self._attrs = dict(ds.attrs)
        self._n_scenes = len(list(f.keys()))
        # A plain HDF5 dataset carries no OME-XML, but the inherited pixel
        # accessors (get_pixels -> get_arraydata) require one.  Synthesise it
        # from the dataset's own shape/dtype and its axistags, mirroring what
        # IMSImageMeta does for Imaris.  Without this self.omemeta stays None and
        # every .h5 conversion dies with "OME metadata not loaded".
        self.omemeta = self._build_omemeta(ds)

    def _build_omemeta(self, ds) -> OME:
        """Synthetic single-``<Image>`` OME describing one HDF5 dataset."""
        axes = self.get_axes()
        sizes = dict(zip(axes, ds.shape))
        scales = self.get_scaledict()

        pixels = Pixels(
            dimension_order=Pixels_DimensionOrder.XYZCT,
            type=_ome_pixeltype_from_dtype(ds.dtype),
            size_x=int(sizes.get('x', 1)),
            size_y=int(sizes.get('y', 1)),
            size_z=int(sizes.get('z', 1)),
            size_c=int(sizes.get('c', 1)),
            size_t=int(sizes.get('t', 1)),
            physical_size_x=float(scales.get('x', 1.0) or 1.0),
            physical_size_x_unit=UnitsLength.MICROMETER,
            physical_size_y=float(scales.get('y', 1.0) or 1.0),
            physical_size_y_unit=UnitsLength.MICROMETER,
            physical_size_z=float(scales.get('z', 1.0) or 1.0),
            physical_size_z_unit=UnitsLength.MICROMETER,
            time_increment=float(scales.get('t', 1.0) or 1.0),
            time_increment_unit=UnitsTime.SECOND,
            channels=[Channel(id=f"Channel:{c}", name=f"Channel {c}",
                              samples_per_pixel=1)
                      for c in range(int(sizes.get('c', 1)))],
        )
        return OME(images=[Image(id="Image:0", name="Series_0", pixels=pixels)])

    async def read_img(self):
        from eubi_bridge.core.h5_reader import read_h5
        self.reader = await asyncio.to_thread(read_h5, self.root)
        self.reader.set_scene(self._series)
        self.arraydata = await self.get_arraydata()

    async def get_arraydata(self):
        # H5Reader yields the dataset in its own axis order; the inherited
        # implementation assumes a bioio reader with a .img.dims attribute.
        return self.reader.get_image_dask_data()

    def _parse_axistags(self) -> dict:
        """Return the axistags dict from h5py attrs, handling str and non-dict forms."""
        raw = self._attrs.get('axistags', {})
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, dict):
            return {}
        return raw

    def get_axes(self) -> str:
        axistags = self._parse_axistags()
        axlist = axistags.get('axes', [])
        axes = []
        for i, ax in enumerate(axlist):
            key = ax.get('key', '')
            axes.append(key if key in 'tczyx' else default_axes[i])
        return ''.join(axes)

    def get_scaledict(self) -> dict:
        axistags = self._parse_axistags()
        sd = {}
        for ax in axistags.get('axes', []):
            key = ax.get('key', '')
            if key == 'c' or key not in self.get_axes():
                continue
            sd[key] = ax.get('scale', ax.get('resolution', scale_map.get(key, 1.0)))
        return sd

    def get_unitdict(self) -> dict:
        axistags = self._parse_axistags()
        ud = {}
        for ax in axistags.get('axes', []):
            key = ax.get('key', '')
            if key == 'c' or key not in self.get_axes():
                continue
            ud[key] = ax.get('unit', unit_map.get(key))
        return ud

    def get_channels(self) -> list:
        return []


# ---------------------------------------------------------------------------
# IMSImageMeta
# ---------------------------------------------------------------------------

class IMSImageMeta(PFFImageMeta):
    """Native HDF5-based reader for Imaris (.ims) files.

    Builds a synthetic single-``<Image>`` OME object so the inherited
    ``get_pixels()``/``get_scaledict()``/``get_unitdict()``/``get_channels()``/
    ``_bfio_shape_from_meta()`` work unchanged. Only ``read_img()`` /
    ``get_arraydata()`` (and ``get_pyramid()`` for ``keep_existing_resolutions``)
    need overriding since the native ``IMSReader`` has no ``.img.dims`` like
    bioio readers do.

    When ``force_bioformats=True``, or when the native HDF5 parse fails, falls
    back to the generic ``PFFImageMeta`` (Bio-Formats) path — Track B's
    resolution-pyramid collapse applies there.
    """
    essential_omexml_fields = PFFImageMeta.essential_omexml_fields

    def __init__(self, path, meta_reader="bioio", **kwargs):
        if not path.lower().endswith('.ims'):
            raise ValueError(f"Not an Imaris file: {path}")
        self._keep_existing_resolutions = bool(kwargs.pop('keep_existing_resolutions', False))
        super().__init__(path, meta_reader, aszarr=False, **kwargs)

    async def read_omemeta(self):
        if self._force_bioformats:
            await super().read_omemeta()   # Track B applies here
            return
        if self._meta_reader == 'micro':
            omemeta = await self._micro_omemeta()
            if omemeta is not None:
                self._micro_meta, self._native_ims = True, True
                self.omemeta, self._n_scenes = omemeta, 1
                return
            self._reader_kwargs = {k: v for k, v in self._reader_kwargs.items()
                                   if k != 'pixel_reader'}
        self._micro_meta = False
        try:
            self.omemeta = await asyncio.to_thread(_build_ims_omemeta, self.root)
            self._n_scenes = 1
            self._native_ims = True
        except Exception as e:
            logger.warning(f"Native IMS reader failed for {self.root} ({e}); "
                            f"falling back to Bio-Formats.")
            self._native_ims = False
            await super().read_omemeta()

    async def read_img(self):
        if self._force_bioformats or not getattr(self, '_native_ims', True):
            await super().read_img()
            return
        if self._reader_kwargs.get('pixel_reader') == 'micro':
            reader = await asyncio.to_thread(self._micro_reader)
            if reader is not None:
                self.reader = reader
                self.arraydata = await self.get_arraydata()
                return
        from eubi_bridge.core.ims_reader import read_ims
        self.reader = await asyncio.to_thread(read_ims, self.root)
        self.arraydata = await self.get_arraydata()

    def _micro_reader(self):
        """micro-reader as the reader (pixel_reader='micro'), checked against
        the metadata built from the file; None to use eubi-bridge's own reader.
        It offers what this class uses of a reader: get_image_dask_data,
        n_resolution_levels, get_resolution_level_dask_data."""
        from eubi_bridge.core.micro_source import MicroReader, metadata_mismatch
        reader = MicroReader.open(self.root)
        if reader is None:
            return None
        if getattr(self, '_micro_meta', False):     # metadata from the same parse
            reader.set_scene(0)
            return reader
        try:
            problem = metadata_mismatch(reader, self.omemeta)
        except Exception as exc:                    # noqa: BLE001 - fall back, say why
            problem = f"check failed: {type(exc).__name__}: {exc}"
        if problem:
            logger.warning(f"pixel_reader=micro: {self.root}: eubi-bridge's Imaris reader "
                           f"used ({problem})")
            return None
        reader.set_scene(0)
        return reader

    async def get_arraydata(self):
        if self._force_bioformats or not getattr(self, '_native_ims', True):
            return await super().get_arraydata()
        return self.reader.get_image_dask_data()

    async def get_pyramid(self, version='0.4') -> Pyramid:
        if (self._force_bioformats or not getattr(self, '_native_ims', True)
                or not self._keep_existing_resolutions):
            return await super().get_pyramid(version)

        if self.reader.n_resolution_levels <= 1:
            return await super().get_pyramid(version)

        arrays = [self.reader.get_resolution_level_dask_data(r)
                  for r in range(self.reader.n_resolution_levels)]
        base_shape = arrays[0].shape
        base_scale = [
            self.pixels.time_increment, 1.0,
            self.pixels.physical_size_z,
            self.pixels.physical_size_y,
            self.pixels.physical_size_x,
        ]
        scales = [
            [s * (b / a) for s, b, a in zip(base_scale, base_shape, arr.shape)]
            for arr in arrays
        ]
        return Pyramid().from_arrays(
            arrays=arrays, axis_order='tczyx',
            unit_list=self.get_units(), scales=scales,
            version=version, name="Series_0",
        )


# ---------------------------------------------------------------------------
# SceneLoader — picks the reader for a file and opens it
# ---------------------------------------------------------------------------

class SceneLoader:
    """Opens an image file with the reader that takes it (``_open``): ``_img``, its
    scene / tile / view / illumination counts. ``conversion/legacy_image.py`` reads the
    images from it."""

    def __init__(self, path: str, metadata_reader: str = 'bfio', skip_dask: bool = False,
                 reader_tile_size_mb: float = 256.0, force_bioformats: bool = False,
                 as_mosaic: bool = False,
                 reader_kwargs: Optional[dict] = None,
                 keep_existing_resolutions: bool = False,
                 pixel_reader: str = 'standard'):
        self.path = path
        #: 'standard', or 'micro': pixels from micro-reader (core/micro_source.py)
        self._pixel_reader = pixel_reader
        self._meta_reader = metadata_reader
        self._skip_dask = skip_dask
        self._tile_mb = float(reader_tile_size_mb)
        self._force_bioformats = bool(force_bioformats)
        self._as_mosaic = bool(as_mosaic)
        self._reader_kwargs: dict = reader_kwargs or {}
        self._keep_existing_resolutions = bool(keep_existing_resolutions)
        self._img: Optional[ImageReader] = None
        self.n_scenes: int = 0
        self.n_scenes_in_file: int = 0
        self.n_tiles: int = 0
        self.n_views: int = 1
        self.n_illuminations: int = 1
        self.may_have_views: bool = True      # set by _open

    async def _open(self) -> None:
        """Detect image type, open reader, populate n_scenes / n_tiles."""
        path = self.path
        if await asyncio.to_thread(is_zarr_group, path):
            raise ValueError(f"{path}: an OME-Zarr input is read by ome_zarr_pyramid "
                             f"(conversion/legacy_image.py: NgffFile), not the older readers")
        if path.lower().endswith('.h5'):
            self._img = await asyncio.to_thread(H5ImageMeta, path, self._meta_reader)
        elif (path.lower().endswith('.ims')
              and (self._meta_reader != 'micro' or self._force_bioformats)):
            # eubi's own Imaris reader; with micro-reader's metadata (the
            # default) an .ims goes the micro-reader way below, its stored
            # pyramid kept by keep_existing_resolutions too
            self._img = await asyncio.to_thread(
                IMSImageMeta, path, self._meta_reader,
                reader_tile_size_mb=self._tile_mb,
                force_bioformats=self._force_bioformats,
                reader_kwargs=self._pff_reader_kwargs(),
                keep_existing_resolutions=self._keep_existing_resolutions,
            )
        elif not self._skip_dask:
            self._img = await asyncio.to_thread(
                PFFImageMeta, path, self._meta_reader, self._skip_dask,
                self._tile_mb, self._force_bioformats, self._as_mosaic,
                self._pff_reader_kwargs(),
            )
            # micro-reader keeps the file's own pyramid (keep_existing_resolutions)
            self._img._keep_existing_resolutions = self._keep_existing_resolutions
        else:
            if path.endswith(('.tif', '.tiff')):
                self._img = await asyncio.to_thread(TIFFImageMeta, path,
                                                     self._meta_reader, self._skip_dask)
            else:
                logger.warning(
                    f"skip_dask is only supported for TIFF files; ignored for {path}")
                self._img = await asyncio.to_thread(
                    PFFImageMeta, path, self._meta_reader, self._skip_dask,
                    self._tile_mb, self._force_bioformats, self._as_mosaic,
                    self._pff_reader_kwargs(),
                )
        await self._img.read_dataset()
        self.n_scenes        = self._img.n_scenes
        self.n_tiles         = self._img.n_tiles
        self.n_views         = getattr(self._img, 'n_views', 1)
        self.n_illuminations = getattr(self._img, 'n_illuminations', 1)
        #: whether the file can have views / illuminations at all: only the CZI
        #: reader has them (every reader inherits set_view, so ask the type),
        #: or micro-reader reading a CZI
        reader = getattr(self._img, 'reader', None)
        self.may_have_views = (isinstance(reader, CZIReader)
                               or bool(getattr(reader, 'may_have_views', False)))
        # The number of scenes the file's index advertises (before any
        # readability cap) — used to report partial conversions accurately.
        self.n_scenes_in_file = self.n_scenes

        # Graceful degradation for truncated / corrupt files.  A reader may
        # advertise more scenes (from the on-disk index/directory) than are
        # actually readable: the metadata reader probes the data and reports only
        # the scenes it could resolve.  A smaller metadata scene count than the
        # reader's directory count therefore signals a damaged file (e.g. a CZI
        # whose subblock directory is inconsistent past some scene — confirmed in
        # Fiji/Bio-Formats too).  We can only convert scenes whose pixel metadata
        # exists, so cap to the readable count and warn loudly.
        meta_n = getattr(self._img, '_n_scenes', None)
        if isinstance(meta_n, int) and 0 < meta_n < self.n_scenes:
            unreadable = self.n_scenes - meta_n
            logger.warning(
                "\n" + "=" * 78 + "\n"
                f"PARTIAL READ — DAMAGED FILE: {path}\n"
                f"Only {meta_n} of {self.n_scenes} scenes can be read. The file's "
                f"index advertises {self.n_scenes} scenes, but its data is "
                f"inconsistent/unreadable past scene {meta_n} (a truncated or "
                f"corrupt acquisition — Fiji/Bio-Formats and the native readers all "
                f"stop at the same point).\n"
                f"Converting the {meta_n} readable scene(s). The remaining "
                f"{unreadable} image(s) DO exist in the file but COULD NOT BE READ "
                f"and will be missing from the output. If you need them, re-export "
                f"or repair the source file (e.g. 'Save As' in Zeiss ZEN).\n"
                + "=" * 78
            )
            self.n_scenes = meta_n

    def _pff_reader_kwargs(self) -> dict:
        """Reader options for PFFImageMeta: with pixel_reader='micro' micro-reader
        is the reader."""
        if self._meta_reader == 'micro' or self._pixel_reader == 'micro':
            # metadata_reader='micro' reads the pixels with micro-reader too
            return {**self._reader_kwargs, 'pixel_reader': 'micro'}
        return self._reader_kwargs


def czi_mosaic_tile_origins(path: str, scene: int = 0) -> dict:
    """Map mosaic tile index -> {'y': px, 'x': px} origin of the tiles of *scene*, or
    {} if unavailable.

    CZI stores each tile's position on the stage; splitting a mosaic into one
    container per tile discards it unless it is carried over explicitly.  Origins
    are returned in pixels and shifted so the minimum over the WHOLE file is zero
    (every scene in one coordinate system), because NGFF translations are offsets
    within a shared coordinate system and CZI centres its mosaic on zero (giving
    negative coordinates).

    Tiles are numbered per scene: until 0.1.5 the boxes were keyed by tile number
    alone, so scenes after the first took another scene's tile positions.

    Only CZI files have them; aicspylibczi (the full eubi-bridge) reads them.
    Files micro-reader reads carry each tile's position already.
    """
    if not str(path).lower().endswith('.czi'):
        return {}
    try:
        from aicspylibczi import CziFile
        czi = CziFile(path)
        if not czi.is_mosaic():
            return {}
        boxes = czi.get_all_mosaic_tile_bounding_boxes()
    except Exception:
        return {}

    origins: dict = {}                       # (scene, tile) -> origin
    for key, box in boxes.items():
        m = getattr(key, 'm_index', None)
        if m is None:
            continue
        s = (getattr(key, 'dimension_coordinates', None) or {}).get('S', 0)
        origins.setdefault((int(s), int(m)), {'y': float(box.y), 'x': float(box.x)})
    if not origins:
        return {}
    min_y = min(o['y'] for o in origins.values())
    min_x = min(o['x'] for o in origins.values())
    return {m: {'y': o['y'] - min_y, 'x': o['x'] - min_x}
            for (s, m), o in origins.items() if s == int(scene)}


# ---------------------------------------------------------------------------
# ChannelIterator
# ---------------------------------------------------------------------------

class ChannelIterator:
    """Generate and cycle through channel colour entries."""

    DEFAULT_COLORS = [
        "FF0000", "00FF00", "0000FF", "FF00FF", "00FFFF", "FFFF00", "FFFFFF",
    ]

    def __init__(self, num_channels: int = 0):
        self._channels: List[dict] = []
        self._current_index = 0
        self._generate_channels(num_channels)

    def _generate_channels(self, count: int) -> None:
        for i in range(len(self._channels), count):
            if i < len(self.DEFAULT_COLORS):
                color = self.DEFAULT_COLORS[i]
            else:
                hue = int((i * 137.5) % 360)
                r, g, b = self._hsv_to_rgb(hue / 360.0, 1.0, 1.0)
                color = f"{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"
            self._channels.append({"label": f"Channel {i + 1}", "color": color})

    @staticmethod
    def _hsv_to_rgb(h: float, s: float, v: float) -> tuple:
        h *= 6.0
        i = int(h)
        f, p, q, t = h - i, v*(1-s), v*(1-s*( h-i)), v*(1-s*(1-(h-i)))
        return [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)][i % 6]

    def __iter__(self):
        self._current_index = 0
        return self

    def __next__(self) -> dict:
        if self._current_index >= len(self._channels):
            self._generate_channels(len(self._channels) + 1)
        result = self._channels[self._current_index]
        self._current_index += 1
        return result

    def get_channel(self, index: int) -> dict:
        if index >= len(self._channels):
            self._generate_channels(index + 1)
        return self._channels[index]

    def __len__(self) -> int:
        return len(self._channels)

