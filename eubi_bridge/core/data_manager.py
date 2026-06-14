"""Image data management for EuBI-Bridge.

Three-layer design
------------------
1. ``ImageReader`` (ABC) — format-specific loaders (PFFImageMeta, TIFFImageMeta,
   H5ImageMeta, NGFFImageMeta).  All async I/O, no array-state knowledge.

2. ``ArrayState`` — pure data container.  Holds the array, axes, scales, units,
   channels.  Owns all array transformations (squeeze, transpose, crop) and
   channel helpers.  No I/O, no async.

3. ``ArrayManager`` — coordinator.  Picks the right ``ImageReader`` in ``init()``,
   populates ``ArrayState`` after loading, orchestrates scene/tile switching, and
   owns pyramid/OME-XML I/O methods (still async but clearly separated from the
   pure-data layer).

``conversion_worker.py`` accesses ``ArrayManager`` through its public interface
which is unchanged: ``manager.array``, ``manager.axes``, ``manager.scaledict``,
``manager.fill_default_meta()``, etc. all forward to ``ArrayState`` transparently.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import dask
import dask.delayed
import natsort
import numpy as np
import psutil
import zarr
from dask import array as da
from ome_types.model import (OME, Channel, Image, Pixels,
                             Pixels_DimensionOrder, PixelType, UnitsLength,
                             UnitsTime)

from eubi_bridge.core.czi_reader import CZIReader
from eubi_bridge.core.readers import (
    read_metadata_via_bfio, read_metadata_via_bioio_bioformats,
    read_metadata_via_extension, read_single_image)
from eubi_bridge.external.dyna_zarr.dynamic_array import DynamicArray
from eubi_bridge.external.dyna_zarr import operations as ops
from eubi_bridge.ngff.defaults import default_axes, scale_map, unit_map
from eubi_bridge.ngff.multiscales import Pyramid
from eubi_bridge.utils.array_utils import autocompute_chunk_shape, get_array_chunks
from eubi_bridge.utils.logging_config import get_logger
from eubi_bridge.utils.path_utils import (is_zarr_array, is_zarr_group,
                                          sensitive_glob, take_filepaths)

logger = get_logger(__name__)


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


def _compute_tile_shape(
    T: int, C: int, Z: int, Y: int, X: int, dtype: np.dtype, tile_mb: float,
) -> tuple[int, int, int, int, int]:
    """Derive a (t, c, z, y, x) tile shape using a priority-ordered budget.

    Budget is allocated in order X → Y → Z → C → T:
    - Fill the XY plane first (whole plane if it fits, otherwise square tile).
    - Any remaining budget is spent on additional Z planes.
    - Any remaining budget is spent on additional C channels.
    - Any remaining budget is spent on additional T frames.

    Maximising each dimension reduces the number of BioReader opens per series,
    which at 7-10 s each dominates total conversion time.
    """
    tile_bytes = tile_mb * 1024 ** 2
    itemsize = np.dtype(dtype).itemsize
    remaining = tile_bytes / itemsize   # budget in pixels

    # Priority 1: X and Y
    if Y * X <= remaining:
        y_tile, x_tile = Y, X
    else:
        yx_side = max(1, int(round(remaining ** 0.5)))
        y_tile = min(yx_side, Y)
        x_tile = min(yx_side, X)
    remaining /= y_tile * x_tile

    # Priority 2: Z
    z_tile = max(1, min(Z, int(remaining)))
    remaining /= z_tile

    # Priority 3: C
    c_tile = max(1, min(C, int(remaining)))
    remaining /= c_tile

    # Priority 4: T
    t_tile = max(1, min(T, int(remaining)))

    return t_tile, c_tile, z_tile, y_tile, x_tile


def _read_bfio_tile(
    path: str,
    y0: int, y1: int,
    x0: int, x1: int,
    z0: int, z1: int,
    c0: int, c1: int,
    t0: int, t1: int,
    series: int = 0,
) -> np.ndarray:
    """Read a (T_tile, C_tile, Z_tile, Y, X) block via bfio.

    One BioReader open covers all (t, c, z) combinations in the tile, amortising
    Java/Bio-Formats init overhead.  Returns shape ``(t1-t0, c1-c0, z1-z0, y1-y0, x1-x0)``.
    """
    # Reuse a cached reader for this thread — avoids the 7-10 s BioReader open
    # cost on every task.  Thread-local storage means no locking is needed.
    br = _get_cached_reader(path, series)
    frames = []
    for t in range(t0, t1):
        # Read plane-by-plane: 0.8 ms each vs 1334 ms for a multi-dim slice.
        c_vols = []
        for c in range(c0, c1):
            z_planes = []
            for z in range(z0, z1):
                p = br[y0:y1, x0:x1, z:z + 1, c:c + 1, t:t + 1]
                z_planes.append(np.asarray(p).reshape(y1 - y0, x1 - x0))
            c_vols.append(np.stack(z_planes, axis=0))   # (Z_tile, Y, X)
        frames.append(np.stack(c_vols, axis=0))         # (C_tile, Z_tile, Y, X)
    return np.stack(frames, axis=0)                     # → (T_tile, C_tile, Z_tile, Y, X)


def _build_tiled_dask_array(
    path: str,
    tile_mb: float = 256.0,
    series: int = 0,
    shape: 'tuple[int,int,int,int,int] | None' = None,
    dtype: 'np.dtype | None' = None,
) -> da.Array:
    """Build a lazy (T, C, Z, Y, X) dask array using bfio tiles.

    Each dask task reads one priority-ordered tile block via ``_read_bfio_tile``.
    Tile shape is derived from *tile_mb* with priority X → Y → Z → C → T.

    *shape* ``(T, C, Z, Y, X)`` and *dtype* may be supplied from already-loaded
    OME-XML metadata to skip the BioReader shape probe, keeping scene loading
    fully lazy and allowing reads to overlap with writes.
    """
    if shape is not None and dtype is not None:
        T, C, Z, Y, X = shape
    else:
        from bfio import BioReader
        with BioReader(path, level=series) as br:
            T, C, Z, Y, X = br.T, br.C, br.Z, br.Y, br.X
            dtype = br.dtype

    t_tile, c_tile, z_tile, y_tile, x_tile = _compute_tile_shape(
        T, C, Z, Y, X, dtype, tile_mb)
    logger.info(
        "bfio tiled read: shape=(T=%d,C=%d,Z=%d,Y=%d,X=%d) dtype=%s "
        "tile=(%d,%d,%d,%d,%d) tile_mb=%.1f",
        T, C, Z, Y, X, dtype, t_tile, c_tile, z_tile, y_tile, x_tile, tile_mb,
    )

    # Priority order T → C → Z → Y → X (outer → inner).
    # Each task reads (t_tile, c_tile, z_tile, y_tile, x_tile) — one BioReader open.
    t_blocks = []
    for t0 in range(0, T, t_tile):
        t1 = min(t0 + t_tile, T)
        c_blocks = []
        for c0 in range(0, C, c_tile):
            c1 = min(c0 + c_tile, C)
            z_blocks = []
            for z0 in range(0, Z, z_tile):
                z1 = min(z0 + z_tile, Z)
                y_rows = []
                for y0 in range(0, Y, y_tile):
                    y1 = min(y0 + y_tile, Y)
                    x_cols = []
                    for x0 in range(0, X, x_tile):
                        x1 = min(x0 + x_tile, X)
                        block = da.from_delayed(
                            dask.delayed(_read_bfio_tile)(
                                path, y0, y1, x0, x1, z0, z1, c0, c1, t0, t1, series
                            ),
                            shape=(t1-t0, c1-c0, z1-z0, y1-y0, x1-x0),
                            dtype=dtype,
                        )
                        x_cols.append(block)
                    y_rows.append(da.concatenate(x_cols, axis=4))   # join X
                z_blocks.append(da.concatenate(y_rows, axis=3))     # join Y
            c_blocks.append(da.concatenate(z_blocks, axis=2))       # join Z
        t_blocks.append(da.concatenate(c_blocks, axis=1))           # join C
    return da.concatenate(t_blocks, axis=0)                         # join T → (T,C,Z,Y,X)


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
    NGFFImageMeta.  All I/O is async; metadata getters are synchronous once
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

    async def read_omemeta(self):
        if self.root.endswith('ome') or self.root.endswith('xml'):
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
            # 'czi_native' on a non-CZI input falls back to the default reader.
            meta_reader = 'bfio' if self._meta_reader == 'czi_native' else self._meta_reader
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
        pix = self.pixels
        dims = self.reader.img.dims
        shape = (pix.size_t, pix.size_c, pix.size_z, pix.size_y, pix.size_x)
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
            self.arraydata = _build_tiled_dask_array(
                self.root, tile_mb=self._tile_mb, series=self._series,
                shape=shape, dtype=dtype)

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
        if self.reader is None or self.root.endswith('.lsm'):
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
        if isinstance(self.reader, CZIReader):
            dims = self.reader.img.dims
            if hasattr(dims, 'S') and dims.S > 1 and pixels.size_c == 1:
                pixels.size_c = dims.S
                pixels.channels = [
                    Channel(id=f"Channel:{i}", samples_per_pixel=1)
                    for i in range(dims.S)
                ]
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
            self.arraydata = _build_tiled_dask_array(
                self.root, tile_mb=self._tile_mb, series=self._series,
                shape=shape, dtype=dtype)
            return

        self.reader = await read_single_image(self.root, aszarr=self._aszarr, as_mosaic=self._as_mosaic, **{**self._reader_kwargs, **kwargs})

        if _is_bioformats_backed(self.reader):
            # bioio_bioformats fallback — Bio-Formats can't read >2 GB planes in one call.
            # Set the flag BEFORE calling set_scene so it doesn't overwrite self.arraydata.
            self._bfio_tiling = True
            self.reader.set_scene(self._series)
            shape, dtype = self._bfio_shape_from_meta()
            self.arraydata = _build_tiled_dask_array(
                self.root, tile_mb=self._tile_mb, series=self._series,
                shape=shape, dtype=dtype)
        else:
            # Native bioio plugin (CZI, ND2, LIF, TIFF…) — unchanged path.
            self._bfio_tiling = False
            await self.set_scene(self._series)
            await self.set_tile(self._tile)
            self.arraydata = await self.get_arraydata()

    async def get_pyramid(self, version='0.4') -> Pyramid:
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
        if not path.endswith('.h5'):
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
        from eubi_bridge.core.ims_reader import read_ims
        self.reader = await asyncio.to_thread(read_ims, self.root)
        self.arraydata = await self.get_arraydata()

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
# NGFFImageMeta
# ---------------------------------------------------------------------------

class NGFFImageMeta(PFFImageMeta):
    def __init__(self, path, aszarr=False):
        if not is_zarr_group(path):
            raise ValueError(f"Not an NGFF group: {path}")
        super().__init__(path=path, aszarr=aszarr)
        self._base_path = '0'

    async def read_omemeta(self, **kwargs):
        if self.reader is None:
            self.reader = await read_single_image(self.root, aszarr=self._aszarr, **kwargs)
        self._meta = self.reader.pyr.meta

    async def read_img(self, **kwargs):
        self.reader = await read_single_image(self.root, aszarr=self._aszarr, **kwargs)
        self.arraydata = await self.get_arraydata()

    async def read_dataset(self) -> None:
        await asyncio.gather(self.read_omemeta(), self.read_img())
        self._meta = self.reader.pyr.meta

    def get_axes(self) -> str:
        return self._meta.axis_order

    def get_scales(self) -> list:
        return self._meta.get_scale(self._base_path)

    def get_scaledict(self) -> dict:
        return self._meta.get_scaledict(self._base_path)

    def get_units(self) -> list:
        return self._meta.unit_list

    def get_unitdict(self) -> dict:
        return self._meta.unit_dict

    def get_channels(self) -> Optional[list]:
        return getattr(self._meta, 'channels', None)

    async def get_arraydata(self):
        return self.reader.get_image_dask_data()

    async def get_pyramid(self, version='0.4') -> Pyramid:
        return self.reader.pyr


# ---------------------------------------------------------------------------
# ArrayState — pure data + transformation layer (no I/O, no async)
# ---------------------------------------------------------------------------

class ArrayState:
    """Immutable-ish container for array data, physical metadata, and channels.

    Owns all transformations (squeeze, transpose, crop) and channel helpers.
    No I/O.  No async.  Every method that modifies state does so in-place.
    """

    def __init__(self):
        self.array: Optional[Union[da.Array, zarr.Array, DynamicArray]] = None
        self.axes: str = ""
        self.scaledict: Dict[str, Any] = {}
        self.unitdict: Dict[str, Any] = {}
        self._channels: Optional[List[Dict[str, Any]]] = None
        self.chunkdict: Dict[str, Any] = {}
        self.shapedict: Dict[str, int] = {}
        self.pyr: Optional[Pyramid] = None
        self.omemeta: Any = None

    # ── derived properties ────────────────────────────────────────────────

    @property
    def ndim(self) -> int:
        return self.array.ndim if self.array is not None else 0

    @property
    def caxes(self) -> str:
        return ''.join(ax for ax in self.axes if ax != 'c')

    @property
    def scales(self) -> list:
        if len(self.scaledict) < len(self.axes):
            return [self.scaledict[ax] for ax in self.caxes if ax in self.scaledict]
        return [self.scaledict[ax] for ax in self.axes if ax in self.scaledict]

    @property
    def units(self) -> list:
        if len(self.unitdict) < len(self.axes):
            return [self.unitdict[ax] for ax in self.caxes if ax in self.unitdict]
        return [self.unitdict[ax] for ax in self.axes if ax in self.unitdict]

    @property
    def channels(self) -> Optional[list]:
        return self._channels

    @property
    def chunks(self) -> list:
        return [self.chunkdict[ax] for ax in self.axes if ax in self.chunkdict]

    # ── state update ──────────────────────────────────────────────────────

    def update(
        self,
        array=None,
        axes: Optional[str] = None,
        units: Optional[List[Any]] = None,
        scales: Optional[List[Any]] = None,
    ) -> None:
        """Update state from explicit values.  All arguments must be provided
        explicitly — there is no fallback to an external reader here."""
        if axes is not None:
            self.axes = axes
        if array is not None:
            self.array = array
            assert len(self.axes) == self.array.ndim, (
                f"axes length {len(self.axes)} != array ndim {self.array.ndim}"
            )

        self._update_chunk_and_shape()

        if units is not None:
            if len(units) == len(self.axes):
                self.unitdict = dict(zip(list(self.axes), units))
            elif len(units) == len(self.caxes):
                self.unitdict = dict(zip(list(self.caxes), units))
            else:
                raise ValueError(
                    f"Unit length {len(units)} doesn't match axes '{self.axes}' "
                    f"or caxes '{self.caxes}'."
                )

        if scales is not None:
            if len(scales) == len(self.axes):
                self.scaledict = dict(zip(list(self.axes), scales))
            elif len(scales) == len(self.caxes):
                self.scaledict = dict(zip(list(self.caxes), scales))
                self.scaledict['c'] = 1
            else:
                raise ValueError(
                    f"Scale length {len(scales)} doesn't match axes '{self.axes}' "
                    f"or caxes '{self.caxes}'."
                )

        if 'c' in self.shapedict:
            self._ensure_correct_channels()

    def _update_chunk_and_shape(self) -> None:
        if self.array is None:
            return
        chunks = get_array_chunks(self.array)
        if chunks is None:
            raise TypeError(f"Unsupported array type: {type(self.array)}")
        self.chunkdict = dict(zip(list(self.axes), chunks))
        self.shapedict = dict(zip(list(self.axes), self.array.shape))

    # ── metadata helpers ──────────────────────────────────────────────────

    def fill_default_meta(self) -> None:
        """Fill None scale/unit values with sensible defaults."""
        if self.array is None:
            raise ValueError("Array is missing — assign an array before filling defaults.")
        if None not in self.scaledict.values():
            return
        new_sd: Dict[str, Any] = {}
        new_ud: Dict[str, Any] = {}
        for ax, val in self.scaledict.items():
            if val is None:
                if ax in ('z', 'y') and self.scaledict.get('x') is not None:
                    new_sd[ax] = self.scaledict['x']
                    new_ud[ax] = self.unitdict.get('x', unit_map.get('x'))
                else:
                    new_sd[ax] = scale_map.get(ax, 1.0)
                    new_ud[ax] = unit_map.get(ax)
            else:
                new_sd[ax] = val
                new_ud[ax] = self.unitdict.get(ax, unit_map.get(ax))
        new_units = [new_ud[ax] for ax in self.axes if ax in new_ud]
        new_scales = [new_sd[ax] for ax in self.axes if ax in new_sd]
        self.update(self.array, self.axes, new_units, new_scales)

    def _ensure_correct_channels(self) -> None:
        if self.array is None or self._channels is None:
            return
        csize = self.shapedict.get('c')
        if csize is not None and len(self._channels) > csize:
            self._channels = [ch for ch in self._channels if ch.get('label') is not None]

    def fix_bad_channels(self) -> None:
        """Replace missing/empty channel labels with auto-generated defaults."""
        if not self._channels:
            return
        chn = ChannelIterator()
        for i, channel in enumerate(self._channels):
            if channel.get('label') in (None, ''):
                self._channels[i] = next(chn)

    def compute_intensity_extrema(self, dtype) -> tuple:
        if 'c' in self.axes:
            n_channels = self.array.shape[self.axes.index('c')]
        else:
            n_channels = 1
        if dtype is None:
            dtype = self.array.dtype
        if np.issubdtype(dtype, np.integer):
            return ([np.iinfo(dtype).min] * n_channels,
                    [np.iinfo(dtype).max] * n_channels)
        elif np.issubdtype(dtype, np.floating):
            return ([np.finfo(dtype).min] * n_channels,
                    [np.finfo(dtype).max] * n_channels)
        raise ValueError(f"Unsupported dtype {dtype}")

    def compute_intensity_limits(
        self,
        from_array: bool = False,
        dtype=None,
        start_intensity=None,
        end_intensity=None,
    ) -> tuple:
        n_channels = (self.array.shape[self.axes.index('c')]
                      if 'c' in self.axes else 1)
        starts = [start_intensity] * n_channels if start_intensity is not None else None
        ends   = [end_intensity]   * n_channels if end_intensity   is not None else None
        if starts is not None and ends is not None:
            return starts, ends

        if from_array:
            if self.array is None:
                raise ValueError("Array required for from_array=True.")
            axes_to_compute = tuple(i for i, ax in enumerate(self.axes) if ax != 'c')
            arr = da.from_zarr(self.array) if isinstance(self.array, zarr.Array) else self.array
            if starts is None:
                v = arr.min(axis=axes_to_compute).compute().tolist()
                starts = [v] if np.isscalar(v) else v
            if ends is None:
                v = arr.max(axis=axes_to_compute).compute().tolist()
                ends = [v] if np.isscalar(v) else v
        else:
            if dtype is None:
                if self.array is None:
                    raise ValueError("dtype or array required.")
                dtype = self.array.dtype
            if np.issubdtype(dtype, np.integer):
                starts = starts or [np.iinfo(dtype).min] * n_channels
                ends   = ends   or [np.iinfo(dtype).max] * n_channels
            elif np.issubdtype(dtype, np.floating):
                starts = starts or [np.finfo(dtype).min] * n_channels
                ends   = ends   or [np.finfo(dtype).max] * n_channels
            else:
                raise ValueError(f"Unsupported dtype {dtype}")
        return starts, ends

    # ── array transformations ─────────────────────────────────────────────

    def squeeze(self) -> None:
        """Remove singleton dimensions in-place."""
        if all(n > 1 for n in self.array.shape):
            return
        if isinstance(self.array, zarr.Array):
            logger.warning("Zarr arrays don't support squeeze — converting to dask.")
            arr = da.from_array(self.array)
        else:
            arr = self.array
        singlet = {ax for ax, sz in self.shapedict.items() if sz == 1}
        newaxes  = ''.join(ax for ax in self.axes  if ax not in singlet)
        newunits = [self.unitdict[ax]  for ax in self.axes if ax not in singlet and ax in self.unitdict]
        newscales = [self.scaledict[ax] for ax in self.axes if ax not in singlet and ax in self.scaledict]
        newarray = ops.squeeze(arr) if isinstance(arr, DynamicArray) else da.squeeze(arr)
        self.update(newarray, newaxes, newunits, newscales)
        version = self.pyr.meta.multiscales.get('version', '0.4') if self.pyr else '0.4'
        self.pyr = Pyramid().from_array(newarray, axis_order=newaxes,
                                         unit_list=newunits, scale=newscales,
                                         version=version, name='squeezed')

    def transpose(self, newaxes: str) -> None:
        """Reorder array axes in-place."""
        newaxes = ''.join(ax for ax in newaxes if ax in self.axes)
        new_ids  = [self.axes.index(ax) for ax in newaxes]
        newunits = [self.unitdict[ax]  for ax in newaxes if ax in self.unitdict]
        newscales = [self.scaledict[ax] for ax in newaxes if ax in self.scaledict]
        if isinstance(self.array, DynamicArray):
            newarray = ops.transpose(self.array, axes=new_ids)
        else:
            newarray = self.array.transpose(*new_ids)
        self.update(newarray, newaxes, newunits, newscales)

    def crop(self,
             trange=None, crange=None, zrange=None,
             yrange=None, xrange=None) -> None:
        """Slice array and update channel list in-place."""
        if self.pyr is None:
            raise ValueError("Pyramid must be set before cropping.")
        omero_copy = copy.copy(self.pyr.meta.omero)
        slicedict = {
            't': slice(*trange) if trange else slice(None),
            'c': slice(*crange) if crange else slice(None),
            'z': slice(*zrange) if zrange else slice(None),
            'y': slice(*yrange) if yrange else slice(None),
            'x': slice(*xrange) if xrange else slice(None),
        }
        slicedict = {ax: s for ax, s in slicedict.items() if ax in self.axes}
        if 'c' in slicedict and 'c' in self.axes:
            cs = slicedict['c']
            start = cs.start or 0
            stop  = cs.stop  or len(omero_copy['channels'])
            self.pyr.meta.omero['channels'] = omero_copy['channels'][start:stop]
        arr = (da.from_array(self.array) if isinstance(self.array, zarr.Array)
               else self.array)
        slices = tuple(slicedict[ax] for ax in self.axes)
        logger.info(f"Cropping {self.array.shape} → {slicedict}")
        arr = arr[slices]
        logger.info(f"Cropped shape: {arr.shape}")
        self.update(arr, self.axes, self.units, self.scales)

    def get_autocomputed_chunks(self, dtype=None):
        return autocompute_chunk_shape(
            array_shape=self.array.shape,
            axes=self.axes,
            dtype=dtype or self.array.dtype,
        )


# ---------------------------------------------------------------------------
# SceneLoader — owns reader lifetime and scene/tile snapshotting
# ---------------------------------------------------------------------------

class SceneLoader:
    """Opens an image file, snapshots all requested scenes into independent
    ``ArrayManager`` objects, then releases the reader.

    The JVM / bioio reader is always freed before ``load()`` returns, even on
    error, so ``ArrayManager`` never holds a long-lived reader reference.
    """

    def __init__(self, path: str, metadata_reader: str = 'bfio', skip_dask: bool = False,
                 reader_tile_size_mb: float = 256.0, force_bioformats: bool = False,
                 as_mosaic: bool = False,
                 reader_kwargs: Optional[dict] = None,
                 keep_existing_resolutions: bool = False):
        self.path = path
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
        self.is_ngff: bool = False

    async def _open(self) -> None:
        """Detect image type, open reader, populate n_scenes / n_tiles / is_ngff."""
        path = self.path
        if await asyncio.to_thread(is_zarr_group, path):
            self._img = await asyncio.to_thread(NGFFImageMeta, path, self._skip_dask)
            self.is_ngff = True
        elif path.endswith('.h5'):
            self._img = await asyncio.to_thread(H5ImageMeta, path, self._meta_reader)
        elif path.lower().endswith('.ims'):
            self._img = await asyncio.to_thread(
                IMSImageMeta, path, self._meta_reader,
                reader_tile_size_mb=self._tile_mb,
                force_bioformats=self._force_bioformats,
                reader_kwargs=self._reader_kwargs,
                keep_existing_resolutions=self._keep_existing_resolutions,
            )
        elif not self._skip_dask:
            self._img = await asyncio.to_thread(
                PFFImageMeta, path, self._meta_reader, self._skip_dask,
                self._tile_mb, self._force_bioformats, self._as_mosaic,
                self._reader_kwargs,
            )
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
                    self._reader_kwargs,
                )
        await self._img.read_dataset()
        self.n_scenes        = self._img.n_scenes
        self.n_tiles         = self._img.n_tiles
        self.n_views         = getattr(self._img, 'n_views', 1)
        self.n_illuminations = getattr(self._img, 'n_illuminations', 1)
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

    async def _snapshot(self, series: int, tile: Optional[int] = None) -> "ArrayManager":
        """Capture the reader's current state into a fresh independent ArrayManager.

        The caller must have already called ``set_scene`` / ``set_tile`` on
        ``self._img`` before invoking this.
        """
        state = ArrayState()
        state.pyr = await self._img.get_pyramid()
        state.update(
            array=self._img.arraydata,
            axes=self._img.get_axes(),
            units=self._img.get_units(),
            scales=self._img.get_scales(),
        )
        state._channels = self._img.get_channels()
        state.omemeta   = self._img.omemeta
        mgr = ArrayManager(
            path=self.path,
            metadata_reader=self._meta_reader,
            skip_dask=self._skip_dask,
            state=state,
            series=series,
            mosaic_tile_index=tile,
        )
        mgr._n_scenes    = self.n_scenes
        mgr._n_tiles     = self.n_tiles
        mgr._is_ngff     = self.is_ngff
        mgr._bfio_tiling = getattr(self._img, '_bfio_tiling', False)
        return mgr

    async def load(
        self,
        scene_indices: Union[int, str, List[int]] = 0,
        mosaic_tile_index: Optional[Union[int, str, List[int]]] = None,
    ) -> List["ArrayManager"]:
        """Open file, snapshot all requested scenes/tiles, release reader.

        Returns one ``ArrayManager`` per (scene, tile) pair.
        """
        await self._open()
        managers: List["ArrayManager"] = []
        try:
            if scene_indices == 'all':
                indices = list(range(self.n_scenes))
            elif np.isscalar(scene_indices):
                indices = [int(scene_indices)]
            else:
                indices = list(scene_indices)
            valid = [i for i in indices if i < self.n_scenes]
            if len(valid) < len(indices):
                logger.warning(
                    f"Skipping out-of-range scene indices: {set(indices) - set(valid)}")

            if mosaic_tile_index is not None:
                if mosaic_tile_index == 'all':
                    tile_indices: List[int] = list(range(self.n_tiles))
                elif np.isscalar(mosaic_tile_index):
                    tile_indices = [int(mosaic_tile_index)]
                else:
                    tile_indices = list(mosaic_tile_index)
                valid_tiles = [t for t in tile_indices if t < self.n_tiles]
                if len(valid_tiles) < len(tile_indices):
                    logger.warning(
                        f"Skipping out-of-range tile indices: "
                        f"{set(tile_indices) - set(valid_tiles)}")
            else:
                valid_tiles = None

            for scene_idx in valid:
                await self._img.set_scene_isolated(scene_idx)
                if valid_tiles is None:
                    managers.append(await self._snapshot(scene_idx))
                else:
                    for tile_idx in valid_tiles:
                        await self._img.set_tile(tile_idx)
                        managers.append(await self._snapshot(scene_idx, tile_idx))
        finally:
            self._img = None
        return managers


# ---------------------------------------------------------------------------
# ArrayManager — coordinator (loading + I/O + scene orchestration)
# ---------------------------------------------------------------------------

class ArrayManager:
    """Coordinates image loading and exposes a unified interface for the
    conversion pipeline.

    Internally owns:
    - ``_reader`` (ImageReader) — format-specific loader
    - ``_state``  (ArrayState)  — pure array data + metadata

    All public attributes and methods that ``conversion_worker.py`` accesses
    (``array``, ``axes``, ``scaledict``, ``fill_default_meta()``, etc.) are
    forwarded transparently to ``_state``.
    """

    essential_omexml_fields = PFFImageMeta.essential_omexml_fields

    def __init__(
        self,
        path: Union[str, Path, None] = None,
        metadata_reader: str = 'bfio',
        skip_dask: bool = False,
        state: Optional[ArrayState] = None,
        **kwargs: Any,
    ):
        self.path = str(path) if path is not None else None
        self.series = kwargs.get('series', 0)
        self.series_path = (self.path or '') + f'_{self.series}'
        self.mosaic_tile_index = kwargs.get('mosaic_tile_index', None)
        if self.mosaic_tile_index is not None:
            self.series_path += f'_tile{self.mosaic_tile_index}'

        self._meta_reader        = metadata_reader
        self._skip_dask          = skip_dask
        self._tile_mb            = float(kwargs.get('reader_tile_size_mb', 256.0))
        self._force_bioformats   = bool(kwargs.get('force_bioformats', False))
        self._as_mosaic          = bool(kwargs.get('as_mosaic', False))
        self._keep_existing_resolutions = bool(kwargs.get('keep_existing_resolutions', False))

        self.loaded_scenes:  Optional[dict] = None
        self.loaded_tiles:   Optional[dict] = None
        self.loaded_views_illuminations: Optional[dict] = None
        self._n_scenes:       int  = 0
        self._n_scenes_in_file: int = 0
        self._n_tiles:        int  = 0
        self._n_views:        int  = 1
        self._n_illuminations: int = 1
        self._is_ngff:        bool = False
        self._bfio_tiling:    bool = False

        # Data + metadata layer (pure, no I/O)
        self.state: ArrayState = state if state is not None else ArrayState()

    def __getstate__(self): return self.__dict__.copy()
    def __setstate__(self, state): self.__dict__.update(state)

    # ── ArrayState property forwarding (backward-compatible interface) ────

    @property
    def array(self): return self.state.array
    @array.setter
    def array(self, v): self.state.array = v

    @property
    def axes(self) -> str: return self.state.axes
    @axes.setter
    def axes(self, v): self.state.axes = v

    @property
    def ndim(self) -> int: return self.state.ndim

    @property
    def caxes(self) -> str: return self.state.caxes

    @property
    def scaledict(self) -> dict: return self.state.scaledict
    @scaledict.setter
    def scaledict(self, v): self.state.scaledict = v

    @property
    def unitdict(self) -> dict: return self.state.unitdict
    @unitdict.setter
    def unitdict(self, v): self.state.unitdict = v

    @property
    def _channels(self): return self.state._channels
    @_channels.setter
    def _channels(self, v): self.state._channels = v

    @property
    def chunkdict(self) -> dict: return self.state.chunkdict
    @property
    def shapedict(self) -> dict: return self.state.shapedict
    @property
    def scales(self) -> list: return self.state.scales
    @property
    def units(self) -> list: return self.state.units
    @property
    def channels(self) -> Optional[list]: return self.state.channels
    @property
    def chunks(self) -> list: return self.state.chunks

    @property
    def pyr(self) -> Optional[Pyramid]: return self.state.pyr
    @pyr.setter
    def pyr(self, v): self.state.pyr = v

    @property
    def omemeta(self): return self.state.omemeta
    @omemeta.setter
    def omemeta(self, v): self.state.omemeta = v

    # ── ArrayState transformation delegation ─────────────────────────────

    def fill_default_meta(self):    return self.state.fill_default_meta()
    def fix_bad_channels(self):     return self.state.fix_bad_channels()
    def squeeze(self):              return self.state.squeeze()
    def transpose(self, newaxes):   return self.state.transpose(newaxes)
    def crop(self, **kwargs):       return self.state.crop(**kwargs)
    def get_autocomputed_chunks(self, dtype=None):
        return self.state.get_autocomputed_chunks(dtype)
    def compute_intensity_extrema(self, dtype):
        return self.state.compute_intensity_extrema(dtype)
    def compute_intensity_limits(self, **kwargs):
        return self.state.compute_intensity_limits(**kwargs)
    def _ensure_correct_channels(self):
        return self.state._ensure_correct_channels()

    def set_arraydata(
        self,
        array=None,
        axes: Optional[str] = None,
        units: Optional[list] = None,
        scales: Optional[list] = None,
        **_,
    ) -> None:
        """Update ArrayState.  Falls back to the current state for any
        argument not explicitly supplied."""
        if axes   is None: axes   = self.state.axes
        if units  is None: units  = self.state.units
        if scales is None: scales = self.state.scales
        self.state.update(array=array, axes=axes, units=units, scales=scales)

    # ── Loading (delegates to SceneLoader) ───────────────────────────────

    async def init(self):
        """Detect image type, load data, populate state via SceneLoader."""
        if self.path is None:
            return self
        loader = SceneLoader(self.path, self._meta_reader, self._skip_dask,
                             reader_tile_size_mb=self._tile_mb,
                             force_bioformats=self._force_bioformats,
                             as_mosaic=self._as_mosaic,
                             keep_existing_resolutions=self._keep_existing_resolutions)
        await loader._open()
        self._n_scenes = loader.n_scenes
        self._n_tiles  = loader.n_tiles
        self._is_ngff  = loader.is_ngff
        try:
            await loader._img.set_scene(self.series)  # type: ignore[union-attr]
            if self.mosaic_tile_index is not None:
                await loader._img.set_tile(self.mosaic_tile_index)  # type: ignore[union-attr]
            mgr = await loader._snapshot(self.series, self.mosaic_tile_index)
            self.state = mgr.state
        finally:
            loader._img = None
        return self

    def close_reader(self) -> None:
        """No-op: the reader is now owned and released by SceneLoader."""

    async def load_scenes(self, scene_indices: Union[int, str, List[int]],
                          mosaic_tile_index: Optional[Union[int, str, List[int]]] = None):
        """Snapshot the requested scenes — and, when ``mosaic_tile_index`` is
        given, every (scene, tile) combination — into independent managers.

        Passing ``mosaic_tile_index`` produces one manager per (scene, tile)
        pair (each carrying ``.series`` and ``.mosaic_tile_index``), enabling
        the multi-scene × multi-tile case in a single file open.
        """
        assert self.path is not None, "Cannot load scenes: path is None."
        loader = SceneLoader(self.path, self._meta_reader, self._skip_dask,
                             reader_tile_size_mb=self._tile_mb,
                             force_bioformats=self._force_bioformats,
                             as_mosaic=self._as_mosaic,
                             keep_existing_resolutions=self._keep_existing_resolutions)
        managers = await loader.load(scene_indices=scene_indices,
                                     mosaic_tile_index=mosaic_tile_index)
        self._n_scenes         = loader.n_scenes
        self._n_scenes_in_file = loader.n_scenes_in_file
        self._n_tiles          = loader.n_tiles
        self._n_views         = loader.n_views
        self._n_illuminations = loader.n_illuminations
        self._is_ngff         = loader.is_ngff
        self.loaded_scenes = {m.series_path: m for m in managers}
        return self.loaded_scenes

    async def load_tiles(self, tile_indices: Union[int, str, List[int]]):
        if not self.loaded_scenes:
            raise RuntimeError("Call load_scenes() before load_tiles().")
        scene_mgr = next(iter(self.loaded_scenes.values()))
        scene_idx = scene_mgr.series
        assert self.path is not None, "Cannot load tiles: path is None."
        loader = SceneLoader(self.path, self._meta_reader, self._skip_dask,
                             reader_tile_size_mb=self._tile_mb,
                             force_bioformats=self._force_bioformats,
                             as_mosaic=self._as_mosaic,
                             keep_existing_resolutions=self._keep_existing_resolutions)
        await loader._open()
        try:
            await loader._img.set_scene(scene_idx)  # type: ignore[union-attr]
            n_tiles = loader.n_tiles
            if tile_indices == 'all':
                tile_list_input: List[int] = list(range(n_tiles))
            elif np.isscalar(tile_indices):
                tile_list_input = [int(tile_indices)]  # type: ignore[arg-type]
            else:
                tile_list_input = list(tile_indices)  # type: ignore[arg-type]
            valid = [i for i in tile_list_input if i < n_tiles]
            if len(valid) < len(tile_list_input):
                logger.warning(
                    f"Skipping out-of-range tile indices: {set(tile_list_input) - set(valid)}")
            tiles = []
            for tile_idx in valid:
                await loader._img.set_tile(tile_idx)  # type: ignore[union-attr]
                tiles.append(await loader._snapshot(scene_idx, tile_idx))
        finally:
            loader._img = None
            self._n_tiles = loader.n_tiles
        self.loaded_tiles = {t.series_path: t for t in tiles}
        return self.loaded_tiles

    async def load_views_illuminations(
        self,
        view_indices: Union[int, str, List[int]],
        illumination_indices: Union[int, str, List[int]],
        concat_views: bool = False,
        concat_illuminations: bool = False,
    ) -> dict:
        """Iterate over requested views and illuminations for all loaded scenes,
        snapshotting each (scene, view, illumination) combination as an
        independent ``ArrayManager``.

        The file is opened exactly **once** regardless of how many scenes are
        loaded.  Formats with only a single view and single illumination (LIF,
        ND2, OME-TIFF, …) early-exit without cost, leaving
        ``loaded_views_illuminations`` as ``None`` so that ``unary_worker``
        falls back to ``loaded_scenes``.

        When *concat_views* or *concat_illuminations* is True the arrays for
        the respective dimension are stacked along the channel axis and a
        single manager is produced per (scene, non-concatenated-dim) group
        instead of one per combination.

        Results are stored in ``self.loaded_views_illuminations`` and also
        returned as ``{series_path: ArrayManager}``.
        """
        if not self.loaded_scenes:
            raise RuntimeError("Call load_scenes() before load_views_illuminations().")

        assert self.path is not None
        import dask.array as da  # local import — mirrors rest of module

        # Open the file ONCE with full view/illumination exposure so that
        # n_views / n_illuminations report the real totals and all
        # (scene × view × illumination) combinations can be processed in a
        # single file open — avoiding the previous per-scene re-open cost.
        loader = SceneLoader(
            self.path, self._meta_reader, self._skip_dask,
            reader_tile_size_mb=self._tile_mb,
            force_bioformats=self._force_bioformats,
            as_mosaic=self._as_mosaic,
            keep_existing_resolutions=self._keep_existing_resolutions,
            reader_kwargs={'view_index': 'all', 'illumination_index': 'all'},
        )
        await loader._open()
        assert loader._img is not None, "SceneLoader._open() failed to initialize _img"
        reader = loader._img

        result: dict = {}

        try:
            # Probe actual total counts using the first loaded scene.
            first_scene_mgr = next(iter(self.loaded_scenes.values()))
            await reader.set_scene(first_scene_mgr.series)
            n_total_views        = reader.n_views
            n_total_illuminations = reader.n_illuminations

            # Persist so callers (e.g. inspect page) can read them.
            self._n_views        = n_total_views
            self._n_illuminations = n_total_illuminations

            # If the file has only a single view AND a single illumination
            # there is nothing to iterate over.  Return without setting
            # loaded_views_illuminations so unary_worker falls through to
            # loaded_scenes (which already holds the correct data).
            if n_total_views <= 1 and n_total_illuminations <= 1:
                return {}

            # ── Inner helper — snapshot one (scene, view, illumination) ──
            async def _snapshot_vi(scene_idx: int, v_idx: int, i_idx: int,
                                   tile_idx: Optional[int] = None) -> "ArrayManager":
                # Honour the mosaic tile this scene-manager represents so that
                # multi-tile files compose with view/illumination iteration —
                # otherwise every tile collapses to the reader's default tile.
                tiled = (
                    tile_idx is not None and reader.n_tiles > 1
                    and getattr(reader, 'reader', None) is not None
                    and hasattr(reader.reader, 'set_tile')
                )
                if tiled:
                    reader.reader.set_tile(tile_idx)  # sync — updates CZI index_map['M']
                reader.set_view(v_idx)           # sync — updates CZI index_map
                reader.set_illumination(i_idx)   # sync — updates CZI index_map
                # Re-read array with updated index_map so the snapshot holds
                # the correct data slice.
                if not getattr(reader, '_bfio_tiling', False):
                    reader.arraydata = await reader.get_arraydata()
                mgr = await loader._snapshot(scene_idx, tile_idx if tiled else None)
                # OME-XML may report channels for all illuminations while
                # get_image_dask_data returns only the slice for the current
                # illumination.  Select the per-illumination window.
                if mgr.array is not None and mgr.state._channels and 'c' in mgr.state.axes:
                    c_size = mgr.array.shape[mgr.state.axes.index('c')]
                    if len(mgr.state._channels) > c_size:
                        offset = i_idx * c_size
                        mgr.state._channels = mgr.state._channels[offset:offset + c_size]
                # Build a clean series_path with view/illumination suffix BEFORE
                # the extension so _generate_output_path parses it correctly.
                p = Path(loader.path)
                vi_parts: list = []
                if n_total_views > 1:
                    vi_parts.append(f'_view{v_idx}')
                if n_total_illuminations > 1:
                    vi_parts.append(f'_illu{i_idx}')
                if tiled:
                    vi_parts.append(f'_tile{tile_idx}')
                if vi_parts:
                    mgr.series_path = str(
                        p.parent / (p.stem + ''.join(vi_parts) + p.suffix)
                    )
                return mgr

            # ── Process every loaded scene in the SAME file open ──────────
            for scene_key, scene_mgr in self.loaded_scenes.items():
                scene_idx = scene_mgr.series
                tile_idx = scene_mgr.mosaic_tile_index
                await reader.set_scene(scene_idx)

                # Resolve view list for this scene
                n_views = reader.n_views
                if view_indices == 'all':
                    v_list: List[int] = list(range(n_views))
                elif np.isscalar(view_indices):
                    v_list = [int(view_indices)]
                else:
                    v_list = list(view_indices)
                v_list = [v for v in v_list if v < n_views]

                # Resolve illumination list for this scene
                n_illu = reader.n_illuminations
                if illumination_indices == 'all':
                    i_list: List[int] = list(range(n_illu))
                elif np.isscalar(illumination_indices):
                    i_list = [int(illumination_indices)]
                else:
                    i_list = list(illumination_indices)
                i_list = [i for i in i_list if i < n_illu]

                combos: List[tuple] = [(v, i) for v in v_list for i in i_list]

                if not concat_views and not concat_illuminations:
                    # ── separate output per (view, illumination) ──────────
                    for v_idx, i_idx in combos:
                        mgr = await _snapshot_vi(scene_idx, v_idx, i_idx, tile_idx)
                        result[mgr.series_path] = mgr

                else:
                    # ── concatenate along channel axis ─────────────────────
                    # Group by the non-concatenated dimension(s).
                    if concat_views and concat_illuminations:
                        groups = {(None, None): combos}
                    elif concat_views:
                        # One output per illumination — views are concatenated
                        groups = {(None, i): [(v, i) for v in v_list] for i in i_list}
                    else:
                        # One output per view — illuminations are concatenated
                        groups = {(v, None): [(v, i) for i in i_list] for v in v_list}

                    for group_key, group_combos in groups.items():
                        arrays: list = []
                        all_channels: list = []
                        snap_mgr = None
                        # Per-view channel name fallback so illuminations with
                        # unlabelled channels inherit names from illumination 0.
                        view_fallback_names: dict = {}
                        for v_idx, i_idx in group_combos:
                            snap_mgr = await _snapshot_vi(scene_idx, v_idx, i_idx, tile_idx)
                            arr = snap_mgr.array
                            if arr is None:
                                continue
                            arrays.append(arr)
                            prefix_parts = []
                            if concat_views:
                                prefix_parts.append(f'View{v_idx}')
                            if concat_illuminations:
                                prefix_parts.append(f'Illu{i_idx}')
                            prefix = '_'.join(prefix_parts) + '_'
                            snap_channels = snap_mgr.state._channels or []
                            if v_idx not in view_fallback_names:
                                view_fallback_names[v_idx] = [
                                    ch.get('label') for ch in snap_channels
                                ]
                            fallback = view_fallback_names[v_idx]
                            for ch_idx, ch in enumerate(snap_channels):
                                name = ch.get('label')
                                if not name:
                                    name = (fallback[ch_idx]
                                            if ch_idx < len(fallback) and fallback[ch_idx]
                                            else f'Ch{ch_idx}')
                                all_channels.append({**ch, 'label': f'{prefix}{name}'})

                        if not arrays or snap_mgr is None:
                            continue

                        concat_arr = da.concatenate(arrays, axis=1)  # axis 1 = C in TCZYX
                        # state.update() keeps shapedict/chunkdict in sync with the
                        # new channel count; direct array assignment would leave
                        # shapedict stale and cause _ensure_correct_channels to
                        # truncate the merged channel list.
                        snap_mgr.state.update(
                            array=concat_arr,
                            axes=snap_mgr.state.axes,
                        )
                        if all_channels:
                            snap_mgr.state._channels = all_channels
                        result[snap_mgr.series_path] = snap_mgr

        finally:
            loader._img = None

        self.loaded_views_illuminations = result
        return result

    # ── Metadata helpers ──────────────────────────────────────────────────

    def get_pixel_size_basemap(self, t=1, z=1, y=1, x=1, **_):
        return {'pixel_size_t': t, 'pixel_size_z': z, 'pixel_size_y': y, 'pixel_size_x': x}

    def get_unit_basemap(self, t='second', z='micrometer', y='micrometer', x='micrometer', **_):
        return {'unit_t': t, 'unit_z': z, 'unit_y': y, 'unit_x': x}

    def update_meta(self, new_scaledict: dict = None, new_unitdict: dict = None):
        new_scaledict = new_scaledict or {}
        new_unitdict  = new_unitdict  or {}
        # Use the already-loaded state as the base.
        sd = dict(self.state.scaledict)
        for k, v in new_scaledict.items():
            if k in sd and v is not None:
                sd[k] = v
        scales = [sd[ax] for ax in (self.axes if 'c' in sd else self.caxes)]

        ud = dict(self.state.unitdict)
        for k, v in new_unitdict.items():
            if k in ud and v is not None:
                ud[k] = v
        units = [expand_units(ud[ax]) for ax in (self.axes if 'c' in ud else self.caxes)]
        self.state.update(array=self.array, axes=self.axes, units=units, scales=scales)

    # ── Pyramid / OME-XML I/O (async, genuine I/O) ───────────────────────

    async def sync_pyramid(
        self,
        create_omexml_if_not_exists: bool = False,
        save_changes: bool = False,
    ) -> None:
        if self.state.pyr is None:
            raise RuntimeError("No pyramid — load data first.")
        pyr = self.state.pyr
        await asyncio.to_thread(pyr.update_scales, **self.scaledict)
        await asyncio.to_thread(pyr.update_units,  **self.unitdict)

        if self._is_ngff:
            for i, ch_meta in enumerate(self.channels):
                pyr.meta.metadata['omero']['channels'][i].update(ch_meta)
        else:
            pyr.meta.metadata['omero']['channels'] = self.channels
        pyr.meta._pending_changes = True

        if self.state.omemeta is None:
            def _create():
                return create_ome_xml(
                    image_shape=pyr.base_array.shape,
                    axis_order=pyr.axes,
                    pixel_size_x=pyr.meta.scaledict.get('0', {}).get('x'),
                    pixel_size_y=pyr.meta.scaledict.get('0', {}).get('y'),
                    pixel_size_z=pyr.meta.scaledict.get('0', {}).get('z'),
                    pixel_size_t=pyr.meta.scaledict.get('0', {}).get('t'),
                    unit_x=pyr.meta.unit_dict.get('x'),
                    unit_y=pyr.meta.unit_dict.get('y'),
                    unit_z=pyr.meta.unit_dict.get('z'),
                    unit_t=pyr.meta.unit_dict.get('t'),
                    dtype=str(pyr.base_array.dtype),
                    image_name=pyr.meta.multiscales.get('name', 'Default image'),
                    channel_names=[ch['label'] for ch in self.channels],
                )
            self.state.omemeta = await asyncio.to_thread(_create)

        try:
            if pyr.gr is not None and ('OME' in list(pyr.gr.keys()) or create_omexml_if_not_exists):
                await self.save_omexml(pyr.gr.store.root, overwrite=True)
        except (OSError, AttributeError) as e:
            logger.warning(f"Could not save OME-XML: {e}")

        if save_changes:
            await asyncio.to_thread(pyr.meta.save_changes)

    async def create_omemeta(self):
        self.state.fill_default_meta()
        ps_map = self.get_pixel_size_basemap(**self.scaledict)
        u_map  = self.get_unit_basemap(**self.unitdict)
        self.state.omemeta = create_ome_xml(
            image_shape=self.array.shape,
            axis_order=self.axes,
            **ps_map, **u_map,
            dtype=str(self.array.dtype),
            channel_names=[ch['label'] for ch in self.channels],
        )
        pixels = self.state.omemeta.images[0].pixels
        missing = self.essential_omexml_fields - pixels.model_fields_set
        pixels.model_fields_set.update(missing)
        self.state.omemeta.images[0].pixels = pixels
        return self

    async def save_omexml(self, base_path: str, overwrite: bool = False) -> None:
        await self.create_omemeta()
        assert self.state.omemeta is not None
        gr = await asyncio.to_thread(zarr.group, base_path)
        try:
            path = os.path.join(gr.store.root, 'OME', 'METADATA.ome.xml')
        except AttributeError as e:
            logger.warning(f"OME-XML can only be written to local stores: {e}")
            return
        await asyncio.to_thread(gr.create_group, 'OME', overwrite=overwrite)

        def _write(p, text):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, 'w', encoding='utf-8') as f:
                f.write(text)

        await asyncio.to_thread(_write, path, self.state.omemeta.to_xml())
        if gr.info._zarr_format == 2:
            gr['OME'].attrs["series"] = [self.series]
        else:
            gr['OME'].attrs["ome"] = dict(version="0.5", series=[str(self.series)])

    # ── Stubs ─────────────────────────────────────────────────────────────

    def split_series(self): pass
    def split(self): pass


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


# ---------------------------------------------------------------------------
# BatchManager
# ---------------------------------------------------------------------------

class BatchManager:
    """Applies operations to a collection of ArrayManagers in step."""

    def __init__(self):
        self.managers: dict = {}

    async def init(self, managers: dict) -> "BatchManager":
        self.managers = managers
        return self

    async def _collect_scaledict(self, **kwargs) -> dict:
        keys = ['t', 'c', 'z', 'y', 'x']
        vals = [kwargs.get(f'{ax}_scale' if ax != 'c' else 'channel_scale', None) for ax in keys]
        return {k: v for k, v in zip(keys, vals) if v is not None}

    async def _collect_unitdict(self, **kwargs) -> dict:
        keys = ['t', 'c', 'z', 'y', 'x']
        vals = [kwargs.get(f'{ax}_unit' if ax != 'c' else 'channel_unit', None) for ax in keys]
        return {k: v for k, v in zip(keys, vals) if v is not None}

    async def _collect_chunks(self, **kwargs) -> dict:
        keys = ['t', 'c', 'z', 'y', 'x']
        vals = [kwargs.get(f'{ax}_chunk' if ax != 'c' else 'channel_chunk', None) for ax in keys]
        return {k: v for k, v in zip(keys, vals) if v is not None}

    async def fill_default_meta(self):
        for mgr in self.managers.values():
            mgr.fill_default_meta()

    async def squeeze(self):
        for mgr in self.managers.values():
            mgr.squeeze()

    async def crop(self, time_range=None, channel_range=None,
                   z_range=None, y_range=None, x_range=None, **_):
        if any(v is not None for v in (time_range, channel_range, z_range, y_range, x_range)):
            for mgr in self.managers.values():
                mgr.crop(trange=time_range, crange=channel_range,
                          zrange=z_range, yrange=y_range, xrange=x_range)

    async def transpose(self, newaxes: str):
        for mgr in self.managers.values():
            mgr.transpose(newaxes)

    async def sync_pyramids(self):
        pass  # stub

    async def to_cupy(self):
        pass  # stub
