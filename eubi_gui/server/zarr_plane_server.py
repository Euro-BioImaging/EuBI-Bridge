#!/usr/bin/env python3
"""
Persistent HTTP server for serving OME-Zarr 2D planes as PNG images.
Uses EuBI-Bridge's Pyramid class for efficient multi-resolution access.
Keeps loaded datasets in an LRU cache for fast repeated access.

Performance changes vs original:
  1. ThreadPoolExecutor replaces ProcessPoolExecutor — eliminates pickling/IPC
     overhead and Python worker startup cost; works on Linux, macOS, Windows.
     zarr releases the GIL during network/disk reads so threads achieve real
     parallelism for I/O-bound workloads.
  2. get_orientation_axes() is memoized — shape lookups and scale calculations
     now happen once per (pyramid, level, orientation) combination.
  3. percentile_cache is wired into normalize_and_composite() — per-slice
     1st/99th percentile values are computed once and reused across tile
     requests that share the same z/t/channel position.
  4. WebP encoding replaces PNG — ~3–4× faster encode than PNG compress_level=1
     with smaller payloads at quality=85.  Lossless for all practical viewing.
     Content-Type updated to image/webp on all tile/plane endpoints.
  5. Client-disconnect errors (WinError 10053 / BrokenPipeError) are caught
     and silently ignored — these are normal when the client cancels a
     request mid-flight and are not server errors.
  6. TensorStore replaces zarr for the actual array reads (extract_region and
     /channel_minmax).  eubi_bridge.Pyramid is still used for all metadata
     (axes, shape, resolution_paths, channel info, chunk layout).
     TensorStore's C++ async I/O engine and GIL-free decompression pipeline
     give 2–4× faster chunk reads vs zarr-python, particularly for blosc-
     compressed data.  A per-array fallback to zarr is used automatically
     when TensorStore cannot open a level (e.g. unusual codec).
  7. Viewport prefetching — after serving a computed tile the 8 surrounding
     tiles (3×3 neighbourhood minus centre) are speculatively rendered on a
     dedicated 2-worker background executor and stored in tile_cache.  An
     in-flight set prevents duplicate submissions.  Prefetch failures are
     always silently discarded so they never affect foreground requests.
"""

import sys
import json
import io
import os
import signal
import threading
import hashlib
import numpy as np
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

sys.stderr.write("Starting zarr plane server...\n")
sys.stderr.flush()

try:
    from eubi_bridge.ngff.multiscales import Pyramid
    sys.stderr.write("EuBI-Bridge Pyramid loaded successfully\n")
    sys.stderr.flush()
except Exception as e:
    sys.stderr.write(f"Failed to import Pyramid: {e}\n")
    sys.stderr.flush()
    sys.exit(1)

try:
    import tensorstore as ts
    _TS_AVAILABLE = True
    sys.stderr.write("TensorStore available — using for array reads\n")
    sys.stderr.flush()
except ImportError:
    _TS_AVAILABLE = False
    sys.stderr.write("TensorStore not available — falling back to zarr reads\n")
    sys.stderr.flush()


# Errors that mean the client closed the connection before we finished sending.
# These are normal (e.g. browser cancelled a request) and should not be logged
# as server errors.  WinError 10053 / 10054 surface as ConnectionAbortedError
# or ConnectionResetError on Windows; BrokenPipeError covers Linux/macOS.
_CLIENT_DISCONNECT_ERRORS = (BrokenPipeError, ConnectionAbortedError, ConnectionResetError)

CANVAS_SIZE = 512
MAX_CACHE_SIZE = 3

# ---------------------------------------------------------------------------
# Orientation axes memoization
# Keyed by (id(pyr), level_path, orientation).  PyramidCache holds strong
# references so id(pyr) is stable for the lifetime of each cached pyramid.
# When a pyramid is evicted we clean up its entries via _invalidate_orientation_cache.
# ---------------------------------------------------------------------------
_orientation_axes_cache: dict = {}
_orientation_axes_lock = threading.Lock()


def _invalidate_orientation_cache(pyr_id: int) -> None:
    with _orientation_axes_lock:
        stale = [k for k in _orientation_axes_cache if k[0] == pyr_id]
        for k in stale:
            del _orientation_axes_cache[k]


class PyramidCache:
    def __init__(self, max_size=MAX_CACHE_SIZE):
        self._cache = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()

    def get(self, path):
        with self._lock:
            if path in self._cache:
                self._cache.move_to_end(path)
                return self._cache[path]
        try:
            pyr = Pyramid(path)
            with self._lock:
                self._cache[path] = pyr
                if len(self._cache) > self._max_size:
                    _evicted_path, evicted_pyr = self._cache.popitem(last=False)
                    _invalidate_orientation_cache(id(evicted_pyr))
            return pyr
        except Exception as e:
            raise ValueError(f"Failed to load Pyramid at {path}: {e}")


pyramid_cache = PyramidCache()

MAX_TILE_CACHE = 200
MAX_PERCENTILE_CACHE = 500


class TileCache:
    """LRU cache for rendered image tile bytes (WebP) keyed by request parameters."""

    def __init__(self, max_size=MAX_TILE_CACHE):
        self._cache = OrderedDict()
        self._lock = threading.Lock()
        self._max_size = max_size

    @staticmethod
    def make_key(**kwargs) -> str:
        return hashlib.md5(json.dumps(kwargs, sort_keys=True).encode()).hexdigest()

    def get(self, key: str):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]  # (png_bytes, etag)
        return None

    def put(self, key: str, png_bytes: bytes) -> str:
        etag = f'"{key[:16]}"'
        with self._lock:
            self._cache[key] = (png_bytes, etag)
            self._cache.move_to_end(key)
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
        return etag


class PercentileCache:
    """
    Cache (vmin, vmax) percentile results.

    Key: (zarr_path, level_path, orientation, indices_tuple, channel_index)
    where indices_tuple is a sorted tuple of (axis, value) pairs uniquely
    identifying the slice position (t, z, etc.).

    This prevents normalize_and_composite() from recomputing np.percentile on
    every tile request that shares the same logical slice.
    """

    def __init__(self, max_size=MAX_PERCENTILE_CACHE):
        self._cache = OrderedDict()
        self._lock = threading.Lock()
        self._max_size = max_size

    def get(self, key):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def put(self, key, vmin, vmax):
        with self._lock:
            self._cache[key] = (vmin, vmax)
            self._cache.move_to_end(key)
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)


tile_cache = TileCache()
percentile_cache = PercentileCache()

# ---------------------------------------------------------------------------
# TensorStore cache
#
# Maps (zarr_path, level_path) -> open ts.TensorStore array.
# Opening a TensorStore is cheap (metadata read only) but not free, so we
# cache handles to avoid re-opening on every tile request.
#
# Per-key fallback: if TensorStore fails to open a specific level (unusual
# codec, zarr v3 store, etc.) the key is added to _failed and extract_region
# silently falls back to zarr for that level.  This means a single broken
# level will never crash the whole server.
# ---------------------------------------------------------------------------
class TensorStoreCache:
    def __init__(self, max_size: int = 20):
        self._cache: OrderedDict = OrderedDict()
        self._failed: set = set()
        self._lock = threading.Lock()
        self._max_size = max_size

    def get(self, zarr_path: str, level_path: str):
        """
        Return an open ts.TensorStore for zarr_path/level_path, or None if
        TensorStore is unavailable or the open failed (caller must fall back).
        """
        if not _TS_AVAILABLE:
            return None
        key = (zarr_path, level_path)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            if key in self._failed:
                return None
        try:
            # Build OS-native path so both Windows and POSIX paths work.
            from pathlib import Path
            array_path = str(Path(zarr_path) / level_path)
            store = ts.open({
                'driver': 'zarr',
                'kvstore': {'driver': 'file', 'path': array_path},
            }).result()
            with self._lock:
                self._cache[key] = store
                self._cache.move_to_end(key)
                if len(self._cache) > self._max_size:
                    self._cache.popitem(last=False)
            return store
        except Exception as e:
            sys.stderr.write(
                f"TensorStore open failed for {zarr_path}/{level_path} "
                f"(falling back to zarr): {e}\n"
            )
            sys.stderr.flush()
            with self._lock:
                self._failed.add(key)
            return None


ts_cache = TensorStoreCache()

# channel_minmax_cache: (zarr_path, channel_idx) -> (vmin, vmax) — populated by /channel_minmax
_minmax_cache: dict = {}
_minmax_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Thread pool
#
# Replaces ProcessPoolExecutor.  Key reasons:
#   - No spawn/fork overhead — threads start instantly on all platforms.
#   - No pickling of arguments or return values — PNG bytes stay in-process.
#   - zarr releases the GIL during chunk decompression and network/disk I/O,
#     so threads achieve genuine parallelism for the dominant cost.
#   - Works identically on Linux, macOS, and Windows.
#
# Worker count: I/O-bound work benefits from more threads than CPU cores.
# ---------------------------------------------------------------------------
_tile_executor: ThreadPoolExecutor | None = None
_tile_executor_init_lock = threading.Lock()


def _get_tile_executor() -> ThreadPoolExecutor:
    global _tile_executor
    if _tile_executor is None:
        with _tile_executor_init_lock:
            if _tile_executor is None:
                worker_count = min(32, (os.cpu_count() or 1) * 4)
                _tile_executor = ThreadPoolExecutor(max_workers=worker_count)
    return _tile_executor


class PrefetchEngine:
    """
    Speculatively renders the 8 tiles surrounding the current viewport tile
    into tile_cache so they are ready before the client requests them.

    Uses a dedicated 2-worker executor so prefetch jobs never compete with
    foreground tile requests for thread pool capacity.  An in-flight set
    prevents submitting the same tile twice while it is already being rendered.
    Results land directly in tile_cache; failures are always silently discarded.
    """

    # (row_delta, col_delta) offsets in tile-size units covering the full 3×3
    # neighbourhood minus the centre tile (which is already being served).
    _OFFSETS = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1),
    ]

    def __init__(self, workers: int = 2):
        self._executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix='prefetch',
        )
        self._in_flight: set = set()
        self._lock = threading.Lock()

    def _on_done(self, tile_key: str, future) -> None:
        """Callback: remove key from in-flight set and store the result."""
        with self._lock:
            self._in_flight.discard(tile_key)
        try:
            tile_cache.put(tile_key, future.result())
        except Exception:
            pass  # prefetch failures are always silent

    def schedule(
        self,
        *,
        zarr_path: str,
        level: str,
        level_path: str,
        orientation: str,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        layer_height: int,
        layer_width: int,
        indices: dict,
        indices_items: tuple,
        enriched_channels_json,
        has_c_axis: bool,
        channels_json_original,
    ) -> None:
        """Submit prefetch jobs for the 8 tiles surrounding (row/col_start/end)."""
        tile_h = row_end - row_start
        tile_w = col_end - col_start
        if tile_h <= 0 or tile_w <= 0:
            return

        for dr, dc in self._OFFSETS:
            nr_start = row_start + dr * tile_h
            nr_end   = row_end   + dr * tile_h
            nc_start = col_start + dc * tile_w
            nc_end   = col_end   + dc * tile_w

            # Skip if entirely outside layer bounds
            if nr_end <= 0 or nr_start >= layer_height:
                continue
            if nc_end <= 0 or nc_start >= layer_width:
                continue

            # Clamp to layer bounds
            nr_start = max(0, nr_start)
            nr_end   = min(layer_height, nr_end)
            nc_start = max(0, nc_start)
            nc_end   = min(layer_width, nc_end)

            pf_key = TileCache.make_key(
                path=zarr_path, level=level, orientation=orientation,
                rowStart=nr_start, rowEnd=nr_end,
                colStart=nc_start, colEnd=nc_end,
                indices=sorted(indices.items()),
                channels=channels_json_original or '',
            )

            # Skip if already cached or already being prefetched
            if tile_cache.get(pf_key) is not None:
                continue
            with self._lock:
                if pf_key in self._in_flight:
                    continue
                self._in_flight.add(pf_key)

            future = self._executor.submit(
                _render_tile_worker,
                zarr_path, level_path, orientation,
                nr_start, nr_end, nc_start, nc_end,
                indices_items, enriched_channels_json, has_c_axis,
            )
            future.add_done_callback(lambda f, k=pf_key: self._on_done(k, f))


prefetch_engine = PrefetchEngine()


def _render_tile_worker(
    zarr_path, level_path, orientation,
    row_start, row_end, col_start, col_end,
    indices_items, channels_json_enriched, has_c_axis,
):
    """
    Renders a single tile in a worker thread.
    Threads share pyramid_cache directly — no per-worker pyramid caches needed.
    Returns raw PNG bytes.
    """
    pyr = pyramid_cache.get(zarr_path)
    indices = dict(indices_items)

    # Base key for percentile_cache — uniquely identifies the slice position.
    percentile_key_base = (zarr_path, level_path, orientation, indices_items)

    if channels_json_enriched:
        channels_config = json.loads(channels_json_enriched)
        if has_c_axis:
            visible_channels = [ch for ch in channels_config if ch.get('visible', True)]
            if not visible_channels:
                visible_channels = (
                    [channels_config[0]] if channels_config
                    else [{'index': 0, 'color': '#FFFFFF', 'visible': True}]
                )
            all_planes = [
                extract_region(
                    pyr, level_path, orientation,
                    row_start, row_end, col_start, col_end,
                    indices, channel_idx=vc.get('index', 0),
                    zarr_path=zarr_path,
                )
                for vc in visible_channels
            ]
            combined = all_planes[0] if len(all_planes) == 1 else np.stack(all_planes, axis=0)
            rgb = normalize_and_composite(combined, visible_channels,
                                          percentile_key_base=percentile_key_base)
        else:
            tile_data = extract_region(
                pyr, level_path, orientation,
                row_start, row_end, col_start, col_end, indices,
                zarr_path=zarr_path,
            )
            rgb = normalize_and_composite(tile_data, channels_config,
                                          percentile_key_base=percentile_key_base)
    else:
        tile_data = extract_region(
            pyr, level_path, orientation,
            row_start, row_end, col_start, col_end, indices,
            zarr_path=zarr_path,
        )
        rgb = normalize_and_composite(
            tile_data, [{'color': '#FFFFFF', 'visible': True}],
            percentile_key_base=percentile_key_base,
        )

    img = Image.fromarray(rgb, 'RGB')
    buf = io.BytesIO()
    # WebP quality=85 is visually lossless at viewing distances and ~3–4×
    # faster to encode than PNG compress_level=1.  method=0 picks the fastest
    # WebP encoder path (no iterative compression search).
    img.save(buf, format='WEBP', quality=85, method=0)
    return buf.getvalue()


def get_axis_index(axes_str, axis_name):
    axes_lower = axes_str.lower()
    return axes_lower.find(axis_name.lower())


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        return (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16))
    return (255, 255, 255)


ORIENTATIONS = {
    'XY': {'h_axis': 'x', 'v_axis': 'y'},
    'XZ': {'h_axis': 'x', 'v_axis': 'z'},
    'YZ': {'h_axis': 'z', 'v_axis': 'y'},
}


def get_orientation_axes(pyr, level_path, orientation):
    """
    Return axis metadata for the given pyramid / level / orientation.

    Results are memoized by (id(pyr), level_path, orientation).  The
    pyramid_cache keeps strong references so id(pyr) is stable, and
    _invalidate_orientation_cache() cleans up entries on eviction.
    """
    cache_key = (id(pyr), level_path, orientation)
    with _orientation_axes_lock:
        cached = _orientation_axes_cache.get(cache_key)
        if cached is not None:
            return cached

    ori = ORIENTATIONS.get(orientation, ORIENTATIONS['XY'])
    h_axis = ori['h_axis']
    v_axis = ori['v_axis']

    layer = pyr.layers[level_path]
    layer_shape = layer.shape
    axes = pyr.axes.lower()
    base_shape = pyr.shape

    h_idx = axes.find(h_axis)
    v_idx = axes.find(v_axis)
    if h_idx < 0:
        h_idx = len(axes) - 1
    if v_idx < 0:
        v_idx = len(axes) - 2 if len(axes) > 1 else len(axes) - 1

    h_scale = (
        base_shape[h_idx] / layer_shape[h_idx]
        if h_idx < len(base_shape) and h_idx < len(layer_shape) and layer_shape[h_idx] > 0
        else 1.0
    )
    v_scale = (
        base_shape[v_idx] / layer_shape[v_idx]
        if v_idx < len(base_shape) and v_idx < len(layer_shape) and layer_shape[v_idx] > 0
        else 1.0
    )

    layer_height = layer_shape[v_idx] if v_idx < len(layer_shape) else layer_shape[-2]
    layer_width = layer_shape[h_idx] if h_idx < len(layer_shape) else layer_shape[-1]

    chunk_shape = layer.chunks if hasattr(layer, 'chunks') else layer_shape
    chunk_h = chunk_shape[v_idx] if v_idx < len(chunk_shape) else layer_height
    chunk_w = chunk_shape[h_idx] if h_idx < len(chunk_shape) else layer_width

    result = {
        'h_axis': h_axis, 'v_axis': v_axis,
        'h_idx': h_idx, 'v_idx': v_idx,
        'h_scale': h_scale, 'v_scale': v_scale,
        'layer_height': layer_height, 'layer_width': layer_width,
        'chunk_h': chunk_h, 'chunk_w': chunk_w,
        'layer': layer, 'axes': axes, 'base_shape': base_shape, 'layer_shape': layer_shape,
    }

    with _orientation_axes_lock:
        _orientation_axes_cache[cache_key] = result
    return result


def compute_fov_region(info, fov_center, fov_size):
    center_row = int(fov_center[0] / info['v_scale'])
    center_col = int(fov_center[1] / info['h_scale'])
    fov_h = min(fov_size, info['layer_height'])
    fov_w = min(fov_size, info['layer_width'])
    row_start = max(0, min(center_row - fov_h // 2, info['layer_height'] - fov_h))
    row_end = row_start + fov_h
    col_start = max(0, min(center_col - fov_w // 2, info['layer_width'] - fov_w))
    col_end = col_start + fov_w
    return row_start, row_end, col_start, col_end, fov_h, fov_w


def build_slices(info, row_start, row_end, col_start, col_end, indices, channel_idx=None):
    axes = info['axes']
    base_shape = info['base_shape']
    layer_shape = info['layer_shape']
    h_axis = info['h_axis']
    v_axis = info['v_axis']

    slices = []
    for i, ax in enumerate(axes):
        if ax == h_axis:
            slices.append(slice(col_start, col_end))
        elif ax == v_axis:
            slices.append(slice(row_start, row_end))
        elif ax == 'c':
            if channel_idx is not None:
                slices.append(channel_idx)
            else:
                slices.append(slice(None))
        elif ax in indices:
            idx_val = indices[ax]
            scaled_idx = (
                int(idx_val / (base_shape[i] / layer_shape[i]))
                if layer_shape[i] > 0 else 0
            )
            slices.append(max(0, min(scaled_idx, layer_shape[i] - 1)))
        else:
            slices.append(0)
    return slices


def fix_axis_order(data, slices, info):
    axes = info['axes']
    h_axis = info['h_axis']
    v_axis = info['v_axis']
    remaining_axes = [ax for idx_s, ax in enumerate(axes) if not isinstance(slices[idx_s], int)]

    if data.ndim >= 2:
        h_pos = -1
        v_pos = -1
        for ri, rax in enumerate(remaining_axes):
            if rax == h_axis:
                h_pos = ri
            elif rax == v_axis:
                v_pos = ri

        if h_pos >= 0 and v_pos >= 0 and h_pos < v_pos:
            if data.ndim == 2:
                data = data.T
            elif data.ndim == 3:
                c_pos = remaining_axes.index('c') if 'c' in remaining_axes else -1
                if c_pos >= 0:
                    data = np.moveaxis(data, c_pos, 0)
                    spatial = list(range(1, data.ndim))
                    spatial.reverse()
                    data = np.transpose(data, [0] + spatial)
    return data


def extract_region(pyr, level_path, orientation, row_start, row_end, col_start, col_end, indices, channel_idx=None, zarr_path=None):
    info = get_orientation_axes(pyr, level_path, orientation)
    slices = build_slices(info, row_start, row_end, col_start, col_end, indices, channel_idx)

    # TensorStore path: C++ async I/O + GIL-free decompression.
    # Falls back to zarr transparently if ts_cache.get() returns None.
    ts_store = ts_cache.get(zarr_path, level_path) if zarr_path else None
    if ts_store is not None:
        data = np.asarray(ts_store[tuple(slices)].read().result())
    else:
        data = info['layer'][tuple(slices)]
        if hasattr(data, 'compute'):
            data = data.compute()
        data = np.asarray(data)

    data = fix_axis_order(data, slices, info)
    return data


def extract_plane(pyr, level_path, orientation, indices, fov_center, fov_size, channel_idx=None, zarr_path=None):
    info = get_orientation_axes(pyr, level_path, orientation)
    row_start, row_end, col_start, col_end, fov_h, fov_w = compute_fov_region(info, fov_center, fov_size)
    data = extract_region(pyr, level_path, orientation, row_start, row_end, col_start, col_end, indices, channel_idx, zarr_path=zarr_path)

    meta = {
        'level': level_path,
        'fov': [int(row_start), int(row_end), int(col_start), int(col_end)],
        'layerShape': [int(x) for x in info['layer_shape']],
        'scaleFactor': [float(info['v_scale']), float(info['h_scale'])],
    }
    return data, meta


def compute_tile_grid(pyr, level_path, orientation, fov_center, fov_size):
    info = get_orientation_axes(pyr, level_path, orientation)
    row_start, row_end, col_start, col_end, fov_h, fov_w = compute_fov_region(info, fov_center, fov_size)

    chunk_h = info['chunk_h']
    chunk_w = info['chunk_w']
    layer_height = info['layer_height']
    layer_width = info['layer_width']

    first_cr = row_start // chunk_h
    last_cr = (row_end - 1) // chunk_h if row_end > row_start else first_cr
    first_cc = col_start // chunk_w
    last_cc = (col_end - 1) // chunk_w if col_end > col_start else first_cc

    tiles = []
    for cr in range(first_cr, last_cr + 1):
        for cc in range(first_cc, last_cc + 1):
            chunk_row_start = cr * chunk_h
            chunk_row_end = min((cr + 1) * chunk_h, layer_height)
            chunk_col_start = cc * chunk_w
            chunk_col_end = min((cc + 1) * chunk_w, layer_width)

            vis_row_start = max(chunk_row_start, row_start)
            vis_row_end = min(chunk_row_end, row_end)
            vis_col_start = max(chunk_col_start, col_start)
            vis_col_end = min(chunk_col_end, col_end)

            if vis_row_end <= vis_row_start or vis_col_end <= vis_col_start:
                continue

            tiles.append({
                'tileRow': int(cr),
                'tileCol': int(cc),
                'canvasX': int(vis_col_start - col_start),
                'canvasY': int(vis_row_start - row_start),
                'width': int(vis_col_end - vis_col_start),
                'height': int(vis_row_end - vis_row_start),
                'dataRowStart': int(vis_row_start),
                'dataRowEnd': int(vis_row_end),
                'dataColStart': int(vis_col_start),
                'dataColEnd': int(vis_col_end),
            })

    return {
        'canvasWidth': int(fov_w),
        'canvasHeight': int(fov_h),
        'chunkSize': [int(chunk_h), int(chunk_w)],
        'tileCount': len(tiles),
        'tiles': tiles,
        'fovRegion': [int(row_start), int(row_end), int(col_start), int(col_end)],
    }


def normalize_and_composite(plane_data, channels_config, percentile_key_base=None):
    """
    Normalize pixel intensities and composite channels into an RGB uint8 array.

    percentile_key_base: optional tuple (zarr_path, level_path, orientation,
    indices_items) used to cache/retrieve per-slice percentile values so that
    np.percentile is not recomputed on every tile request for the same slice.
    The channel index is appended to this base key per channel.
    """
    if plane_data.ndim == 2:
        ch_cfg = channels_config[0] if channels_config else {
            'color': '#FFFFFF', 'intensityMin': None, 'intensityMax': None,
        }
        data = plane_data.astype(np.float32)
        vmin = ch_cfg.get('intensityMin')
        vmax = ch_cfg.get('intensityMax')

        if vmin is None or vmax is None or vmax <= vmin:
            pct_key = (*percentile_key_base, 0) if percentile_key_base else None
            cached = percentile_cache.get(pct_key) if pct_key else None
            if cached:
                vmin, vmax = cached
            else:
                vmin = float(np.percentile(data, 1))
                vmax = float(np.percentile(data, 99))
                if pct_key:
                    percentile_cache.put(pct_key, vmin, vmax)

        if vmax > vmin:
            data = np.clip((data - vmin) / (vmax - vmin), 0, 1)
        else:
            data = np.zeros_like(data)

        r, g, b = hex_to_rgb(ch_cfg.get('color', '#FFFFFF'))
        h, w = data.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[..., 0] = (data * r).astype(np.uint8)
        rgb[..., 1] = (data * g).astype(np.uint8)
        rgb[..., 2] = (data * b).astype(np.uint8)
        return rgb

    elif plane_data.ndim == 3:
        n_ch = plane_data.shape[0]
        h, w = plane_data.shape[1], plane_data.shape[2]
        rgb = np.zeros((h, w, 3), dtype=np.float32)

        for i in range(n_ch):
            if i >= len(channels_config):
                break
            ch_cfg = channels_config[i]
            if not ch_cfg.get('visible', True):
                continue

            ch_data = plane_data[i].astype(np.float32)
            vmin = ch_cfg.get('intensityMin')
            vmax = ch_cfg.get('intensityMax')

            if vmin is None or vmax is None or vmax <= vmin:
                pct_key = (*percentile_key_base, i) if percentile_key_base else None
                cached = percentile_cache.get(pct_key) if pct_key else None
                if cached:
                    vmin, vmax = cached
                else:
                    vmin = float(np.percentile(ch_data, 1))
                    vmax = float(np.percentile(ch_data, 99))
                    if pct_key:
                        percentile_cache.put(pct_key, vmin, vmax)

            if vmax > vmin:
                ch_data = np.clip((ch_data - vmin) / (vmax - vmin), 0, 1)
            else:
                ch_data = np.zeros_like(ch_data)

            r, g, b = hex_to_rgb(ch_cfg.get('color', '#FFFFFF'))
            rgb[..., 0] += ch_data * (r / 255.0)
            rgb[..., 1] += ch_data * (g / 255.0)
            rgb[..., 2] += ch_data * (b / 255.0)

        rgb = np.clip(rgb, 0, 1)
        return (rgb * 255).astype(np.uint8)

    else:
        flat = plane_data.flatten()
        side = max(1, int(np.sqrt(len(flat))))
        flat = flat[:side * side].reshape(side, side)
        return normalize_and_composite(flat, channels_config,
                                       percentile_key_base=percentile_key_base)


def _enrich_channels_with_cached_minmax(channels_config, zarr_path):
    """Inject cached min/max into channels that have no intensityMin/Max set."""
    enriched = []
    for ch in channels_config:
        ch = dict(ch)
        if ch.get('intensityMin') is None or ch.get('intensityMax') is None:
            cache_key = (zarr_path, ch.get('index', 0))
            with _minmax_lock:
                cached = _minmax_cache.get(cache_key)
            if cached:
                ch['intensityMin'] = cached[0]
                ch['intensityMax'] = cached[1]
        enriched.append(ch)
    return enriched


class ZarrPlaneServer(ThreadingHTTPServer):
    """ThreadingHTTPServer subclass that suppresses client-disconnect noise."""

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, _CLIENT_DISCONNECT_ERRORS):
            return  # client cancelled — not a server error, don't log
        super().handle_error(request, client_address)


class PlaneHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _end_headers_safe(self) -> bool:
        """Call end_headers(), return False if the client already disconnected."""
        try:
            self.end_headers()
            return True
        except _CLIENT_DISCONNECT_ERRORS:
            return False

    def _write_safe(self, data: bytes) -> bool:
        """Write bytes to the socket, return False if the client disconnected."""
        try:
            self.wfile.write(data)
            return True
        except _CLIENT_DISCONNECT_ERRORS:
            return False

    def send_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, If-None-Match')
        self.send_header('Access-Control-Expose-Headers', 'ETag, Content-Length')

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        if not self._end_headers_safe(): return

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        def get_param(name, default=None):
            val = params.get(name, [default])
            return val[0] if val else default

        if parsed.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_cors_headers()
            if not self._end_headers_safe(): return
            self._write_safe(json.dumps({'status': 'ok'}).encode())
            return

        if parsed.path == '/info':
            zarr_path = get_param('path')
            if not zarr_path:
                self.send_error(400, 'Missing path parameter')
                return
            try:
                pyr = pyramid_cache.get(zarr_path)
                axes = pyr.axes.lower()
                shape = list(pyr.shape)
                resolution_paths = sorted(
                    pyr.meta.resolution_paths,
                    key=lambda x: int(x) if x.isdigit() else 0,
                )

                levels_info = []
                for rp in resolution_paths:
                    layer = pyr.layers.get(rp)
                    if layer:
                        levels_info.append({
                            'path': rp,
                            'shape': list(layer.shape),
                            'dtype': str(layer.dtype),
                        })

                channels_meta = (
                    pyr.meta.channels
                    if hasattr(pyr.meta, 'channels') and pyr.meta.channels
                    else []
                )
                channels_list = []
                if channels_meta:
                    for i, ch in enumerate(channels_meta):
                        channels_list.append({
                            'index': i,
                            'label': ch.get('label', f'Channel {i}'),
                            'color': ch.get('color', 'FFFFFF'),
                            'visible': ch.get('active', True),
                            'window': ch.get('window', {
                                'min': 0, 'max': 65535, 'start': 0, 'end': 65535,
                            }),
                        })
                else:
                    channels_list.append({
                        'index': 0,
                        'label': 'Channel 0',
                        'color': 'FFFFFF',
                        'visible': True,
                        'window': {'min': 0, 'max': 255, 'start': 0, 'end': 255},
                    })

                dim_sizes = {ax: shape[i] for i, ax in enumerate(axes)}

                info = {
                    'axes': axes,
                    'shape': shape,
                    'resolutionPaths': resolution_paths,
                    'levelsInfo': levels_info,
                    'numLevels': len(levels_info),
                    'channels': channels_list,
                    'dimSizes': dim_sizes,
                    'dtype': str(pyr.layers[resolution_paths[0]].dtype) if levels_info else 'unknown',
                }

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
                self.send_header('Pragma', 'no-cache')
                self.send_cors_headers()
                if not self._end_headers_safe(): return
                self._write_safe(json.dumps(info).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                if not self._end_headers_safe(): return
                self._write_safe(json.dumps({'error': str(e)}).encode())
            return

        if parsed.path == '/tile_grid':
            zarr_path = get_param('path')
            if not zarr_path:
                self.send_error(400, 'Missing path parameter')
                return
            try:
                pyr = pyramid_cache.get(zarr_path)
                level = get_param('level', '0')
                resolution_paths = sorted(
                    pyr.meta.resolution_paths,
                    key=lambda x: int(x) if x.isdigit() else 0,
                )
                num_levels = len(resolution_paths)
                zoom_level = int(level)
                inverted_idx = max(0, min(num_levels - 1 - zoom_level, num_levels - 1))
                level_path = resolution_paths[inverted_idx] if inverted_idx < len(resolution_paths) else '0'

                orientation = get_param('orientation', 'XY')
                fov_size = int(get_param('fovSize', str(CANVAS_SIZE)))
                fov_center_y = int(get_param(
                    'fovCenterY',
                    str(pyr.shape[-2] // 2 if len(pyr.shape) >= 2 else 0),
                ))
                fov_center_x = int(get_param(
                    'fovCenterX',
                    str(pyr.shape[-1] // 2 if len(pyr.shape) >= 1 else 0),
                ))

                grid = compute_tile_grid(pyr, level_path, orientation,
                                         (fov_center_y, fov_center_x), fov_size)

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
                self.send_cors_headers()
                if not self._end_headers_safe(): return
                self._write_safe(json.dumps(grid).encode())
            except Exception as e:
                import traceback
                sys.stderr.write(f"Error computing tile grid: {e}\n{traceback.format_exc()}\n")
                sys.stderr.flush()
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                if not self._end_headers_safe(): return
                self._write_safe(json.dumps({'error': str(e)}).encode())
            return

        if parsed.path == '/tile':
            zarr_path = get_param('path')
            if not zarr_path:
                self.send_error(400, 'Missing path parameter')
                return
            try:
                pyr = pyramid_cache.get(zarr_path)
                axes = pyr.axes.lower()

                level = get_param('level', '0')
                resolution_paths = sorted(
                    pyr.meta.resolution_paths,
                    key=lambda x: int(x) if x.isdigit() else 0,
                )
                num_levels = len(resolution_paths)
                zoom_level = int(level)
                inverted_idx = max(0, min(num_levels - 1 - zoom_level, num_levels - 1))
                level_path = resolution_paths[inverted_idx] if inverted_idx < len(resolution_paths) else '0'

                orientation = get_param('orientation', 'XY')
                row_start = int(get_param('rowStart', '0'))
                row_end = int(get_param('rowEnd', '0'))
                col_start = int(get_param('colStart', '0'))
                col_end = int(get_param('colEnd', '0'))

                indices = {}
                if 't' in axes:
                    indices['t'] = int(get_param('t', '0'))
                if 'z' in axes and orientation == 'XY':
                    indices['z'] = int(get_param('z', '0'))
                if 'y' in axes and orientation == 'XZ':
                    indices['y'] = int(get_param('sliceIdx', '0'))
                if 'x' in axes and orientation == 'YZ':
                    indices['x'] = int(get_param('sliceIdx', '0'))

                channels_json = get_param('channels')
                has_c_axis = 'c' in axes

                # In-memory tile cache check
                tile_key = TileCache.make_key(
                    path=zarr_path, level=level, orientation=orientation,
                    rowStart=row_start, rowEnd=row_end,
                    colStart=col_start, colEnd=col_end,
                    indices=sorted(indices.items()),
                    channels=channels_json or '',
                )
                cached_tile = tile_cache.get(tile_key)
                if cached_tile:
                    img_data, etag = cached_tile
                    client_etag = self.headers.get('If-None-Match', '')
                    if client_etag == etag:
                        self.send_response(304)
                        self.send_header('ETag', etag)
                        self.send_cors_headers()
                        if not self._end_headers_safe(): return
                        return
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/webp')
                    self.send_header('Content-Length', str(len(img_data)))
                    self.send_header('Cache-Control', 'public, max-age=0')
                    self.send_header('ETag', etag)
                    self.send_cors_headers()
                    if not self._end_headers_safe(): return
                    self._write_safe(img_data)
                    return

                # Enrich channels with cached min/max before dispatch
                enriched_channels_json = channels_json
                if channels_json:
                    ch_cfg = json.loads(channels_json)
                    ch_cfg = _enrich_channels_with_cached_minmax(ch_cfg, zarr_path)
                    enriched_channels_json = json.dumps(ch_cfg)

                future = _get_tile_executor().submit(
                    _render_tile_worker,
                    zarr_path, level_path, orientation,
                    row_start, row_end, col_start, col_end,
                    tuple(sorted(indices.items())),
                    enriched_channels_json,
                    has_c_axis,
                )
                img_data = future.result(timeout=60)

                etag = tile_cache.put(tile_key, img_data)

                # Schedule prefetch of surrounding tiles into tile_cache.
                # get_orientation_axes is memoized — this lookup is free.
                _oa = get_orientation_axes(pyr, level_path, orientation)
                prefetch_engine.schedule(
                    zarr_path=zarr_path,
                    level=level,
                    level_path=level_path,
                    orientation=orientation,
                    row_start=row_start, row_end=row_end,
                    col_start=col_start, col_end=col_end,
                    layer_height=_oa['layer_height'],
                    layer_width=_oa['layer_width'],
                    indices=indices,
                    indices_items=tuple(sorted(indices.items())),
                    enriched_channels_json=enriched_channels_json,
                    has_c_axis=has_c_axis,
                    channels_json_original=channels_json,
                )

                self.send_response(200)
                self.send_header('Content-Type', 'image/webp')
                self.send_header('Content-Length', str(len(img_data)))
                self.send_header('Cache-Control', 'public, max-age=0')
                self.send_header('ETag', etag)
                self.send_cors_headers()
                if not self._end_headers_safe(): return
                self._write_safe(img_data)
            except Exception as e:
                import traceback
                sys.stderr.write(f"Error serving tile: {e}\n{traceback.format_exc()}\n")
                sys.stderr.flush()
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                if not self._end_headers_safe(): return
                self._write_safe(json.dumps({'error': str(e)}).encode())
            return

        if parsed.path == '/channel_minmax':
            zarr_path = get_param('path')
            channel_idx = get_param('channel')
            if not zarr_path or channel_idx is None:
                self.send_error(400, 'Missing path or channel parameter')
                return
            try:
                c_idx = int(channel_idx)

                mm_cache_key = (zarr_path, c_idx)
                with _minmax_lock:
                    cached_mm = _minmax_cache.get(mm_cache_key)
                if cached_mm:
                    result = {'channel': c_idx, 'min': cached_mm[0], 'max': cached_mm[1]}
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Cache-Control', 'no-store')
                    self.send_cors_headers()
                    if not self._end_headers_safe(): return
                    self._write_safe(json.dumps(result).encode())
                    return

                pyr = pyramid_cache.get(zarr_path)
                axes = pyr.axes.lower()
                resolution_paths = sorted(
                    pyr.meta.resolution_paths,
                    key=lambda x: int(x) if x.isdigit() else 0,
                )
                lowest_res_path = resolution_paths[-1]
                layer = pyr.layers[lowest_res_path]

                SAMPLE_SIZE = 512

                if 'c' in axes:
                    c_pos = axes.index('c')
                    slices = []
                    for i, ax in enumerate(axes):
                        if i == c_pos:
                            slices.append(c_idx)
                        elif ax in ('x', 'y'):
                            slices.append(slice(0, min(layer.shape[i], SAMPLE_SIZE)))
                        else:
                            slices.append(slice(None))
                else:
                    if c_idx != 0:
                        self.send_error(400, 'Dataset has no channel axis, only channel 0 valid')
                        return
                    slices = [
                        slice(0, min(layer.shape[i], SAMPLE_SIZE)) if ax in ('x', 'y')
                        else slice(None)
                        for i, ax in enumerate(axes)
                    ]

                ts_store = ts_cache.get(zarr_path, lowest_res_path)
                if ts_store is not None:
                    arr = np.asarray(ts_store[tuple(slices)].read().result())
                else:
                    channel_data = layer[tuple(slices)]
                    if hasattr(channel_data, 'compute'):
                        arr = channel_data.compute()
                    else:
                        arr = np.asarray(channel_data)
                vmin = float(np.nanmin(arr))
                vmax = float(np.nanmax(arr))

                with _minmax_lock:
                    _minmax_cache[mm_cache_key] = (vmin, vmax)

                result = {'channel': c_idx, 'min': vmin, 'max': vmax}
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Cache-Control', 'no-store')
                self.send_cors_headers()
                if not self._end_headers_safe(): return
                self._write_safe(json.dumps(result).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                if not self._end_headers_safe(): return
                self._write_safe(json.dumps({'error': str(e)}).encode())
            return

        if parsed.path == '/plane':
            zarr_path = get_param('path')
            if not zarr_path:
                self.send_error(400, 'Missing path parameter')
                return

            try:
                pyr = pyramid_cache.get(zarr_path)
                axes = pyr.axes.lower()

                level = get_param('level', '0')
                resolution_paths = sorted(
                    pyr.meta.resolution_paths,
                    key=lambda x: int(x) if x.isdigit() else 0,
                )
                num_levels = len(resolution_paths)
                zoom_level = int(level)
                inverted_idx = max(0, min(num_levels - 1 - zoom_level, num_levels - 1))
                level_path = resolution_paths[inverted_idx] if inverted_idx < len(resolution_paths) else '0'

                orientation = get_param('orientation', 'XY')
                fov_size = int(get_param('fovSize', str(CANVAS_SIZE)))
                fov_center_y = int(get_param(
                    'fovCenterY',
                    str(pyr.shape[-2] // 2 if len(pyr.shape) >= 2 else 0),
                ))
                fov_center_x = int(get_param(
                    'fovCenterX',
                    str(pyr.shape[-1] // 2 if len(pyr.shape) >= 1 else 0),
                ))

                indices = {}
                if 't' in axes:
                    indices['t'] = int(get_param('t', '0'))
                if 'z' in axes and orientation == 'XY':
                    indices['z'] = int(get_param('z', '0'))
                if 'y' in axes and orientation == 'XZ':
                    indices['y'] = int(get_param('sliceIdx', '0'))
                if 'x' in axes and orientation == 'YZ':
                    indices['x'] = int(get_param('sliceIdx', '0'))

                channels_json = get_param('channels')
                has_c_axis = 'c' in axes

                percentile_key_base = (
                    zarr_path, level_path, orientation,
                    tuple(sorted(indices.items())),
                )

                if channels_json:
                    channels_config = json.loads(channels_json)
                    if has_c_axis:
                        visible_channels = [ch for ch in channels_config if ch.get('visible', True)]
                        if not visible_channels:
                            visible_channels = (
                                [channels_config[0]] if channels_config
                                else [{'index': 0, 'color': '#FFFFFF', 'visible': True}]
                            )

                        all_planes = []
                        all_configs = []
                        for vc in visible_channels:
                            ch_idx = vc.get('index', 0)
                            plane_data, meta = extract_plane(
                                pyr, level_path, orientation, indices,
                                (fov_center_y, fov_center_x), fov_size,
                                channel_idx=ch_idx,
                                zarr_path=zarr_path,
                            )
                            all_planes.append(plane_data)
                            all_configs.append(vc)

                        combined = all_planes[0] if len(all_planes) == 1 else np.stack(all_planes, axis=0)
                        rgb = normalize_and_composite(combined, all_configs,
                                                      percentile_key_base=percentile_key_base)
                    else:
                        plane_data, meta = extract_plane(
                            pyr, level_path, orientation, indices,
                            (fov_center_y, fov_center_x), fov_size,
                            zarr_path=zarr_path,
                        )
                        rgb = normalize_and_composite(plane_data, channels_config,
                                                      percentile_key_base=percentile_key_base)
                else:
                    plane_data, meta = extract_plane(
                        pyr, level_path, orientation, indices,
                        (fov_center_y, fov_center_x), fov_size,
                        zarr_path=zarr_path,
                    )
                    rgb = normalize_and_composite(
                        plane_data, [{'color': '#FFFFFF', 'visible': True}],
                        percentile_key_base=percentile_key_base,
                    )

                img = Image.fromarray(rgb, 'RGB')
                buf = io.BytesIO()
                img.save(buf, format='WEBP', quality=85, method=0)
                img_data = buf.getvalue()

                self.send_response(200)
                self.send_header('Content-Type', 'image/webp')
                self.send_header('Content-Length', str(len(img_data)))
                self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
                self.send_header('Pragma', 'no-cache')
                self.send_cors_headers()
                if not self._end_headers_safe(): return
                self._write_safe(img_data)
            except Exception as e:
                import traceback
                sys.stderr.write(f"Error serving plane: {e}\n{traceback.format_exc()}\n")
                sys.stderr.flush()
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                if not self._end_headers_safe(): return
                self._write_safe(json.dumps({'error': str(e)}).encode())
            return

        self.send_error(404, 'Not found')


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5555

    server = ZarrPlaneServer(('127.0.0.1', port), PlaneHandler)
    sys.stderr.write(f"Zarr plane server listening on port {port}\n")
    sys.stderr.flush()

    def shutdown(signum, frame):
        sys.stderr.write("Shutting down zarr plane server...\n")
        sys.stderr.flush()
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    print(json.dumps({"type": "ready", "port": port}), flush=True)

    server.serve_forever()


if __name__ == '__main__':
    main()
