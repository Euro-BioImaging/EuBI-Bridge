"""
Async zarr plane rendering worker.

Imports rendering functions directly from zarr_plane_server — no HTTP.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

# ── Import rendering functions from zarr_plane_server ────────────────────────
_SERVER_DIR = os.path.join(os.path.dirname(__file__), "..", "server")
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

from zarr_plane_server import (  # type: ignore
    pyramid_cache,
    extract_plane,
    normalize_and_composite,
    ORIENTATIONS,
    hex_to_rgb,
)


def do_render(params: dict, executor: ThreadPoolExecutor) -> np.ndarray:
    """Render a single 2-D plane and return an RGB uint8 array.

    Orientation-agnostic: fov_center=(height_center, width_center) and fov_size
    apply identically for XY, XZ, and YZ.  extract_plane maps them to the
    correct physical axes via get_orientation_axes.
    """
    pyr = pyramid_cache.get(params["path"])
    level_path = pyr.meta.resolution_paths[params["level_idx"]]
    axes = pyr.axes.lower()
    has_c = "c" in axes
    channels_config = params["channels"]
    # Stable key for percentile caching — avoids recomputing np.percentile on every pan frame
    pct_key = (params["path"], params["level_idx"], params["orientation"])

    if has_c and len(channels_config) > 1:
        visible = [ch for ch in channels_config if ch.get("visible", True)]
        if not visible:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        results: list = [None] * len(visible)

        def fetch(i, vc):
            results[i] = extract_plane(
                pyr,
                level_path,
                params["orientation"],
                params["indices"],
                params["fov_center"],
                params["fov_size"],
                channel_idx=vc["index"],
                zarr_path=params["path"],
            )[0]

        futs = [executor.submit(fetch, i, vc) for i, vc in enumerate(visible)]
        for f in futs:
            f.result()
        combined = results[0] if len(results) == 1 else np.stack(results, axis=0)
        return normalize_and_composite(combined, visible, percentile_key_base=pct_key)

    plane_data, _ = extract_plane(
        pyr,
        level_path,
        params["orientation"],
        params["indices"],
        params["fov_center"],
        params["fov_size"],
        zarr_path=params["path"],
    )
    return normalize_and_composite(plane_data, channels_config or [{"color": "#FFFFFF", "visible": True}],
                                   percentile_key_base=pct_key)


class RenderWorker(QThread):
    """Background thread that renders zarr planes on demand.

    Call ``request(params)`` from the main thread; the worker renders the
    plane and emits ``frame_ready(rgb_array, elapsed_ms)``.  A new request
    always supersedes the previous pending one.
    """

    frame_ready = pyqtSignal(object, float, int)   # (np.ndarray, elapsed_ms, generation)
    render_error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cond = threading.Condition()
        self._pending: dict | None = None
        self._stop = False
        self._generation: int = 0          # monotonic request counter

    def request(self, params: dict):
        """Queue *params* for rendering (replaces any pending request)."""
        with self._cond:
            self._generation += 1
            params["_generation"] = self._generation
            self._pending = params
            self._cond.notify()

    def stop(self):
        with self._cond:
            self._stop = True
            self._cond.notify()

    def run(self):
        n_workers = min(8, (os.cpu_count() or 2) * 2)
        executor = ThreadPoolExecutor(max_workers=n_workers)
        try:
            while True:
                with self._cond:
                    while self._pending is None and not self._stop:
                        self._cond.wait(timeout=0.1)
                    if self._stop:
                        break
                    params = self._pending
                    self._pending = None

                try:
                    gen = params.get("_generation", 0)
                    t0 = time.perf_counter()
                    rgb = do_render(params, executor)
                    elapsed = (time.perf_counter() - t0) * 1000
                    # If the caller requested a coarse-level preview, upsample
                    # back to target_fov_size via nearest-neighbour index mapping
                    # so the displayed frame always fills the same canvas area.
                    target = params.get("target_fov_size", 0)
                    if target:
                        h, w = rgb.shape[:2]
                        if h != target or w != target:
                            row_idx = np.round(np.linspace(0, h - 1, target)).astype(np.intp)
                            col_idx = np.round(np.linspace(0, w - 1, target)).astype(np.intp)
                            rgb = rgb[np.ix_(row_idx, col_idx)]
                    self.frame_ready.emit(rgb, elapsed, gen)
                except Exception as exc:
                    self.render_error.emit(str(exc))
        finally:
            executor.shutdown(wait=False)
