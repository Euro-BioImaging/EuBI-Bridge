"""
Per-channel min/max worker — computes 1st/99th percentile for histogram auto-adjustment.

Ports the /channel_minmax endpoint logic from zarr_plane_server directly.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

_SERVER_DIR = os.path.join(os.path.dirname(__file__), "..", "server")
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

from zarr_plane_server import pyramid_cache  # type: ignore


def _compute_minmax(zarr_path: str, channel_idx: int) -> tuple[float, float]:
    """
    Read a sample plane from the coarsest pyramid level for *channel_idx* and
    return (p1, p99) — mirroring the /channel_minmax endpoint.
    """
    pyr = pyramid_cache.get(zarr_path)
    axes = pyr.axes.lower()

    # Use the coarsest level (last resolution path = smallest array)
    level_path = pyr.meta.resolution_paths[-1]
    layer = pyr.layers[level_path]
    shape = layer.shape

    c_idx_in_axes = axes.find("c")
    z_idx_in_axes = axes.find("z")
    t_idx_in_axes = axes.find("t")

    slices = []
    for i, ax in enumerate(axes):
        if ax == "c":
            slices.append(channel_idx if c_idx_in_axes >= 0 else slice(None))
        elif ax == "t":
            slices.append(shape[i] // 2)
        elif ax == "z":
            slices.append(shape[i] // 2)
        else:
            slices.append(slice(None))  # full y/x plane

    data = layer[tuple(slices)]
    if hasattr(data, "compute"):
        data = data.compute()
    data = np.asarray(data, dtype=np.float32)

    if data.size == 0:
        return 0.0, 1.0

    p1 = float(np.percentile(data, 1))
    p99 = float(np.percentile(data, 99))
    if p99 <= p1:
        p99 = p1 + 1.0
    return p1, p99


class MinMaxWorker(QThread):
    """Computes per-channel intensity range in a background thread.

    Signals:
        result(channel_idx, vmin, vmax)
        failed(channel_idx, error_message)
    """

    result = pyqtSignal(int, float, float)
    failed = pyqtSignal(int, str)

    def __init__(self, zarr_path: str, channel_idx: int, parent=None):
        super().__init__(parent)
        self._zarr_path = zarr_path
        self._channel_idx = channel_idx

    def run(self):
        try:
            vmin, vmax = _compute_minmax(self._zarr_path, self._channel_idx)
            self.result.emit(self._channel_idx, vmin, vmax)
        except Exception as exc:
            self.failed.emit(self._channel_idx, str(exc))
