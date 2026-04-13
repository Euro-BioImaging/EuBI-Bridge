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


def _compute_minmax(
    zarr_path: str,
    channel_idx: int,
    level_path: str,
    orientation: str,
    indices: dict,
    fov_center: tuple,
    fov_size: int,
) -> tuple[float, float]:
    """
    Extract exactly the visible region (current level, frame, viewport) for
    *channel_idx* and return (p1, p99).
    """
    from zarr_plane_server import extract_plane  # type: ignore

    pyr = pyramid_cache.get(zarr_path)
    data, _meta = extract_plane(
        pyr, level_path, orientation, indices, fov_center, fov_size,
        channel_idx=channel_idx, zarr_path=zarr_path,
    )
    if hasattr(data, "compute"):
        data = data.compute()
    data = np.asarray(data, dtype=np.float32).ravel()

    if data.size == 0:
        return 0.0, 1.0

    p1  = float(np.percentile(data, 1))
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

    def __init__(
        self,
        zarr_path: str,
        channel_idx: int,
        level_path: str,
        orientation: str,
        indices: dict,
        fov_center: tuple,
        fov_size: int,
        parent=None,
    ):
        super().__init__(parent)
        self._zarr_path   = zarr_path
        self._channel_idx = channel_idx
        self._level_path  = level_path
        self._orientation = orientation
        self._indices     = indices
        self._fov_center  = fov_center
        self._fov_size    = fov_size

    def run(self):
        try:
            vmin, vmax = _compute_minmax(
                self._zarr_path, self._channel_idx,
                self._level_path, self._orientation,
                self._indices, self._fov_center, self._fov_size,
            )
            self.result.emit(self._channel_idx, vmin, vmax)
        except Exception as exc:
            self.failed.emit(self._channel_idx, str(exc))
