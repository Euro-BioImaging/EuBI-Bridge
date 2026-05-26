"""Thresholding wave processors (REGION type, no overlap).

All thresholding operations are point-wise; no overlap is needed.
Output dtype is always ``uint8`` (0 = below threshold, 255 = above).

Laziness model
--------------
Every threshold scalar (Otsu histogram, percentile) is wrapped in
``dask.delayed`` + ``da.from_delayed``, producing a 0-d dask node that is
broadcast lazily against the data array.  Nothing is materialised until the
executor writes the result to disk.

``process()`` accepts numpy or dask arrays and always returns a ``dask.Array``.
"""
from __future__ import annotations

import numpy as np
import dask
import dask.array as da
from pydantic import BaseModel, ConfigDict, Field

from eubi_flow.processing.base import BaseWaveProcessor
from eubi_flow.registry import register_wave


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_dask(data) -> da.Array:
    """Wrap numpy (or keep dask) as a single-chunk dask array."""
    if isinstance(data, da.Array):
        return data
    return da.from_array(data, chunks=data.shape)


def _apply_mask(arr: da.Array, threshold: float, above: bool) -> da.Array:
    """Lazy comparison + cast to uint8 (0 / 255)."""
    mask = arr > threshold if above else arr < threshold
    return (mask.astype(np.uint8) * 255)


# ---------------------------------------------------------------------------
# Processors
# ---------------------------------------------------------------------------

@register_wave
class FixedThresholdWave(BaseWaveProcessor):
    """Apply a fixed intensity threshold.

    Parameters
    ----------
    threshold : float
        The threshold value.
    above : bool, default True
        If True, pixels above the threshold are foreground (255).
        If False, pixels below the threshold are foreground (255).
    """

    name = "threshold_fixed"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        threshold: float          # required — no default
        above: bool = True

    params_model = Params

    def process(self, data, params: dict) -> da.Array:
        if "threshold" not in params:
            raise ValueError(
                "threshold_fixed requires a 'threshold' parameter. "
                "Set it with: eubi flow configure_engine FLOW WAVE_ID --threshold VALUE"
            )
        threshold = float(params["threshold"])
        above     = bool(params.get("above", True))
        arr = _as_dask(data)
        return _apply_mask(arr, threshold, above)


@register_wave
class OtsuThresholdWave(BaseWaveProcessor):
    """Otsu's method: find threshold that minimises intra-class variance.

    The global histogram and threshold are computed as a lazy node in the
    dask graph (``dask.delayed`` + ``da.from_delayed``) and only materialised
    when the executor writes the result — no eager ``.compute()`` call.

    Parameters
    ----------
    nbins : int, default 256
        Number of histogram bins used to compute the threshold.
    above : bool, default True
        If True, pixels above the Otsu threshold are foreground.
    """

    name = "threshold_otsu"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        nbins: int = Field(256, gt=0)
        above: bool = True

    params_model = Params

    def process(self, data, params: dict) -> da.Array:
        nbins = int(params.get("nbins", 256))
        above = bool(params.get("above", True))
        arr   = _as_dask(data)

        # Build a 0-d dask node for the Otsu threshold.
        # dask.delayed automatically computes dask-array arguments before
        # calling the function, so _otsu_from_arr receives a numpy array.
        threshold_delayed = dask.delayed(_otsu_from_arr)(arr, nbins)
        threshold_da = da.from_delayed(
            threshold_delayed, shape=(), dtype=np.float64
        )
        return _apply_mask(arr, threshold_da, above)


@register_wave
class PercentileThresholdWave(BaseWaveProcessor):
    """Threshold at a given percentile of the intensity distribution.

    The percentile scalar is computed as a lazy node in the dask graph and
    only materialised when the executor writes the result.

    Parameters
    ----------
    percentile : float, default 95.0
        Percentile value (0–100).
    above : bool, default True
        If True, pixels above the percentile threshold are foreground.
    """

    name = "threshold_percentile"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        percentile: float = Field(95.0, ge=0.0, le=100.0)
        above: bool = True

    params_model = Params

    def process(self, data, params: dict) -> da.Array:
        pct   = float(params.get("percentile", 95.0))
        above = bool(params.get("above", True))
        arr   = _as_dask(data)

        threshold_delayed = dask.delayed(_percentile_from_arr)(arr, pct)
        threshold_da = da.from_delayed(
            threshold_delayed, shape=(), dtype=np.float64
        )
        return _apply_mask(arr, threshold_da, above)


# ---------------------------------------------------------------------------
# Helpers called inside dask.delayed (receive numpy arrays)
# ---------------------------------------------------------------------------

def _otsu_from_arr(arr: np.ndarray, nbins: int) -> np.float64:
    """Compute the Otsu threshold from a numpy array."""
    counts, bin_edges = np.histogram(arr.ravel(), bins=nbins)
    return np.float64(_otsu_from_hist(counts, bin_edges))


def _percentile_from_arr(arr: np.ndarray, pct: float) -> np.float64:
    """Compute a percentile threshold from a numpy array."""
    return np.float64(np.percentile(arr.ravel(), pct))


def _otsu_from_hist(counts: np.ndarray, bin_edges: np.ndarray) -> float:
    """Compute the Otsu threshold from a pre-computed histogram."""
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = int(counts.sum())
    if total == 0:
        return float(bin_centers[len(bin_centers) // 2])

    weight1      = np.cumsum(counts).astype(np.float64)
    weight2      = total - weight1
    cumsum_vals  = np.cumsum(counts * bin_centers)
    mean1        = cumsum_vals / np.where(weight1 > 0, weight1, 1.0)
    mean_total   = cumsum_vals[-1] / total
    mean2        = np.where(
        weight2 > 0,
        (mean_total * total - cumsum_vals) / np.where(weight2 > 0, weight2, 1.0),
        0.0,
    )
    variance_between = weight1 * weight2 * (mean1 - mean2) ** 2
    return float(bin_centers[int(np.argmax(variance_between))])
