"""Reductive wave processors — collapse one axis (REDUCTIVE type).

All processors accept numpy or dask arrays and return a **lazy ``dask.Array``**.
The reduction operation itself (``da.max``, ``da.mean``, ``da.sum``) is lazy;
materialisation happens only when the executor writes to disk.

The reduced axis must span its full extent in each processing region so the
executor will never partition along it (see ``partitioned_axes``).

NGFF metadata (axes, scales, units, shape) is automatically updated:
the reduced axis is removed from every metadata dict.
"""
from __future__ import annotations

import numpy as np
import dask.array as da
from pydantic import BaseModel, ConfigDict
from typing import Literal

from eubi_flow.processing.base import (
    AXIS_INDEX,
    BaseWaveProcessor,
    WaveType,
)
from eubi_flow.registry import register_wave


def _as_dask(data) -> da.Array:
    if isinstance(data, da.Array):
        return data
    return da.from_array(np.asarray(data), chunks=np.asarray(data).shape)


class _ProjectionBase(BaseWaveProcessor):
    """Shared logic for all projection processors."""

    _da_method: str = "max"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        axis: Literal["t", "c", "z", "y", "x"] = "z"

    params_model = Params

    def wave_type(self) -> WaveType:
        return WaveType.REDUCTIVE

    def reduce_axes(self, params: dict) -> list[str]:
        return [params.get("axis", "z")]

    def output_axes(self, input_axes: str, params: dict) -> str:
        ax = params.get("axis", "z")
        return input_axes.replace(ax, "")

    def output_shape(self, input_shape: tuple[int, ...], params: dict) -> tuple[int, ...]:
        ax  = params.get("axis", "z")
        idx = AXIS_INDEX[ax]
        return input_shape[:idx] + input_shape[idx + 1:]

    def output_scales(self, input_scales: dict, params: dict) -> dict:
        ax = params.get("axis", "z")
        return {k: v for k, v in input_scales.items() if k != ax}

    def output_units(self, input_units: dict, params: dict) -> dict:
        ax = params.get("axis", "z")
        return {k: v for k, v in input_units.items() if k != ax}

    def overlap(self, params: dict) -> tuple[int, ...]:
        return (0, 0, 0, 0, 0)

    def process(self, data, params: dict) -> da.Array:
        arr = _as_dask(data)
        ax  = params.get("axis", "z")
        idx = AXIS_INDEX[ax]
        fn  = getattr(da, self._da_method)
        return fn(arr, axis=idx)   # lazy dask reduction


@register_wave
class MaxProjectionWave(_ProjectionBase):
    """Maximum-intensity projection along a named axis.

    Parameters
    ----------
    axis : str, default "z"
        The axis to collapse (one of t, c, z, y, x).
    """

    name       = "max_projection"
    _da_method = "max"


@register_wave
class MeanProjectionWave(_ProjectionBase):
    """Mean-intensity projection along a named axis.

    Parameters
    ----------
    axis : str, default "z"
        The axis to collapse.
    """

    name       = "mean_projection"
    _da_method = "mean"


@register_wave
class SumProjectionWave(_ProjectionBase):
    """Sum projection along a named axis.

    Parameters
    ----------
    axis : str, default "z"
        The axis to collapse.
    """

    name       = "sum_projection"
    _da_method = "sum"
