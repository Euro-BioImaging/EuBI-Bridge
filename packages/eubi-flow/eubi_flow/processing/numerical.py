"""Numerical / arithmetic wave processors (REGION type).

All processors operate element-wise and return a lazy ``dask.Array``.
No materialisation happens inside ``process()`` — the executor handles writing.

Where a global statistic (min, max) is required (e.g. ``normalize_minmax``),
it is computed eagerly once; the subsequent element-wise scaling stays lazy.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import dask.array as da
from pydantic import BaseModel, ConfigDict, Field, field_validator

from eubi_flow.processing.base import BaseWaveProcessor
from eubi_flow.registry import register_wave


def _as_dask(data) -> da.Array:
    if isinstance(data, da.Array):
        return data
    return da.from_array(np.asarray(data), chunks=np.asarray(data).shape)


# ---------------------------------------------------------------------------
# Arithmetic — scalar operations
# ---------------------------------------------------------------------------

@register_wave
class AddScalarWave(BaseWaveProcessor):
    """Add a constant to every voxel."""

    name = "add_scalar"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        value: float = 0.0

    params_model = Params

    def process(self, data, params: dict) -> da.Array:
        return _as_dask(data) + float(params.get("value", 0.0))


@register_wave
class SubtractScalarWave(BaseWaveProcessor):
    """Subtract a constant from every voxel."""

    name = "subtract_scalar"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        value: float = 0.0

    params_model = Params

    def process(self, data, params: dict) -> da.Array:
        return _as_dask(data) - float(params.get("value", 0.0))


@register_wave
class MultiplyScalarWave(BaseWaveProcessor):
    """Multiply every voxel by a constant."""

    name = "multiply_scalar"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        value: float = 1.0

    params_model = Params

    def process(self, data, params: dict) -> da.Array:
        return _as_dask(data) * float(params.get("value", 1.0))


@register_wave
class DivideScalarWave(BaseWaveProcessor):
    """Divide every voxel by a constant (must be non-zero)."""

    name = "divide_scalar"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        value: float = 1.0

        @field_validator("value")
        @classmethod
        def _nonzero(cls, v: float) -> float:
            if v == 0.0:
                raise ValueError("value must not be zero")
            return v

    params_model = Params

    def process(self, data, params: dict) -> da.Array:
        v = float(params.get("value", 1.0))
        if v == 0.0:
            raise ValueError("divide_scalar: value must not be zero")
        return _as_dask(data) / v


# ---------------------------------------------------------------------------
# Unary math
# ---------------------------------------------------------------------------

@register_wave
class AbsValueWave(BaseWaveProcessor):
    """Absolute value of every voxel."""

    name = "abs_val"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")

    params_model = Params

    def process(self, data, params: dict) -> da.Array:
        return da.absolute(_as_dask(data))


@register_wave
class SqrtWave(BaseWaveProcessor):
    """Element-wise square root (variance-stabilising transform).

    Input is cast to float32 before the operation.
    """

    name = "sqrt"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")

    params_model = Params

    def process(self, data, params: dict) -> da.Array:
        return da.sqrt(_as_dask(data).astype(np.float32))


@register_wave
class SquareWave(BaseWaveProcessor):
    """Element-wise square (x²)."""

    name = "square"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")

    params_model = Params

    def process(self, data, params: dict) -> da.Array:
        arr = _as_dask(data)
        return arr * arr


@register_wave
class PowerWave(BaseWaveProcessor):
    """Raise every voxel to the power ``exponent`` (x^p).

    Input is cast to float32. Useful for gamma correction and
    non-linear contrast adjustment.
    """

    name = "power"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        exponent: float = 2.0

    params_model = Params

    def process(self, data, params: dict) -> da.Array:
        p = float(params.get("exponent", 2.0))
        return da.power(_as_dask(data).astype(np.float32), p)


@register_wave
class Log1pWave(BaseWaveProcessor):
    """Natural log(1 + x) transform.

    Safe for zero-valued pixels; useful for compressing high dynamic-range
    fluorescence intensities before display or further processing.
    Input is cast to float32.
    """

    name = "log1p"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")

    params_model = Params

    def process(self, data, params: dict) -> da.Array:
        return da.log1p(_as_dask(data).astype(np.float32))


@register_wave
class ExpWave(BaseWaveProcessor):
    """Element-wise e^x (inverse of log1p after subtracting 1).

    Input is cast to float32.
    """

    name = "exp"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")

    params_model = Params

    def process(self, data, params: dict) -> da.Array:
        return da.exp(_as_dask(data).astype(np.float32))


# ---------------------------------------------------------------------------
# Clip
# ---------------------------------------------------------------------------

@register_wave
class ClipWave(BaseWaveProcessor):
    """Clamp voxel values to [min_val, max_val].

    Values below ``min_val`` are set to ``min_val``;
    values above ``max_val`` are set to ``max_val``.
    Useful for removing saturated pixels or outliers before further analysis.
    """

    name = "clip"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        min_val: float = 0.0
        max_val: float = 65535.0

    params_model = Params

    def process(self, data, params: dict) -> da.Array:
        lo = float(params.get("min_val", 0.0))
        hi = float(params.get("max_val", 65535.0))
        return da.clip(_as_dask(data), lo, hi)


# ---------------------------------------------------------------------------
# Normalise
# ---------------------------------------------------------------------------

@register_wave
class NormalizeMinMaxWave(BaseWaveProcessor):
    """Rescale intensities linearly to [out_min, out_max].

    The global minimum and maximum are computed eagerly from the full array;
    the subsequent linear scaling is a lazy dask operation.
    Output is float32.
    """

    name = "normalize_minmax"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        out_min: float = 0.0
        out_max: float = 1.0

    params_model = Params

    def process(self, data, params: dict) -> da.Array:
        out_min = float(params.get("out_min", 0.0))
        out_max = float(params.get("out_max", 1.0))
        arr = _as_dask(data).astype(np.float32)
        # arr.min() / arr.max() return 0-d lazy dask arrays — no .compute() needed.
        arr_min = arr.min()
        arr_max = arr.max()
        denom = arr_max - arr_min
        # Avoid division by zero for constant arrays; result is out_min either way.
        safe_denom = da.where(denom > 0.0, denom, np.float32(1.0))
        return (arr - arr_min) / safe_denom * (out_max - out_min) + out_min


# ---------------------------------------------------------------------------
# Invert
# ---------------------------------------------------------------------------

@register_wave
class InvertWave(BaseWaveProcessor):
    """Invert intensities: ``out = max_val - x``.

    When ``max_val`` is -1 (the default), it is auto-detected:
    integer arrays use ``np.iinfo(dtype).max``; float arrays use 1.0.
    Set ``max_val`` explicitly for e.g. 8-bit images stored as float32.
    """

    name = "invert"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        max_val: float = Field(-1.0, description="-1 = auto-detect from dtype")

    params_model = Params

    def process(self, data, params: dict) -> da.Array:
        arr = _as_dask(data)
        max_val = float(params.get("max_val", -1.0))
        if max_val < 0:
            if np.issubdtype(arr.dtype, np.integer):
                max_val = float(np.iinfo(arr.dtype).max)
            else:
                max_val = 1.0
        return max_val - arr


# ---------------------------------------------------------------------------
# Cast dtype
# ---------------------------------------------------------------------------

_DTYPE = Literal["float32", "float64", "uint8", "uint16", "uint32", "int16", "int32"]


@register_wave
class CastDtypeWave(BaseWaveProcessor):
    """Cast the array to a different numpy dtype.

    No rescaling is performed — values are truncated/wrapped when converting
    from a wider to a narrower type.  Combine with ``clip`` or
    ``normalize_minmax`` beforehand if rescaling is needed.
    """

    name = "cast_dtype"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        dtype: _DTYPE = "float32"

    params_model = Params

    def process(self, data, params: dict) -> da.Array:
        dtype = params.get("dtype", "float32")
        return _as_dask(data).astype(np.dtype(dtype))
