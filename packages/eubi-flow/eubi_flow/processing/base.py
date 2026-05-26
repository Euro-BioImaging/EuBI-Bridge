"""Base classes and shared utilities for wave processors."""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Optional, Type, Union

import numpy as np
from pydantic import BaseModel, ValidationError

try:
    import dask.array as da
    _ArrayLike = Union[np.ndarray, da.Array]
except ImportError:  # pragma: no cover
    _ArrayLike = np.ndarray  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Axis constants
# ---------------------------------------------------------------------------

AXES = "tczyx"
AXIS_INDEX: dict[str, int] = {ax: i for i, ax in enumerate(AXES)}


# ---------------------------------------------------------------------------
# Per-axis parameter helper
# ---------------------------------------------------------------------------

def axis_param(params: dict, key: str, default: float) -> tuple[float, ...]:
    """Build a per-axis tuple from axis-prefixed entries in params.

    Resolution order for each axis ``ax``:
    1. ``params[f"{ax}_{key}"]``  — axis-specific value
    2. ``params[key]``            — uniform fallback
    3. ``default``                — hard default

    Examples
    --------
    >>> axis_param({"z_sigma": 2.0, "y_sigma": 1.0}, "sigma", 0.0)
    (0.0, 0.0, 2.0, 1.0, 0.0)   # tczyx
    >>> axis_param({"sigma": 1.5}, "sigma", 0.0)
    (1.5, 1.5, 1.5, 1.5, 1.5)
    """
    uniform = params.get(key, default)
    return tuple(float(params.get(f"{ax}_{key}", uniform)) for ax in AXES)


# ---------------------------------------------------------------------------
# Wave type
# ---------------------------------------------------------------------------

class WaveType(str, Enum):
    REGION     = "region"       # shape in == shape out; all axes partitioned freely
    REDUCTIVE  = "reductive"    # reduces one or more axes; reduced axes need full extent
    GENERATIVE = "generative"   # adds one or more axes (fan-in)


# ---------------------------------------------------------------------------
# Abstract base processor
# ---------------------------------------------------------------------------

class BaseWaveProcessor(ABC):
    """Abstract base for all wave processors.

    Subclasses must define:
    - ``name`` (class attribute, str): registry key
    - ``process(data, params)``: apply operation to a 5D numpy array

    Subclasses should override the output descriptor methods when the wave
    changes array dimensionality (REDUCTIVE / GENERATIVE types).
    """

    name: str
    use_gpu: bool = False          # future cupy hook
    params_model: Optional[Type[BaseModel]] = None  # set per-processor to enable validation

    # ── wave type & partitioning ──────────────────────────────────────────

    def wave_type(self) -> WaveType:
        return WaveType.REGION

    def reduce_axes(self, params: dict) -> list[str]:
        """Axes collapsed by this wave (REDUCTIVE only)."""
        return []

    def partitioned_axes(self, input_axes: str, params: dict) -> list[str]:
        """Axes the executor may split into spatial regions.

        For REDUCTIVE waves the reduced axes must span their full extent so
        they are excluded from partitioning.
        """
        if self.wave_type() == WaveType.REDUCTIVE:
            excluded = set(self.reduce_axes(params))
            return [ax for ax in input_axes if ax not in excluded]
        return list(input_axes)

    # ── output metadata descriptors ──────────────────────────────────────

    def output_axes(self, input_axes: str, params: dict) -> str:
        """Axis order string for the output array. Default: unchanged."""
        return input_axes

    def output_shape(self, input_shape: tuple[int, ...], params: dict) -> tuple[int, ...]:
        """Shape of the output base-level array. Default: unchanged."""
        return input_shape

    def output_scales(self, input_scales: dict, params: dict) -> dict:
        """Physical pixel sizes for the output axes. Default: unchanged."""
        return dict(input_scales)

    def output_units(self, input_units: dict, params: dict) -> dict:
        """Physical units for the output axes. Default: unchanged."""
        return dict(input_units)

    # ── overlap ──────────────────────────────────────────────────────────

    def overlap(self, params: dict) -> tuple[int, ...]:
        """Overlap in pixels on each side, one value per axis in tczyx order.

        The executor reads (region + overlap) from the source heave, calls
        process(), then trims the overlap to obtain the canonical output.
        Default: no overlap (point-wise operations).
        """
        return (0, 0, 0, 0, 0)

    # ── parameter validation ─────────────────────────────────────────────

    def validate_params(self, params: dict) -> list[str]:
        """Validate *params* against this processor's ``params_model``.

        Returns a (possibly empty) list of human-readable error strings.
        An empty list means all parameters are valid.
        """
        if self.params_model is None:
            return []
        try:
            self.params_model(**params)
            return []
        except ValidationError as exc:
            return [
                f"{'->'.join(str(x) for x in e['loc'])}: {e['msg']}"
                for e in exc.errors()
            ]

    # ── core operation ───────────────────────────────────────────────────

    @abstractmethod
    def process(self, data: "_ArrayLike", params: dict) -> "_ArrayLike":
        """Apply the operation to a padded 5D array (tczyx).

        Input may be a numpy or dask array; return type should match.
        REGION waves must return the same shape as ``data``.
        REDUCTIVE waves return an array with the reduced axis removed.
        Overlap trimming is handled by the executor — do not trim inside here.
        """

    # ── GPU stubs ─────────────────────────────────────────────────────────

    def _to_device(self, arr: np.ndarray) -> np.ndarray:
        return arr

    def _from_device(self, arr: np.ndarray) -> np.ndarray:
        return arr
