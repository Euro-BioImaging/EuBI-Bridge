"""Reshape / axis-manipulation wave processors (GENERATIVE type).

These waves change the number of axes without performing any pixel-level
computation.  NGFF metadata is updated accordingly.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from eubi_flow.processing.base import BaseWaveProcessor, WaveType
from eubi_flow.registry import register_wave


@register_wave
class NewAxisWave(BaseWaveProcessor):
    """Insert a new singleton axis into the array (np.expand_dims).

    This is the simplest GENERATIVE wave — it adds a length-1 axis at a
    given position and records the corresponding physical scale and unit.

    Parameters
    ----------
    axis : str, default "t"
        The name of the new axis to insert (e.g. "t", "c").
    position : int, default 0
        The index (0-based) at which to insert the new axis.
    scale : float, default 1.0
        Physical pixel size for the new axis.
    unit : str, default ""
        Physical unit for the new axis (e.g. "second").
    """

    name = "new_axis"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        axis: Literal["t", "c", "z", "y", "x"] = "t"
        position: int = Field(0, ge=0)
        scale: float = Field(1.0, gt=0.0)
        unit: str = ""

    params_model = Params

    def wave_type(self) -> WaveType:
        return WaveType.GENERATIVE

    def output_axes(self, input_axes: str, params: dict) -> str:
        ax = params.get("axis", "t")
        pos = int(params.get("position", 0))
        axes = list(input_axes)
        axes.insert(pos, ax)
        return "".join(axes)

    def output_shape(self, input_shape: tuple[int, ...], params: dict) -> tuple[int, ...]:
        pos = int(params.get("position", 0))
        return input_shape[:pos] + (1,) + input_shape[pos:]

    def output_scales(self, input_scales: dict, params: dict) -> dict:
        ax = params.get("axis", "t")
        scale = float(params.get("scale", 1.0))
        return {ax: scale, **input_scales}

    def output_units(self, input_units: dict, params: dict) -> dict:
        ax = params.get("axis", "t")
        unit = str(params.get("unit", ""))
        return {ax: unit, **input_units}

    def overlap(self, params: dict) -> tuple[int, ...]:
        # Output has one more axis than input; return zeros for each output axis
        n_out = 5  # always expanding a 5D input → 6D is unusual; keep 5D convention
        return (0,) * n_out

    def process(self, data: np.ndarray, params: dict) -> np.ndarray:
        pos = int(params.get("position", 0))
        return np.expand_dims(data, axis=pos)
