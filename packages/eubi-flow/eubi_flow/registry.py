"""Wave processor registry.

Usage
-----
Register a processor with the @register_wave class decorator::

    @register_wave
    class GaussianFilterWave(BaseWaveProcessor):
        name = "gaussian_filter"
        ...

Look up a processor by name::

    proc = get_processor("gaussian_filter")
    result = proc.process(data, params)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eubi_flow.processing.base import BaseWaveProcessor

_WAVE_REGISTRY: dict[str, type["BaseWaveProcessor"]] = {}


def register_wave(cls: type) -> type:
    """Class decorator — registers cls under its .name attribute."""
    if not hasattr(cls, "name") or not cls.name:
        raise TypeError(f"Wave class {cls.__qualname__} must define a non-empty 'name'.")
    _WAVE_REGISTRY[cls.name] = cls
    return cls


def get_processor(name: str) -> "BaseWaveProcessor":
    """Return a fresh instance of the processor registered under name."""
    # Ensure all processors are imported so decorators have fired
    import eubi_flow.processing  # noqa: F401
    if name not in _WAVE_REGISTRY:
        raise KeyError(
            f"Unknown wave '{name}'. "
            f"Available: {sorted(_WAVE_REGISTRY)}"
        )
    return _WAVE_REGISTRY[name]()


def list_processors() -> dict[str, type["BaseWaveProcessor"]]:
    """Return the full registry dict (name → class)."""
    import eubi_flow.processing  # noqa: F401
    return dict(_WAVE_REGISTRY)
