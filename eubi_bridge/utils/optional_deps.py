"""The standard-only stack -- dask, distributed, the Bio-Formats / bioio
readers, the JVM -- imported only where it is used.

eubi-bridge-lite installs none of it, so no module of eubi_bridge may import
it at load time (tests/test_lite_core.py).  Code that needs it imports it
where it runs, through ``require``; code that only asks whether an object
belongs to it asks without importing it (``is_dask_array``); code that does
something else without it asks first (``is_installed``).  In a lite run,
these are the only places that may try to import it: the lite tests fail on
any other attempt -- a feature lost silently.
"""
from __future__ import annotations

import contextlib
import importlib
import importlib.util
import sys
from types import ModuleType


def is_installed(module: str) -> bool:
    """Whether *module* can be imported (without importing it)."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def is_dask_array(obj) -> bool:
    """Whether *obj* is a dask array, without importing dask: if dask.array
    was never imported, no dask array can exist."""
    da = sys.modules.get("dask.array")
    return da is not None and isinstance(obj, da.Array)


def dask_config(**settings):
    """``dask.config.set(**settings)`` when dask is installed; without dask
    there is no dask scheduler to configure, so nothing."""
    if not is_installed("dask"):
        return contextlib.nullcontext()
    import dask
    return dask.config.set(**settings)


def require(module: str, feature: str) -> ModuleType:
    """Import *module* for *feature*; if it is not installed, say that the
    feature needs the full eubi-bridge."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        package = module.partition(".")[0]
        raise ImportError(
            f"{feature} needs {package}, which eubi-bridge-lite does not include. "
            f"Install the full eubi-bridge: pip install eubi-bridge") from exc
