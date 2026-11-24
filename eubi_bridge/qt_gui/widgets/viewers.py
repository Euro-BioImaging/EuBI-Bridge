"""Which viewer the Inspect page shows: the GPU viewer (micro-glancer) or the classic one.

The choice comes from the environment variable ``EUBI_VIEWER`` ("gpu" or "classic")
when it is set, else from the settings ("Viewer" in Settings: "auto", "gpu" or
"classic"). "auto" takes the GPU viewer when micro-glancer is installed. When the GPU
viewer is chosen but micro-glancer cannot be imported, or the GPU viewer cannot be built,
the classic viewer is used and the reason is kept in ``fallback_reason()``.

micro-glancer is in development (Workplan WP14) and not on PyPI yet: to try it, put its
``python`` folder (built with ``maturin develop``) on ``PYTHONPATH``, or install it into
eubi's environment.
"""
from __future__ import annotations

import os

_reason = ""


def gpu_available() -> tuple[bool, str]:
    """Whether micro-glancer can be imported: (True, its version) or (False, why not)."""
    try:
        import micro_glancer
        import micro_glancer.qt  # noqa: F401 (the widget needs PyQt6 and the core)
    except Exception as exc:  # noqa: BLE001 (any failure means: not available)
        return False, f"micro-glancer cannot be imported: {exc}"
    return True, getattr(micro_glancer, "__version__", "?")


def viewer_choice() -> str:
    """ "gpu" or "classic" (see the module's notes)."""
    wanted = os.environ.get("EUBI_VIEWER", "").strip().lower()
    if wanted not in ("gpu", "classic"):
        try:
            from eubi_bridge.qt_gui.settings_dialog import current_viewer
            wanted = current_viewer()
        except Exception:  # noqa: BLE001
            wanted = "auto"
    if wanted == "classic":
        return "classic"
    ok, _ = gpu_available()
    return "gpu" if ok else "classic"


def fallback_reason() -> str:
    """Why the classic viewer stands in for the GPU viewer that was asked for ("" when
    it does not)."""
    return _reason


def make_viewer(parent=None, **classic_kwargs):
    """The Inspect page's viewer: the GPU viewer or eubi's classic ``ZarrViewer`` (built
    with `classic_kwargs`), as chosen (see the module's notes)."""
    global _reason
    from eubi_bridge.qt_gui.widgets.zarr_viewer import ZarrViewer
    wanted = os.environ.get("EUBI_VIEWER", "").strip().lower()
    choice = viewer_choice()
    _reason = ""
    if choice == "gpu":
        try:
            from eubi_bridge.qt_gui.widgets.gpu_viewer import GpuZarrViewer
            return GpuZarrViewer(parent)
        except Exception as exc:  # noqa: BLE001 (the classic viewer stands in)
            _reason = f"the GPU viewer could not be built: {exc}"
    elif wanted == "gpu":
        _reason = gpu_available()[1]
    return ZarrViewer(parent, **classic_kwargs)
