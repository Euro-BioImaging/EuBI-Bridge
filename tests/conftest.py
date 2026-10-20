"""
Pytest configuration for EuBI-Bridge tests.
Auto-imports fixtures from conftest_fixtures.py
"""

import os
import tempfile

# Every test -- and every `eubi` subprocess a test starts -- uses a fresh
# config folder with the installation defaults, never the developer's own
# ~/.eubi_bridge: a local run then sees what CI sees, and no test writes into
# the real config.  (Set before eubi_bridge is imported; a test that needs a
# particular config passes `configpath=` itself.)
os.environ["EUBI_CONFIG_DIR"] = tempfile.mkdtemp(prefix="eubi_test_config_")

import pytest  # noqa: E402

# Before qt_available() below imports PyQt6: importing eubi_bridge loads the
# system's C++ runtime first on Windows, so Qt's old copy cannot crash the
# later native readers (see eubi_bridge/__init__.py, test_windows_cpp_runtime).
import eubi_bridge  # noqa: E402, F401

from .conftest_fixtures import *  # noqa: F401, F403, E402

# Resolved once, lazily, by qt_available() below.
_QT_AVAILABLE = None

# Make pytest discover all fixtures from conftest_fixtures
pytest_plugins = []


def qt_available() -> bool:
    """True when PyQt6 can actually be imported and a QApplication built.

    ``pytest.importorskip("PyQt6.QtWidgets")`` is not enough: on a headless
    Linux runner the package is installed but importing it raises ImportError
    for a missing system library (``libEGL.so.1``), which aborts collection
    rather than skipping.  Qt also needs an offscreen platform plugin there, so
    that is set before the import is attempted.
    """
    global _QT_AVAILABLE
    if _QT_AVAILABLE is None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PyQt6.QtWidgets import QApplication
            QApplication.instance() or QApplication([])
            _QT_AVAILABLE = True
        except Exception:            # ImportError, or Qt failing to initialise
            _QT_AVAILABLE = False
    return _QT_AVAILABLE


#: Decorator for tests and modules that need a working Qt GUI stack.
requires_qt = pytest.mark.skipif(
    not qt_available(),
    reason="PyQt6 unavailable or no usable Qt platform plugin",
)


def full_installed() -> bool:
    """The full eubi-bridge's stack (Bio-Formats / bioio, dask) is installed:
    False in eubi-bridge-lite."""
    from eubi_bridge.utils.capabilities import has_bioformats, has_dask
    return has_bioformats() and has_dask()


#: For tests of what only the full eubi-bridge has (Bio-Formats metadata, the
#: standard pixel readers, eubi's own dask readers, the JVM), or that use it as
#: a reference: they skip in eubi-bridge-lite, whose CI job runs the whole
#: suite against the installed lite wheel.
requires_full = pytest.mark.skipif(
    not full_installed(),
    reason="needs the full eubi-bridge (Bio-Formats / dask), not eubi-bridge-lite",
)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "realdata: real-microscopy conversion audit; requires sample files "
        "(MICROSCOPY_DATA dir). Skipped automatically when files are absent.",
    )
