"""
Pytest configuration for EuBI-Bridge tests.
Auto-imports fixtures from conftest_fixtures.py
"""

import os

import pytest

from .conftest_fixtures import *  # noqa: F401, F403

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


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "realdata: real-microscopy conversion audit; requires sample files "
        "(EUBI_TEST_DATA dir). Skipped automatically when files are absent.",
    )
