"""
Pytest configuration for EuBI-Bridge tests.
Auto-imports fixtures from conftest_fixtures.py
"""

from .conftest_fixtures import *  # noqa: F401, F403

# Make pytest discover all fixtures from conftest_fixtures
pytest_plugins = []


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "realdata: real-microscopy conversion audit; requires sample files "
        "(EUBI_TEST_DATA dir). Skipped automatically when files are absent.",
    )
