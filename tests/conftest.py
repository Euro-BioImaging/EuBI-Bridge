"""
Pytest configuration for EuBI-Bridge tests.
Auto-imports fixtures from conftest_fixtures.py
"""

from .conftest_fixtures import *  # noqa: F401, F403

# Make pytest discover all fixtures from conftest_fixtures
pytest_plugins = []
