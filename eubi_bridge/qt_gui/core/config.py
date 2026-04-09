"""
Config management for the Qt GUI.

Wraps config_manager.py functions directly — no subprocess.
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path

# Add server dir to path so we can import config_manager directly
_SERVER_DIR = os.path.join(os.path.dirname(__file__), "..", "server")
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

import config_manager as _cm  # type: ignore

DEFAULT_CONFIG_DIR = str(Path("~/.eubi_bridge").expanduser())


def load_config(path: str | None = None) -> dict:
    """Load config from *path* (dir or .json file) and return camelCase dict."""
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        _cm.action_get(_cm._resolve_configpath(path))
    return json.loads(buf.getvalue())


def save_config(react_data: dict, path: str | None = None) -> dict:
    """Save *react_data* (camelCase) to *path*. Returns the saved config."""
    import io
    from contextlib import redirect_stdout

    resolved = _cm._resolve_configpath(path) or DEFAULT_CONFIG_DIR
    buf = io.StringIO()
    with redirect_stdout(buf):
        _cm.action_save(resolved, json.dumps(react_data))
    return json.loads(buf.getvalue())


def reset_config(path: str | None = None) -> dict:
    """Reset config at *path* to defaults. Returns the reset config."""
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        _cm.action_reset(_cm._resolve_configpath(path))
    return json.loads(buf.getvalue())


def react_to_snake(react_data: dict) -> dict:
    """Convert camelCase React config to snake_case for EuBIBridge."""
    return _cm._react_to_config(react_data)


def snake_to_react(cfg: dict) -> dict:
    """Convert snake_case EuBIBridge config to camelCase."""
    return _cm._config_to_react(cfg)
