"""Sidecar canvas-position persistence for the flow builder.

Stores node x/y positions in ``~/.eubi_bridge/.flows/{name}_layout.json``.
Never touches the FlowSpec JSON — the data model stays clean.
"""
from __future__ import annotations

import json
from pathlib import Path


def _flows_dir() -> Path:
    from eubi_flow.eubiflow import _flows_dir as _wd
    return _wd()


class LayoutStore:
    """Read/write ``{name}_layout.json`` alongside the flow JSON files."""

    def load(self, flow_name: str) -> dict[str, tuple[float, float]]:
        """Return ``{node_id: (x, y)}`` or ``{}`` if no sidecar exists."""
        path = _flows_dir() / f"{flow_name}_layout.json"
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {k: (float(v[0]), float(v[1])) for k, v in data.items()}
        except Exception:
            return {}

    def save(self, flow_name: str, positions: dict[str, tuple[float, float]]) -> None:
        """Persist ``positions`` to the sidecar file."""
        path = _flows_dir() / f"{flow_name}_layout.json"
        path.write_text(
            json.dumps({k: [v[0], v[1]] for k, v in positions.items()}, indent=2),
            encoding="utf-8",
        )
