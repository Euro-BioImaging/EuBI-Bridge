"""JSON serialisation for FlowSpec."""
from __future__ import annotations

from pathlib import Path

from eubi_flow.models import FlowSpec


def save_flow(flow: FlowSpec, path: str | Path) -> None:
    """Write a flow to a JSON file (pretty-printed, UTF-8)."""
    Path(path).write_text(flow.model_dump_json(indent=2), encoding="utf-8")


def load_flow(path: str | Path) -> FlowSpec:
    """Load a flow from a JSON file."""
    return FlowSpec.model_validate_json(Path(path).read_text(encoding="utf-8"))
