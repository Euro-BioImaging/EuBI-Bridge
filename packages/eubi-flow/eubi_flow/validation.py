"""Pre-flight validation (linting) for EuBI-Bridge flows.

``lint_flow`` resolves the effective parameters for every wave, validates them
against the wave's ``Params`` model, and returns all errors as a flat list.
Nothing is read from or written to disk.

``execute_flow`` calls ``lint_flow`` before touching any OME-Zarr, so a
misconfigured flow fails fast with a clear, actionable message.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eubi_flow.models import FlowSpec


class FlowValidationError(Exception):
    """Raised by ``execute_flow`` when ``lint_flow`` finds parameter errors."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        bullet = "\n".join(f"  • {e}" for e in errors)
        super().__init__(
            f"Flow validation failed with {len(errors)} error(s):\n{bullet}"
        )


def lint_flow(flow: "FlowSpec") -> list[str]:
    """Validate effective parameters for every wave in *flow*.

    Resolves the three-level parameter merge (``add_wave`` params <
    ``configure_engine`` defaults) and validates the result against each
    wave's ``Params`` Pydantic model.

    Returns a (possibly empty) list of human-readable error strings.  Each
    entry is prefixed with the wave id and processor name so the user can
    locate the problem immediately.
    """
    from eubi_flow.registry import get_processor

    errors: list[str] = []
    for wave in flow.waves:
        try:
            processor = get_processor(wave.name)
        except KeyError:
            errors.append(f"[{wave.wave_id}] Unknown processor '{wave.name}'")
            continue

        effective = flow.effective_wave_params(wave)
        for msg in processor.validate_params(effective):
            errors.append(f"[{wave.wave_id} / {wave.name}] {msg}")

    return errors
