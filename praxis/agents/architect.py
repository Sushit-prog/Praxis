"""Architect: produce a hardware-calibrated engineering blueprint."""

from __future__ import annotations

from praxis.config import HardwareProfile


def architect(analysis, config: HardwareProfile | None = None):
    """Draft a blueprint calibrated to the target hardware. Filled in later."""
    raise NotImplementedError(
        "Architect agent is not implemented yet (scaffolding phase)."
    )
