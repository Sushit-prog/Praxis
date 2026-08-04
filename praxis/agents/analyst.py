"""Analyst: assess a candidate's novelty and feasibility."""

from __future__ import annotations

from praxis.config import HardwareProfile


def analyze(candidate, config: HardwareProfile | None = None):
    """Extract implementable techniques from a candidate. Filled in later."""
    raise NotImplementedError(
        "Analyst agent is not implemented yet (scaffolding phase)."
    )
