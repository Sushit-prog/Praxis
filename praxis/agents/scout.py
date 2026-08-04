"""Scout: discover candidates from a source (arxiv/github/hn)."""

from __future__ import annotations

from praxis.config import HardwareProfile


def scout(source: str, topic: str, config: HardwareProfile | None = None):
    """Discover candidate items matching the topic. Filled in later."""
    raise NotImplementedError("Scout agent is not implemented yet (scaffolding phase).")
