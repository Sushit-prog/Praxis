"""Orchestrates the 4 agents sequentially with retry/backoff."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from praxis import agents
from praxis.config import HardwareProfile, load_config

logger = logging.getLogger(__name__)

DEFAULT_RETRIES = 3
BACKOFF_BASE_S = 1.0


def run_with_retry(
    fn: Callable[..., Any],
    retries: int = DEFAULT_RETRIES,
    base_backoff: float = BACKOFF_BASE_S,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Call fn, retrying on transient errors with exponential backoff."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError:
            raise
        except Exception as exc:  # noqa: BLE001 - retry on any transient failure
            last_error = exc
            if attempt < retries - 1:
                delay = base_backoff * (2**attempt)
                logger.warning(
                    "Attempt %d/%d failed (%s); retrying in %.1fs",
                    attempt + 1,
                    retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
    raise RuntimeError(f"All {retries} attempts failed") from last_error


def run_pipeline(
    source: str,
    topic: str,
    config: HardwareProfile | None = None,
    limit: int = 20,
    retries: int = DEFAULT_RETRIES,
) -> Any:
    """Run Scout -> Analyst -> Architect -> Coder on a topic."""
    config = config or load_config()
    candidates = run_with_retry(agents.scout, retries, source=source, topic=topic, limit=limit)
    result = candidates
    for step in (agents.analyze, agents.architect, agents.coder):
        result = run_with_retry(step, retries, result, config)
    return result
