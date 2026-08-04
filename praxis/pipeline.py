"""Orchestrates the 4 agents sequentially with retry/backoff."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from praxis import agents
from praxis.config import HardwareProfile, load_config

logger = logging.getLogger(__name__)

DEFAULT_RETRIES = 3
BACKOFF_BASE_S = 1.0


def run_with_retry(
    fn: Callable[..., Any],
    retries: int = DEFAULT_RETRIES,
    *,
    base_backoff: float = BACKOFF_BASE_S,
    **kwargs: Any,
) -> Any:
    """Call fn(*args, **kwargs), retrying on transient errors with backoff."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return fn(**kwargs)
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
    scratch_root: Path | None = None,
    timeout: float | None = None,
) -> Any:
    """Run Scout -> Analyst -> Architect -> Coder on a topic."""
    config = config or load_config()
    candidates = run_with_retry(agents.scout, retries, source=source, topic=topic, limit=limit)

    accepted = []
    for candidate in candidates:
        result = run_with_retry(agents.analyze, retries, candidate=candidate, profile=config)
        if not result.rejected:
            accepted.append((candidate, result))

    blueprints = []
    for candidate, analysis in accepted:
        bp = run_with_retry(
            agents.architect,
            retries,
            candidate=candidate,
            analysis=analysis,
            profile=config,
        )
        blueprints.append(bp)

    prototypes = []
    for bp in blueprints:
        path = run_with_retry(
            agents.coder,
            retries,
            blueprint=bp,
            scratch_root=scratch_root,
            timeout=timeout,
        )
        prototypes.append(path)
    return prototypes
