"""Orchestrates the 4 agents sequentially with retry/backoff.

The pipeline runs Scout -> Analyst -> Architect -> Coder over a batch of
candidates. Individual candidates that are rejected or fail at any stage are
skipped; a single bad candidate never aborts the rest of the batch.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select

from praxis import agents
from praxis.agents.analyst import AnalysisResult
from praxis.config import HardwareProfile, load_config
from praxis.db import Candidate, UsageTotals, get_session, usage_totals

logger = logging.getLogger(__name__)

DEFAULT_RETRIES = 3
BACKOFF_BASE_S = 1.0

FAILED_STATUS = "failed"
REVIEWED_STATUS = "reviewed"


def run_with_retry(
    fn: Callable[..., Any],
    retries: int = DEFAULT_RETRIES,
    *,
    base_backoff: float = BACKOFF_BASE_S,
    **kwargs: Any,
) -> Any:
    """Call fn(**kwargs), retrying on transient errors with backoff."""
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


@dataclass
class CandidateOutcome:
    """Final outcome for a single candidate in a pipeline run."""

    title: str
    url: str
    status: str
    prototype_path: str | None = None


@dataclass
class BuildOutcome:
    """Result of running the Architect + Coder stages for one candidate."""

    blueprinted: bool
    status: str
    prototype_path: str | None = None


@dataclass
class PipelineResult:
    """Batch summary of a pipeline run."""

    source: str
    topic: str
    discovered: int = 0
    analyzed: int = 0
    rejected: int = 0
    borderline: int = 0
    blueprinted: int = 0
    prototyped: int = 0
    failed: int = 0
    resumed: int = 0
    usage_calls: int = 0
    usage_total_tokens: int = 0
    usage_cost_usd: float = 0.0
    usage_cached_hits: int = 0
    candidates: list[CandidateOutcome] = field(default_factory=list)


def _mark_failed(candidate_id: int | None) -> None:
    """Persist a generic failure status for a candidate."""
    if candidate_id is None:
        return
    session = get_session()
    try:
        stored = session.get(Candidate, candidate_id)
        if stored is not None:
            stored.status = FAILED_STATUS
            session.commit()
    finally:
        session.close()


def _unfinished_candidates() -> list[Candidate]:
    """Candidates from earlier runs that still have work: status new, failed, or reviewed."""
    session = get_session()
    try:
        return list(
            session.scalars(
                select(Candidate).where(
                    Candidate.status.in_(("new", FAILED_STATUS, REVIEWED_STATUS))
                )
            ).all()
        )
    finally:
        session.close()


def _analysis_from_candidate(candidate: Candidate) -> AnalysisResult:
    """Reconstruct an AnalysisResult from the Analyst's persisted verdict columns."""
    return AnalysisResult(
        technique_summary=candidate.technique_summary or "",
        feasibility_score=(
            candidate.feasibility_score if candidate.feasibility_score is not None else 0
        ),
        feasibility_reasoning=candidate.feasibility_reasoning or "",
        rejected=False,
    )


def build_from_analysis(
    candidate: Candidate,
    analysis: AnalysisResult,
    *,
    config: HardwareProfile,
    retries: int = DEFAULT_RETRIES,
    scratch_root: Path | None = None,
    timeout: float | None = None,
) -> BuildOutcome:
    """Run the Architect and Coder stages for an analyzed candidate.

    Shared by the pipeline loop and the human review gate (`praxis review
    approve`), so an approved candidate takes exactly the same build path as a
    freshly accepted one.
    """
    url = getattr(candidate, "url", "") or ""
    try:
        blueprint = run_with_retry(
            agents.architect, retries, candidate=candidate, analysis=analysis, profile=config
        )
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001 - isolate candidate failures
        logger.warning("architect failed for %s: %s", url, exc)
        _mark_failed(candidate.id)
        return BuildOutcome(blueprinted=False, status=FAILED_STATUS)

    logger.info("architect blueprinted %s", url)
    try:
        path = run_with_retry(
            agents.coder,
            retries,
            blueprint=blueprint,
            scratch_root=scratch_root,
            timeout=timeout,
        )
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001 - isolate candidate failures
        logger.warning("coder failed for %s: %s", url, exc)
        _mark_failed(candidate.id)
        return BuildOutcome(blueprinted=True, status=FAILED_STATUS)

    if path is None:
        logger.warning("coder failed for %s (opencode non-zero or timeout)", url)
        return BuildOutcome(blueprinted=True, status="prototype_failed")

    logger.info("coder prototyped %s -> %s", url, path)
    return BuildOutcome(blueprinted=True, status="prototyped", prototype_path=str(path))


def _snapshot_usage() -> UsageTotals | None:
    """Snapshot recorded LLM usage, tolerating a missing/old usage table."""
    session = get_session()
    try:
        return usage_totals(session=session)
    except Exception as exc:  # noqa: BLE001 - observability must never break a run
        logger.warning("usage snapshot failed: %s", exc)
        return None
    finally:
        session.close()


def run(
    source: str,
    topic: str,
    limit: int = 20,
    *,
    config: HardwareProfile | None = None,
    retries: int = DEFAULT_RETRIES,
    scratch_root: Path | None = None,
    timeout: float | None = None,
    resume: bool = False,
) -> PipelineResult:
    """Run Scout -> Analyst -> Architect -> Coder over a batch of candidates.

    With ``resume=True``, candidates left in status ``new`` or ``failed`` by
    earlier runs are processed alongside the freshly scouted ones, so an
    interrupted batch can continue instead of restarting from scratch. Scout
    failures degrade to the resumed candidates rather than aborting.
    """
    config = config or load_config()
    result = PipelineResult(source=source, topic=topic)
    usage_before = _snapshot_usage()

    candidates: list[Any] = []
    if resume:
        candidates = _unfinished_candidates()
        result.resumed = len(candidates)
        logger.info("resume: picked up %d unfinished candidate(s)", len(candidates))

    try:
        new_candidates = run_with_retry(
            agents.scout, retries, source=source, topic=topic, limit=limit
        )
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001 - scout failure aborts, unless there is work to resume
        logger.warning("scout failed for topic=%r: %s", topic, exc)
        if not candidates:
            return result
        logger.warning("scout failed; continuing with %d resumed candidate(s)", len(candidates))
    else:
        candidates.extend(new_candidates)
        result.discovered = len(new_candidates)
        logger.info("scout: discovered %d new candidate(s)", len(new_candidates))

    for candidate in candidates:
        url = getattr(candidate, "url", "")
        title = getattr(candidate, "title", "") or url
        candidate_id = getattr(candidate, "id", None)

        if getattr(candidate, "status", "") == REVIEWED_STATUS:
            # Human-approved candidate: continue from the Architect using the
            # persisted analysis; the Analyst stage is skipped.
            analysis = _analysis_from_candidate(candidate)
        else:
            try:
                analysis = run_with_retry(
                    agents.analyze, retries, candidate=candidate, profile=config
                )
            except NotImplementedError:
                raise
            except Exception as exc:  # noqa: BLE001 - isolate candidate failures
                logger.warning("analyst failed for %s: %s", url, exc)
                _mark_failed(candidate_id)
                result.failed += 1
                result.candidates.append(
                    CandidateOutcome(title=title, url=url, status=FAILED_STATUS)
                )
                continue

            if analysis.rejected:
                logger.info("analyst rejected %s", url)
                result.rejected += 1
                result.candidates.append(
                    CandidateOutcome(title=title, url=url, status="rejected")
                )
                continue

            if analysis.borderline:
                # Confidence-aware routing: scores inside the threshold band are
                # held for review rather than auto-built or silently rejected.
                logger.info("analyst flagged %s as borderline", url)
                result.borderline += 1
                result.candidates.append(
                    CandidateOutcome(title=title, url=url, status="borderline")
                )
                continue

            result.analyzed += 1
            logger.info("analyst accepted %s", url)

        outcome = build_from_analysis(
            candidate,
            analysis,
            config=config,
            retries=retries,
            scratch_root=scratch_root,
            timeout=timeout,
        )
        if outcome.blueprinted:
            result.blueprinted += 1
        if outcome.status == "prototyped":
            result.prototyped += 1
            logger.info("coder prototyped %s -> %s", url, outcome.prototype_path)
        else:
            result.failed += 1
            logger.warning("build ended %s for %s", outcome.status, url)
        result.candidates.append(
            CandidateOutcome(
                title=title,
                url=url,
                status=outcome.status,
                prototype_path=outcome.prototype_path,
            )
        )

    usage_after = _snapshot_usage()
    if usage_before is not None and usage_after is not None:
        result.usage_calls = usage_after.calls - usage_before.calls
        result.usage_total_tokens = usage_after.total_tokens - usage_before.total_tokens
        result.usage_cost_usd = max(0.0, usage_after.cost_usd - usage_before.cost_usd)
        result.usage_cached_hits = max(0, usage_after.cached_hits - usage_before.cached_hits)
    elif usage_before is not None or usage_after is not None:
        logger.warning(
            "usage snapshot incomplete (before=%s after=%s); spend footer skipped",
            usage_before is not None,
            usage_after is not None,
        )

    return result


def format_summary(result: PipelineResult) -> str:
    """Render a human-readable pipeline summary."""
    rows = [
        ("discovered", result.discovered),
        ("analyzed", result.analyzed),
        ("rejected", result.rejected),
        ("borderline", result.borderline),
        ("blueprinted", result.blueprinted),
        ("prototyped", result.prototyped),
        ("failed", result.failed),
    ]
    lines = [f"Summary for topic={result.topic!r} source={result.source}"]
    lines.extend(f"  {label}: {value}" for label, value in rows)
    if result.resumed:
        lines.append(f"  resumed: {result.resumed}")
    if result.usage_calls:
        cache_note = f" ({result.usage_cached_hits} from cache)" if result.usage_cached_hits else ""
        lines.append(
            f"  LLM spend: ${result.usage_cost_usd:.4f} across {result.usage_calls} calls"
            f"{cache_note} ({result.usage_total_tokens:,} tokens)"
        )
    if result.candidates:
        lines.append("Candidates:")
        for outcome in result.candidates:
            suffix = f" ({outcome.prototype_path})" if outcome.prototype_path else ""
            lines.append(f"  - {outcome.title} [{outcome.status}]{suffix}")
    return "\n".join(lines)
