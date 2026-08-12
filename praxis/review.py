"""Human-in-the-loop review gate for borderline candidates (Task #7).

Candidates the Analyst flags as ``borderline`` are held for review instead of
being auto-built. This module powers ``praxis review``: list the pending ones,
approve (which builds them through the normal Architect -> Coder path) or
reject them. Approved candidates are persisted with status ``reviewed`` before
the build, and ``praxis run --resume`` picks up any ``reviewed`` candidate that
was left unbuilt (e.g. an interrupted approval) and continues from the
Architect using the persisted analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from praxis.config import HardwareProfile, load_config
from praxis.db import BuildMemory, Candidate, get_session
from praxis.pipeline import (
    DEFAULT_RETRIES,
    REVIEWED_STATUS,
    _analysis_from_candidate,
    build_from_analysis,
)

logger = logging.getLogger(__name__)

BORDERLINE_STATUS = "borderline"


@dataclass
class ReviewResult:
    """Outcome of a review action on one candidate."""

    candidate_id: int
    title: str
    url: str
    action: str  # "approved" | "rejected"
    status: str
    prototype_path: str | None = None
    error: str | None = None


def _load_candidate(candidate_id: int) -> Candidate | None:
    session = get_session()
    try:
        return session.get(Candidate, candidate_id)
    finally:
        session.close()


def _record_memory(candidate: Candidate, decision: str, outcome: str) -> None:
    """Persist a human review decision so future Analyst scoring can learn from it."""
    session = get_session()
    try:
        session.add(
            BuildMemory(
                candidate_id=candidate.id,
                technique=candidate.technique_summary or "",
                decision=decision,
                outcome=outcome,
            )
        )
        session.commit()
    finally:
        session.close()


def pending_candidates() -> list[Candidate]:
    """Candidates awaiting review (Analyst verdict: borderline)."""
    session = get_session()
    try:
        return list(
            session.scalars(
                select(Candidate)
                .where(Candidate.status == BORDERLINE_STATUS)
                .order_by(Candidate.id)
            ).all()
        )
    finally:
        session.close()


def approve(
    candidate_id: int,
    *,
    config: HardwareProfile | None = None,
    retries: int = DEFAULT_RETRIES,
    scratch_root: Path | None = None,
    timeout: float | None = None,
) -> ReviewResult:
    """Approve a borderline candidate and build it through the normal path."""
    candidate = _load_candidate(candidate_id)
    if candidate is None:
        return ReviewResult(
            candidate_id, "", "", "approved", "missing", error="no such candidate"
        )
    title = candidate.title or ""
    url = candidate.url or ""
    if candidate.status != BORDERLINE_STATUS:
        return ReviewResult(
            candidate_id,
            title,
            url,
            "approved",
            candidate.status,
            error=f"candidate is not awaiting review (status={candidate.status!r})",
        )

    # Record the human decision, then build through Architect -> Coder.
    session = get_session()
    try:
        stored = session.get(Candidate, candidate_id)
        stored.status = REVIEWED_STATUS
        session.commit()
    finally:
        session.close()

    analysis = _analysis_from_candidate(candidate)
    # build_from_analysis handles every non-NotImplementedError failure internally
    # and reports it via BuildOutcome.status; only the pipeline contract re-raises.
    outcome = build_from_analysis(
        candidate,
        analysis,
        config=config or load_config(),
        retries=retries,
        scratch_root=scratch_root,
        timeout=timeout,
    )
    _record_memory(candidate, "approved", outcome.status)

    return ReviewResult(
        candidate_id,
        title,
        url,
        "approved",
        outcome.status,
        prototype_path=outcome.prototype_path,
    )


def reject(candidate_id: int) -> ReviewResult:
    """Reject a borderline candidate; it will not be built."""
    candidate = _load_candidate(candidate_id)
    if candidate is None:
        return ReviewResult(
            candidate_id, "", "", "rejected", "missing", error="no such candidate"
        )
    title = candidate.title or ""
    url = candidate.url or ""
    if candidate.status != BORDERLINE_STATUS:
        return ReviewResult(
            candidate_id,
            title,
            url,
            "rejected",
            candidate.status,
            error=f"candidate is not awaiting review (status={candidate.status!r})",
        )

    session = get_session()
    try:
        stored = session.get(Candidate, candidate_id)
        stored.status = "rejected"
        session.commit()
    finally:
        session.close()
    _record_memory(candidate, "rejected", "rejected")
    return ReviewResult(candidate_id, title, url, "rejected", "rejected")
