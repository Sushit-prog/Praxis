"""Interactive discover flow: scout + analyze, then pick a candidate to design.

`praxis discover --topic T` runs the Scout and Analyst only (no Architect),
prints a numbered decision table, and either prompts for a pick (interactive)
or takes `--pick N` (non-interactive). The picked candidate's analysis is used
directly by `praxis design`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from praxis.config import HardwareProfile
from praxis.db import Candidate, get_session

logger = logging.getLogger(__name__)

VERDICT_ACCEPT = "accept"
VERDICT_BORDERLINE = "borderline"
VERDICT_REJECT = "reject"


@dataclass
class DiscoverRow:
    """One row of the discover decision table."""

    index: int
    candidate: Candidate
    score: int | None
    technique: str
    fit_reason: str
    verdict: str


@dataclass
class DiscoverResult:
    """Outcome of a discover run."""

    topic: str
    source: str
    discovered: int = 0
    analyzed: int = 0
    rejected: int = 0
    rows: list[DiscoverRow] = field(default_factory=list)

    @property
    def pickable(self) -> list[DiscoverRow]:
        """Rows a human might pick (accepted or borderline; not rejects)."""
        return [r for r in self.rows if r.verdict != VERDICT_REJECT]


def _verdict_of(candidate: Candidate, threshold: int, margin: int) -> str:
    score = candidate.feasibility_score
    if score is None or score < threshold:
        return VERDICT_REJECT
    if score <= min(threshold + margin, 10):
        return VERDICT_BORDERLINE
    return VERDICT_ACCEPT


def _fit_reason(candidate: Candidate, profile: HardwareProfile) -> str:
    """One-line hardware-fit reason: the Analyst's reasoning, compressed."""
    text = (candidate.feasibility_reasoning or "").strip().replace("\n", " ")
    if len(text) > 110:
        text = text[:107].rstrip() + "..."
    return text or "no reasoning recorded"


def analyze_candidates(
    candidates: list[Candidate],
    profile: HardwareProfile,
    *,
    threshold: int | None = None,
    margin: int | None = None,
    retries: int = 3,
) -> list[DiscoverRow]:
    """Analyze candidates (persisting verdicts) and build the decision table."""
    from praxis import agents
    from praxis.agents.analyst import _resolve_margin, _resolve_threshold
    from praxis.pipeline import FAILED_STATUS, _mark_failed

    threshold = _resolve_threshold(threshold)
    margin = _resolve_margin(margin)

    rows: list[DiscoverRow] = []
    for index, candidate in enumerate(candidates, start=1):
        url = getattr(candidate, "url", "")
        try:
            agents.analyze(candidate, profile, threshold=threshold, margin=margin)
        except NotImplementedError:
            raise
        except Exception as exc:  # noqa: BLE001 - isolate candidate failures
            logger.warning("analyst failed for %s: %s", url, exc)
            _mark_failed(getattr(candidate, "id", None))
            continue

        session = get_session()
        try:
            stored = session.get(Candidate, candidate.id)
        finally:
            session.close()
        candidate = stored or candidate
        if getattr(candidate, "status", "") == FAILED_STATUS:
            continue

        verdict = _verdict_of(candidate, threshold, margin)
        rows.append(
            DiscoverRow(
                index=index,
                candidate=candidate,
                score=candidate.feasibility_score,
                technique=(candidate.technique_summary or "").strip(),
                fit_reason=_fit_reason(candidate, profile),
                verdict=verdict,
            )
        )
    return rows


def format_table(result: DiscoverResult) -> str:
    """Render the numbered decision table."""
    lines = [
        f"Discovered {result.discovered} candidate(s) for {result.topic!r} "
        f"from {result.source} — analyzed {result.analyzed}, "
        f"rejected {result.rejected}.",
        "",
        "  #  score  verdict      technique                        source",
        "  " + "-" * 96,
    ]
    for row in result.pickable:
        score = str(row.score) if row.score is not None else "?"
        title = (row.candidate.title or "").strip()
        if len(title) > 52:
            title = title[:49].rstrip() + "..."
        lines.append(
            f"  {row.index:>2}  {score:>5}  {row.verdict:<12} {title:<52} "
            f"{row.candidate.source}"
        )
        lines.append(f"      build: {row.technique}")
        lines.append(f"      fit:   {row.fit_reason}")
    if not result.pickable:
        lines.append("  (nothing pickable — everything was rejected or failed)")
    return "\n".join(lines)


def discover(
    source: str,
    topic: str,
    limit: int = 10,
    *,
    profile: HardwareProfile | None = None,
    retries: int = 3,
) -> DiscoverResult:
    """Run Scout + Analyst for a topic and return the decision table."""
    from praxis import agents
    from praxis.config import load_config
    from praxis.pipeline import run_with_retry

    profile = profile or load_config()
    result = DiscoverResult(topic=topic, source=source)

    candidates = run_with_retry(agents.scout, retries, source=source, topic=topic, limit=limit)
    result.discovered = len(candidates)
    if not candidates:
        return result

    rows = analyze_candidates(candidates, profile, retries=retries)
    result.rows = rows
    result.analyzed = len(rows)
    result.rejected = sum(1 for r in rows if r.verdict == VERDICT_REJECT)
    return result


def get_candidate(candidate_id: int) -> Candidate | None:
    """Load a persisted candidate by id."""
    session = get_session()
    try:
        return session.get(Candidate, candidate_id)
    finally:
        session.close()


def prompt_pick(result: DiscoverResult, input_fn=None) -> DiscoverRow | None:
    """Prompt 'Pick a number (or q)' and return the chosen row (or None).

    Only pickable rows are selectable; rejects and failed analyses are not
    offered. EOF/keyboard-interrupt count as 'q'. ``input_fn`` defaults to
    builtins.input resolved at call time (so tests can patch it).
    """
    if input_fn is None:
        input_fn = input
    pickable = result.pickable
    if not pickable:
        return None
    while True:
        try:
            raw = input_fn("Pick a number (or q): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if raw.lower() in ("q", "quit", "exit", ""):
            return None
        if raw.isdigit():
            num = int(raw)
            for row in pickable:
                if row.index == num:
                    return row
        print(f"  invalid pick: {raw!r} (choose a # from the table, or q)")


def focus_note(row: DiscoverRow, input_fn=None) -> str | None:
    """Ask for an optional focus note for the picked candidate."""
    if input_fn is None:
        input_fn = input
    print(f"\nPicked #{row.index}: {row.candidate.title}")
    print(f"  technique: {row.technique}")
    try:
        note = input_fn("Focus note (optional, Enter to skip): ").strip()
    except (EOFError, KeyboardInterrupt):
        note = ""
    return note or None
