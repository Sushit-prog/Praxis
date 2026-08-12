"""Golden-set evaluation harness for the Analyst and Architect agents.

Runs the agents against a fixed, hand-labeled set of candidate fixtures and
reports how their outputs compare to the expected verdicts, score bands, and
blueprint structure. Blueprint quality is judged by a deterministic rubric so
the harness stays stable and cheap to run; the LLM output is the only
non-deterministic part.

Design notes:
- ``analyze``/``draft_blueprint`` are called exactly as the pipeline calls
  them, but every agent callable is injectable so the harness logic can be
  tested without a live model.
- ``praxis eval`` runs against a throwaway SQLite database, so evaluating
  never pollutes the real ledger.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from praxis.agents.analyst import AnalysisResult, analyze
from praxis.agents.architect import draft_blueprint
from praxis.config import HardwareProfile
from praxis.db import Candidate

logger = logging.getLogger(__name__)

FIXTURES_RELATIVE_PATH = Path("tests") / "fixtures" / "golden_candidates.json"

REQUIRED_SECTIONS = (
    "Problem Statement",
    "Proposed Architecture",
    "Phased Build Plan",
    "Deferred to Later Versions",
    "Difficulty & Time Estimate",
)

_MILESTONE_RE = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)
_SUBITEM_RE = re.compile(r"^\s*[-*]\s", re.MULTILINE)
_GPU_PATTERNS = (
    re.compile(r"\b(cuda|tpu)\b", re.IGNORECASE),
    re.compile(r"\bgpu\s+(cluster|farm|server|array)\b", re.IGNORECASE),
    re.compile(r"\bgpu\b", re.IGNORECASE),
)
_NEGATOR_RE = re.compile(
    r"\b(no|not|without|never|none|avoids?|exclud(?:e|es|ing)|"
    r"don'?t|doesn'?t|won'?t|unnecessary|not required|unavailable)\b",
    re.IGNORECASE,
)
_DEFERRAL_RE = re.compile(
    r"\b(deferred|later|future|next version|not implemented|out of scope|v2)\b",
    re.IGNORECASE,
)
_RAM_REQ_RE = re.compile(
    r"(requires?\b|needs?\b|demands?\b|must have\b)\s*.{0,40}?(\d+)\s*GB\b",
    re.IGNORECASE,
)
_HARDWARE_TERM_RE = re.compile(r"\b(cpu|ram|gpu|gb|memory|budget|hardware)\b", re.IGNORECASE)

# Negation/deferral words only suppress a hardware flag when they are close to
# the token itself; a wider window lets an unrelated "no GPU needed" sentence
# hide a real "requires CUDA" requirement two sentences later. RAM needs an even
# tighter window: "no GPU needed. Requires 32GB" puts "no" only ~14 chars from
# the requirement keyword, and that negation belongs to a different clause.
_GPU_GUARD_BEFORE = 30
_GPU_GUARD_AFTER = 25
_RAM_GUARD_BEFORE = 12


# --------------------------------------------------------------------------
# Golden set
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldenFixture:
    """A hand-labeled candidate whose expected outcome is known in advance."""

    id: str
    source: str
    url: str
    title: str
    raw_text: str
    expected_verdict: str
    score_min: int
    score_max: int


def default_golden_path() -> Path:
    """Path to the bundled golden set (repo checkout layout)."""
    return Path(__file__).resolve().parents[1] / FIXTURES_RELATIVE_PATH


def _validate_fixture(entry: Any, *, index: int, path: Path) -> GoldenFixture:
    if not isinstance(entry, dict):
        raise ValueError(f"golden set {path}: entry {index} is not an object")
    fixture_id = str(entry.get("id", f"<entry {index}>"))
    errors: list[str] = []
    for key in ("source", "url", "title", "raw_text", "expected_verdict", "score_min", "score_max"):
        if key not in entry:
            errors.append(f"missing {key!r}")
    if entry.get("expected_verdict") not in ("accept", "reject"):
        errors.append("expected_verdict must be 'accept' or 'reject'")
    try:
        score_min = int(entry["score_min"])
        score_max = int(entry["score_max"])
    except (KeyError, TypeError, ValueError):
        score_min = score_max = 0
        errors.append("score_min/score_max must be integers")
    if not (0 <= score_min <= score_max <= 10):
        errors.append("expected 0 <= score_min <= score_max <= 10")
    if errors:
        raise ValueError(f"golden set {path}: fixture {fixture_id!r}: {'; '.join(errors)}")
    return GoldenFixture(
        id=fixture_id,
        source=str(entry["source"]),
        url=str(entry["url"]),
        title=str(entry["title"]),
        raw_text=str(entry["raw_text"]),
        expected_verdict=str(entry["expected_verdict"]),
        score_min=score_min,
        score_max=score_max,
    )


def load_golden_set(path: str | Path) -> list[GoldenFixture]:
    """Load and validate a golden set from a JSON list of fixtures."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"golden set {path} must be a JSON list of fixtures")
    fixtures = [_validate_fixture(entry, index=i, path=path) for i, entry in enumerate(data)]
    ids = [fx.id for fx in fixtures]
    if len(ids) != len(set(ids)):
        duplicates = sorted({fx_id for fx_id in ids if ids.count(fx_id) > 1})
        raise ValueError(f"golden set {path}: duplicate fixture id(s): {', '.join(duplicates)}")
    return fixtures


def _candidate_from_fixture(fx: GoldenFixture) -> Candidate:
    return Candidate(
        source=fx.source, url=fx.url, title=fx.title, raw_text=fx.raw_text, status="new"
    )


# --------------------------------------------------------------------------
# Deterministic blueprint rubric
# --------------------------------------------------------------------------


def _headings(md: str) -> list[str]:
    return [line.strip() for line in md.splitlines() if line.strip().lower().startswith("##")]


def _section_span(md: str, heading: str) -> tuple[int, int] | None:
    """Return the (start, end) char offsets of a '## <heading>' body in raw md.

    Start is just past the heading line; end is the next '##' heading (or end
    of text). Offsets are computed from the raw string so CRLF or LF line
    endings both resolve correctly.
    """
    offset = 0
    start: int | None = None
    for line in md.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.lower().startswith("##"):
            if start is not None:
                return start, offset
            if heading.lower() in stripped.lower():
                start = offset + len(line)
        offset += len(line)
    return (start, offset) if start is not None else None


def _section(md: str, heading: str) -> str:
    """Return the text under '## <heading>' up to the next '##' heading."""
    span = _section_span(md, heading)
    return md[span[0] : span[1]].strip() if span is not None else ""


def check_required_sections(md: str) -> tuple[bool, str]:
    """All five mandated blueprint sections must be present as headings."""
    headings = [h.lower() for h in _headings(md)]
    missing = [s for s in REQUIRED_SECTIONS if not any(s.lower() in h for h in headings)]
    if missing:
        return False, f"missing section(s): {', '.join(missing)}"
    return True, "all required sections present"


def check_milestones(md: str) -> tuple[bool, str]:
    """The Phased Build Plan must contain numbered or bulleted milestones."""
    plan = _section(md, "Phased Build Plan")
    if not plan:
        return False, "no Phased Build Plan section"
    if _MILESTONE_RE.search(plan) or _SUBITEM_RE.search(plan):
        return True, "phased plan has milestones"
    return False, "phased plan has no milestones (no numbered/bulleted items)"


def _gpu_requirement_mentioned(md: str) -> bool:
    """True if the blueprint requires GPU hardware, ignoring negated/deferred mentions."""
    deferred_span = _section_span(md, "Deferred to Later Versions")
    for pattern in _GPU_PATTERNS:
        for match in pattern.finditer(md):
            if deferred_span and deferred_span[0] <= match.start() < deferred_span[1]:
                continue  # explicitly deferred to a later version: expected, not a violation
            # Guard windows are heuristic: a negator/deferral word must be close
            # to the token itself, or an unrelated "no GPU needed" clause would
            # hide a real requirement (and a terse section ending right before
            # the "## Deferred" heading could suppress one the other way).
            window = md[
                max(0, match.start() - _GPU_GUARD_BEFORE) : match.end() + _GPU_GUARD_AFTER
            ]
            if _NEGATOR_RE.search(window) or _DEFERRAL_RE.search(window):
                continue
            return True
    return False


def check_no_gpu_requirement(md: str, profile: HardwareProfile) -> tuple[bool, str]:
    """A CPU-only profile must not be paired with a GPU/CUDA/TPU requirement."""
    if profile.gpu or not profile.cpu_only:
        return True, "skipped (profile allows GPU)"
    if _gpu_requirement_mentioned(md):
        return False, "blueprint requires GPU/CUDA/TPU on a CPU-only profile"
    return True, "no GPU requirement"


def check_ram_within_profile(md: str, profile: HardwareProfile) -> tuple[bool, str]:
    """Flag explicit RAM requirements that exceed the profile."""
    for match in _RAM_REQ_RE.finditer(md):
        window = md[max(0, match.start() - _RAM_GUARD_BEFORE) : match.end()]
        if _NEGATOR_RE.search(window):
            continue
        if int(match.group(2)) > profile.ram_gb:
            return False, f"requires {match.group(2)}GB RAM (profile has {profile.ram_gb}GB)"
    return True, "no oversized RAM requirement"


def check_hardware_scoped_architecture(md: str) -> tuple[bool, str]:
    """The Proposed Architecture section must reference hardware constraints."""
    arch = _section(md, "Proposed Architecture")
    if not arch:
        return False, "no Proposed Architecture section"
    if _HARDWARE_TERM_RE.search(arch):
        return True, "architecture references hardware constraints"
    return False, "architecture section does not reference the hardware profile"


@dataclass
class RubricCheck:
    """A single deterministic check result for a blueprint."""

    name: str
    passed: bool
    detail: str = ""


def run_rubric(md: str, profile: HardwareProfile) -> list[RubricCheck]:
    """Run every rubric check against a blueprint and return per-check results."""
    return [
        RubricCheck("required_sections", *check_required_sections(md)),
        RubricCheck("milestones", *check_milestones(md)),
        RubricCheck("hardware_scoped_architecture", *check_hardware_scoped_architecture(md)),
        RubricCheck("no_gpu_requirement", *check_no_gpu_requirement(md, profile)),
        RubricCheck("ram_within_profile", *check_ram_within_profile(md, profile)),
    ]


# --------------------------------------------------------------------------
# Evaluation runs
# --------------------------------------------------------------------------


@dataclass
class AnalystEvalResult:
    """Outcome of evaluating one fixture against the Analyst."""

    fixture_id: str
    expected_verdict: str
    actual_verdict: str
    actual_score: int | None
    expected_score_min: int
    expected_score_max: int
    technique_summary: str = ""
    reasoning: str = ""
    error: str | None = None

    @property
    def passed(self) -> bool:
        if self.error or self.actual_verdict == "error" or self.actual_score is None:
            return False
        score_ok = self.expected_score_min <= self.actual_score <= self.expected_score_max
        return self.actual_verdict == self.expected_verdict and score_ok


@dataclass
class ArchitectEvalResult:
    """Outcome of evaluating one accepted fixture against the Architect."""

    fixture_id: str
    checks: list[RubricCheck] = field(default_factory=list)
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and bool(self.checks) and all(c.passed for c in self.checks)


@dataclass
class EvalReport:
    """Aggregate result of a full analyst + architect evaluation run."""

    analyst: list[AnalystEvalResult] = field(default_factory=list)
    architect: list[ArchitectEvalResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.analyst) and all(r.passed for r in self.architect)


def run_analyst_eval(
    fixtures: list[GoldenFixture],
    profile: HardwareProfile,
    *,
    threshold: int | None = None,
    analyze_fn: Callable[..., AnalysisResult] = analyze,
) -> list[AnalystEvalResult]:
    """Score the Analyst against every fixture; failures surface, never skip."""
    results: list[AnalystEvalResult] = []
    for fx in fixtures:
        candidate = _candidate_from_fixture(fx)
        try:
            analysis = analyze_fn(candidate, profile, threshold=threshold)
        except Exception as exc:  # noqa: BLE001 - a crash is an eval failure
            logger.warning("analyst eval fixture %r crashed: %s", fx.id, exc)
            results.append(
                AnalystEvalResult(
                    fixture_id=fx.id,
                    expected_verdict=fx.expected_verdict,
                    actual_verdict="error",
                    actual_score=None,
                    expected_score_min=fx.score_min,
                    expected_score_max=fx.score_max,
                    error=str(exc),
                )
            )
            continue
        verdict = "accept" if not analysis.rejected else "reject"
        results.append(
            AnalystEvalResult(
                fixture_id=fx.id,
                expected_verdict=fx.expected_verdict,
                actual_verdict=verdict,
                actual_score=analysis.feasibility_score,
                expected_score_min=fx.score_min,
                expected_score_max=fx.score_max,
                technique_summary=analysis.technique_summary,
                reasoning=analysis.feasibility_reasoning,
            )
        )
    return results


def run_architect_eval(
    fixtures: list[GoldenFixture],
    profile: HardwareProfile,
    *,
    threshold: int | None = None,
    analyze_fn: Callable[..., AnalysisResult] = analyze,
    blueprint_fn: Callable[..., Any] = draft_blueprint,
    analyses: dict[str, AnalysisResult] | None = None,
) -> list[ArchitectEvalResult]:
    """Rubric-score the Architect's blueprint for every accepted fixture.

    ``analyses`` maps fixture ids to the actual Analyst results from a prior
    ``run_analyst_eval`` so the LLM is only called once per fixture; when
    omitted, the Analyst is re-run for standalone use. When ``analyses`` is
    provided, ``analyze_fn``/``threshold`` are used only for the fallback path.
    """
    results: list[ArchitectEvalResult] = []
    for fx in fixtures:
        if fx.expected_verdict != "accept":
            continue
        candidate = _candidate_from_fixture(fx)
        analysis = (analyses or {}).get(fx.id)
        if analysis is None:
            try:
                analysis = analyze_fn(candidate, profile, threshold=threshold)
            except Exception as exc:  # noqa: BLE001 - a crash is an eval failure
                results.append(ArchitectEvalResult(fx.id, error=str(exc)))
                continue
        if analysis.rejected:
            results.append(
                ArchitectEvalResult(
                    fx.id,
                    error="analyst rejected the candidate; no blueprint produced",
                )
            )
            continue
        try:
            blueprint = blueprint_fn(candidate, analysis, profile)
        except Exception as exc:  # noqa: BLE001 - a crash is an eval failure
            results.append(ArchitectEvalResult(fx.id, error=str(exc)))
            continue
        md = getattr(blueprint, "blueprint_md", "") or ""
        results.append(ArchitectEvalResult(fx.id, run_rubric(md, profile)))
    return results


def run_eval(
    fixtures: list[GoldenFixture],
    profile: HardwareProfile,
    *,
    threshold: int | None = None,
    analyze_fn: Callable[..., AnalysisResult] = analyze,
    blueprint_fn: Callable[..., Any] = draft_blueprint,
) -> EvalReport:
    """Run the full analyst + architect evaluation against a golden set."""
    analyst_results = run_analyst_eval(
        fixtures, profile, threshold=threshold, analyze_fn=analyze_fn
    )
    analyses = {
        r.fixture_id: AnalysisResult(
            technique_summary=r.technique_summary,
            feasibility_score=r.actual_score if r.actual_score is not None else 0,
            feasibility_reasoning=r.reasoning,
            rejected=r.actual_verdict != "accept",
        )
        for r in analyst_results
    }
    architect_results = run_architect_eval(
        fixtures,
        profile,
        threshold=threshold,
        analyze_fn=analyze_fn,
        blueprint_fn=blueprint_fn,
        analyses=analyses,
    )
    return EvalReport(analyst=analyst_results, architect=architect_results)


def format_report(report: EvalReport) -> str:
    """Render a human-readable eval report."""
    analyst_total = len(report.analyst)
    analyst_passed = sum(1 for r in report.analyst if r.passed)
    architect_total = len(report.architect)
    architect_passed = sum(1 for r in report.architect if r.passed)
    lines = [
        "Golden-set evaluation",
        f"  analyst: {analyst_passed}/{analyst_total} passed",
        f"  architect: {architect_passed}/{architect_total} passed",
        f"  OVERALL: {'PASS' if report.passed else 'FAIL'}",
        "",
        "Analyst details:",
    ]
    for r in report.analyst:
        marker = "PASS" if r.passed else "FAIL"
        if r.error:
            lines.append(f"  [{marker}] {r.fixture_id}: error: {r.error}")
            continue
        lines.append(
            f"  [{marker}] {r.fixture_id}: expected={r.expected_verdict} "
            f"actual={r.actual_verdict} score={r.actual_score} "
            f"(band {r.expected_score_min}-{r.expected_score_max})"
        )
    lines.append("")
    lines.append("Architect details:")
    for r in report.architect:
        if r.error:
            lines.append(f"  [FAIL] {r.fixture_id}: error: {r.error}")
            continue
        if r.passed:
            lines.append(f"  [PASS] {r.fixture_id}: all {len(r.checks)} rubric checks passed")
            continue
        failed = [f"{c.name}: {c.detail}" for c in r.checks if not c.passed]
        lines.append(f"  [FAIL] {r.fixture_id}: {'; '.join(failed)}")
    return "\n".join(lines)
