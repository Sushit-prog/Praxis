"""Analyst: extract a candidate's core technique and score feasibility."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

from praxis.config import HardwareProfile
from praxis.db import Candidate, get_session
from praxis.llm import call_llm

logger = logging.getLogger(__name__)

DEFAULT_FEASIBILITY_THRESHOLD = 4
THRESHOLD_ENV = "PRAXIS_FEASIBILITY_THRESHOLD"
MAX_RAW_TEXT_CHARS = 6000


@dataclass
class AnalysisResult:
    """The Analyst's verdict on a candidate."""

    technique_summary: str
    feasibility_score: int
    feasibility_reasoning: str
    rejected: bool


def _resolve_threshold(threshold: int | None) -> int:
    if threshold is not None:
        return threshold
    raw = os.environ.get(THRESHOLD_ENV)
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            logger.warning("invalid %s=%r; using default", THRESHOLD_ENV, raw)
    return DEFAULT_FEASIBILITY_THRESHOLD


def _system_prompt() -> str:
    return (
        "You are the Analyst agent in Praxis, a system that turns research into "
        "implementable engineering projects. Your job is to read a candidate item "
        "(paper, repo, or post) and extract the single core technique that could "
        "be implemented, then score how feasible it is to build on the given "
        "hardware within the given monthly budget.\n\n"
        "Respond with JSON ONLY, no prose, no markdown. Use exactly this schema:\n"
        '{"technique_summary": "one or two sentences describing the core '
        'implementable technique", "feasibility_score": 0, '
        '"feasibility_reasoning": "brief reasoning covering hardware and budget '
        'constraints", "rejected": false}\n\n'
        "feasibility_score is an integer 0-10. Higher is more buildable. Score "
        "down when the technique requires GPUs, more RAM than available, "
        "expensive API usage, or infrastructure that exceeds the budget."
    )


def _build_prompt(candidate: Candidate, profile: HardwareProfile) -> str:
    raw_text = (candidate.raw_text or "").strip()
    if len(raw_text) > MAX_RAW_TEXT_CHARS:
        raw_text = raw_text[:MAX_RAW_TEXT_CHARS] + "\n[truncated]"
    return (
        f"Candidate:\n"
        f"- title: {candidate.title}\n"
        f"- source: {candidate.source}\n"
        f"- url: {candidate.url}\n"
        f"- raw text:\n{raw_text}\n\n"
        f"Target hardware profile:\n"
        f"- CPU-only: {profile.cpu_only}\n"
        f"- RAM (GB): {profile.ram_gb}\n"
        f"- GPU: {profile.gpu}\n"
        f"- Monthly budget (USD): {profile.monthly_budget_usd}\n\n"
        f"Decide whether this technique is actually buildable within these "
        f"constraints - not just whether it is academically interesting. "
        f"Return the JSON verdict."
    )


def _repair_prompt(bad_response: str) -> str:
    """Ask the model to re-emit a response that failed strict JSON parsing."""
    return (
        "Your previous response was not valid JSON, so it could not be parsed. "
        "Here is what you returned:\n"
        f"---\n{bad_response[:2000]}\n---\n\n"
        "Respond again with JSON ONLY, exactly matching the schema in your "
        "instructions. No prose, no markdown fences."
    )


def _extract_json(text: str) -> str:
    """Strip code fences / surrounding prose and return the JSON substring."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
    if "{" in stripped and "}" in stripped:
        stripped = stripped[stripped.find("{") : stripped.rfind("}") + 1]
    return stripped.strip()


def _parse_response(text: str, threshold: int) -> AnalysisResult | None:
    """Parse the LLM reply into an AnalysisResult, or None if malformed."""
    try:
        data = json.loads(_extract_json(text))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    try:
        score = int(data.get("feasibility_score", 0))
    except (TypeError, ValueError):
        return None
    score = max(0, min(10, score))

    rejected = bool(data.get("rejected", score < threshold))
    return AnalysisResult(
        technique_summary=str(data.get("technique_summary", "")).strip(),
        feasibility_score=score,
        feasibility_reasoning=str(data.get("feasibility_reasoning", "")).strip(),
        rejected=rejected or score < threshold,
    )


def _rejected_analysis(reason: str) -> AnalysisResult:
    return AnalysisResult(
        technique_summary="",
        feasibility_score=0,
        feasibility_reasoning=reason,
        rejected=True,
    )


def analyze(
    candidate: Candidate,
    profile: HardwareProfile,
    threshold: int | None = None,
) -> AnalysisResult:
    """Extract and score a candidate's technique, persisting status + analysis."""
    threshold = _resolve_threshold(threshold)
    prompt = _build_prompt(candidate, profile)
    response = call_llm(
        prompt, system=_system_prompt(), stage="analyst", candidate_id=candidate.id
    )
    first_response = response

    result = _parse_response(response, threshold)
    if result is None:
        logger.warning(
            "analyst: malformed LLM response for candidate %r; requesting strict JSON repair",
            candidate.url,
        )
        response = call_llm(
            _repair_prompt(first_response),
            system=_system_prompt(),
            stage="analyst",
            candidate_id=candidate.id,
        )
        result = _parse_response(response, threshold)

    if result is None:
        logger.warning(
            "analyst: malformed LLM response for candidate %r: %r",
            candidate.url,
            first_response[:200],
        )
        result = _rejected_analysis("malformed LLM response; could not parse verdict")

    session = get_session()
    try:
        stored = session.get(Candidate, candidate.id) if candidate.id is not None else None
        if stored is None:
            stored = candidate
            session.add(stored)
        stored.status = "rejected" if result.rejected else "analyzed"
        stored.technique_summary = result.technique_summary
        stored.feasibility_score = result.feasibility_score
        stored.feasibility_reasoning = result.feasibility_reasoning
        session.commit()
    finally:
        session.close()

    return result
