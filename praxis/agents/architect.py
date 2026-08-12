"""Architect: produce a hardware-calibrated engineering blueprint."""

from __future__ import annotations

import logging

from praxis.agents.analyst import AnalysisResult
from praxis.config import HardwareProfile
from praxis.db import Blueprint, Candidate, get_session
from praxis.llm import call_llm

logger = logging.getLogger(__name__)

MAX_RAW_TEXT_CHARS = 6000


def _system_prompt() -> str:
    return (
        "You are the Architect agent in Praxis, a system that turns research into "
        "implementable engineering projects. A Scout found a candidate item and an "
        "Analyst extracted its core technique and scored its feasibility. Your job "
        "is to turn that into a detailed, hardware-calibrated engineering "
        "blueprint.\n\n"
        "Respond with MARKDOWN ONLY, no JSON, no prose before or after. Use exactly "
        "this section structure:\n"
        "# <Short Title> — Blueprint\n"
        "## Problem Statement\n"
        "## Proposed Architecture\n"
        "## Phased Build Plan\n"
        "## Deferred to Later Versions\n"
        "## Difficulty & Time Estimate\n\n"
        "The Proposed Architecture must be explicitly scoped to the target "
        "hardware profile (CPU-only, RAM, GPU, monthly budget): if the technique "
        "needs more than the hardware allows, say so and show what IS buildable. "
        "The Phased Build Plan must be a sequence of self-contained milestones, "
        "not one giant build. In Deferred to Later Versions, explicitly call out "
        "what you are leaving out of this first pass and why. End with a rough "
        "difficulty and time estimate."
    )


def _build_prompt(
    candidate: Candidate,
    analysis: AnalysisResult,
    profile: HardwareProfile,
) -> str:
    raw_text = (candidate.raw_text or "").strip()
    if len(raw_text) > MAX_RAW_TEXT_CHARS:
        raw_text = raw_text[:MAX_RAW_TEXT_CHARS] + "\n[truncated]"
    return (
        f"Candidate:\n"
        f"- title: {candidate.title}\n"
        f"- source: {candidate.source}\n"
        f"- url: {candidate.url}\n"
        f"- raw text:\n{raw_text}\n\n"
        f"Analyst's assessment:\n"
        f"- technique_summary: {analysis.technique_summary}\n"
        f"- feasibility_score: {analysis.feasibility_score}\n"
        f"- feasibility_reasoning: {analysis.feasibility_reasoning}\n\n"
        f"Target hardware profile:\n"
        f"- CPU-only: {profile.cpu_only}\n"
        f"- RAM (GB): {profile.ram_gb}\n"
        f"- GPU: {profile.gpu}\n"
        f"- Monthly budget (USD): {profile.monthly_budget_usd}\n\n"
        f"Write the blueprint as Markdown."
    )


def draft_blueprint(
    candidate: Candidate,
    analysis: AnalysisResult,
    profile: HardwareProfile,
) -> Blueprint:
    """Draft a hardware-calibrated blueprint, persist it, and return the row."""
    md = call_llm(
        _build_prompt(candidate, analysis, profile),
        system=_system_prompt(),
        stage="architect",
        candidate_id=candidate.id,
    )
    if not (md or "").strip():
        logger.warning(
            "architect: empty LLM response for candidate %r; storing as-is",
            candidate.url,
        )

    blueprint: Blueprint | None = None
    session = get_session()
    try:
        stored = session.get(Candidate, candidate.id) if candidate.id is not None else None
        if stored is None:
            stored = candidate
            session.add(stored)
        stored.status = "blueprinted"
        session.flush()
        blueprint = Blueprint(
            candidate_id=stored.id,
            feasibility_score=float(analysis.feasibility_score),
            blueprint_md=md,
        )
        session.add(blueprint)
        session.commit()
    finally:
        session.close()

    return blueprint


architect = draft_blueprint
