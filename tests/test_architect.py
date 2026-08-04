"""Architect agent tests: mocked call_llm, persistence, prompts, alias."""

from __future__ import annotations

import importlib

from praxis.agents.analyst import AnalysisResult
from praxis.agents.architect import draft_blueprint
from praxis.db import Blueprint, Candidate

architect_module = importlib.import_module("praxis.agents.architect")

BLUEPRINT_MD = (
    "# Fine-Tune on CPU \u2014 Blueprint\n\n"
    "## Problem Statement\nBuild a small fine-tuner.\n\n"
    "## Proposed Architecture\nCPU-only, 8GB RAM.\n\n"
    "## Phased Build Plan\n1. Data loader 2. Training loop\n\n"
    "## Deferred to Later Versions\nNo GPU acceleration.\n\n"
    "## Difficulty & Time Estimate\nEasy, 1 week."
)


def make_candidate(
    session, url="https://example.com/paper", title="A Paper", raw_text="some abstract body"
):
    cand = Candidate(source="arxiv", url=url, title=title, raw_text=raw_text, status="new")
    session.add(cand)
    session.commit()
    session.refresh(cand)
    return cand


def make_analysis():
    return AnalysisResult(
        technique_summary="Fine-tune a small transformer on CPU.",
        feasibility_score=8,
        feasibility_reasoning="Fits in 8GB RAM.",
        rejected=False,
    )


def mock_call_llm(monkeypatch, text):
    """Patch architect.call_llm to return a canned markdown response."""
    calls = {}

    def fake(prompt, system=None, model=None):
        calls["prompt"] = prompt
        calls["system"] = system
        return text

    monkeypatch.setattr(architect_module, "call_llm", fake)
    return calls


def test_architect_persists_blueprint(db_session, hardware_profile, monkeypatch):
    cand = make_candidate(db_session)
    analysis = make_analysis()
    calls = mock_call_llm(monkeypatch, BLUEPRINT_MD)

    bp = draft_blueprint(cand, analysis, hardware_profile)

    assert isinstance(bp, Blueprint)
    assert bp.blueprint_md == BLUEPRINT_MD
    assert bp.candidate_id == cand.id
    assert calls["prompt"]
    assert calls["system"]

    db_session.expire_all()
    stored = db_session.get(Candidate, cand.id)
    assert stored.status == "blueprinted"
    assert stored.blueprints[0].blueprint_md == BLUEPRINT_MD
    assert stored.blueprints[0].feasibility_score == 8.0
    assert stored.blueprints[0].candidate_id == cand.id


def test_architect_prompt_includes_hardware_and_analysis(db_session, hardware_profile, monkeypatch):
    cand = make_candidate(db_session, raw_text="attention is all you need")
    analysis = make_analysis()
    calls = mock_call_llm(monkeypatch, BLUEPRINT_MD)

    draft_blueprint(cand, analysis, hardware_profile)

    prompt = calls["prompt"]
    assert "CPU-only: True" in prompt
    assert "RAM (GB): 8" in prompt
    assert "Monthly budget (USD): 15.0" in prompt
    assert "Fine-tune a small transformer on CPU." in prompt
    assert "feasibility_score: 8" in prompt
    assert "attention is all you need" in prompt


def test_architect_empty_output_persists(db_session, hardware_profile, monkeypatch, caplog):
    cand = make_candidate(db_session)
    analysis = make_analysis()
    monkeypatch.setattr(architect_module, "call_llm", lambda prompt, system=None: "   ")

    with caplog.at_level("WARNING"):
        bp = draft_blueprint(cand, analysis, hardware_profile)

    assert bp.blueprint_md == "   "
    assert "empty LLM response" in caplog.text
    db_session.expire_all()
    assert db_session.get(Candidate, cand.id).status == "blueprinted"


def test_architect_alias():
    from praxis.agents import architect

    assert architect is draft_blueprint
