"""Analyst agent tests: mocked call_llm, accept/reject/malformed paths, prompts."""

from __future__ import annotations

import importlib
import json

from praxis.agents.analyst import AnalysisResult, analyze
from praxis.db import Candidate

analyst_module = importlib.import_module("praxis.agents.analyst")


def make_candidate(
    session, url="https://example.com/paper", title="A Paper", raw_text="some abstract body"
):
    cand = Candidate(source="arxiv", url=url, title=title, raw_text=raw_text, status="new")
    session.add(cand)
    session.commit()
    session.refresh(cand)
    return cand


def mock_call_llm(monkeypatch, payload):
    """Patch analyst.call_llm to return a JSON payload."""
    calls = {}

    def fake(prompt, system=None, model=None, **kwargs):
        calls["prompt"] = prompt
        calls["system"] = system
        calls["stage"] = kwargs.get("stage")
        calls["candidate_id"] = kwargs.get("candidate_id")
        return json.dumps(payload)

    monkeypatch.setattr(analyst_module, "call_llm", fake)
    return calls


def test_analyze_accept_path(db_session, hardware_profile, monkeypatch):
    cand = make_candidate(db_session)
    calls = mock_call_llm(
        monkeypatch,
        {
            "technique_summary": "Finetune a small transformer on CPU.",
            "feasibility_score": 8,
            "feasibility_reasoning": "Fits in 8GB RAM.",
            "rejected": False,
        },
    )

    result = analyze(cand, hardware_profile)

    assert isinstance(result, AnalysisResult)
    assert result.feasibility_score == 8
    assert result.rejected is False
    assert result.technique_summary == "Finetune a small transformer on CPU."
    assert calls["prompt"]
    assert calls["system"]
    assert calls["stage"] == "analyst"
    assert calls["candidate_id"] == cand.id

    db_session.expire_all()
    stored = db_session.get(Candidate, cand.id)
    assert stored.status == "analyzed"
    assert stored.feasibility_score == 8
    assert stored.technique_summary.startswith("Finetune")


def test_analyze_reject_path(db_session, hardware_profile, monkeypatch):
    cand = make_candidate(db_session)
    mock_call_llm(
        monkeypatch,
        {
            "technique_summary": "Train a 70B model from scratch.",
            "feasibility_score": 2,
            "feasibility_reasoning": "Needs multiple GPUs and huge budget.",
            "rejected": True,
        },
    )

    result = analyze(cand, hardware_profile)

    assert result.rejected is True
    db_session.expire_all()
    assert db_session.get(Candidate, cand.id).status == "rejected"


def test_analyze_malformed_json(db_session, hardware_profile, monkeypatch, caplog):
    cand = make_candidate(db_session)
    monkeypatch.setattr(
        analyst_module, "call_llm", lambda prompt, system=None, **kwargs: "not json at all"
    )

    with caplog.at_level("WARNING"):
        result = analyze(cand, hardware_profile)

    assert result.rejected is True
    assert result.feasibility_score == 0
    assert "malformed LLM response" in caplog.text
    db_session.expire_all()
    assert db_session.get(Candidate, cand.id).status == "rejected"


def test_analyze_strips_code_fences(db_session, hardware_profile, monkeypatch):
    cand = make_candidate(db_session)
    payload = {
        "technique_summary": "Parse and filter data with plain Python.",
        "feasibility_score": 7,
        "feasibility_reasoning": "Trivially fits constraints.",
        "rejected": False,
    }
    monkeypatch.setattr(
        analyst_module,
        "call_llm",
        lambda prompt, system=None, **kwargs: f"```json\n{json.dumps(payload)}\n```",
    )

    result = analyze(cand, hardware_profile)

    assert result.feasibility_score == 7
    assert result.rejected is False


def test_analyze_prompt_includes_hardware(db_session, hardware_profile, monkeypatch):
    cand = make_candidate(db_session, raw_text="attention is all you need")
    calls = mock_call_llm(
        monkeypatch,
        {
            "technique_summary": "x",
            "feasibility_score": 5,
            "feasibility_reasoning": "y",
            "rejected": False,
        },
    )

    analyze(cand, hardware_profile)

    prompt = calls["prompt"]
    assert "CPU-only: True" in prompt
    assert "RAM (GB): 8" in prompt
    assert "Monthly budget (USD): 15.0" in prompt
    assert "attention is all you need" in prompt


def test_analyze_threshold_via_env(db_session, hardware_profile, monkeypatch):
    monkeypatch.setenv("PRAXIS_FEASIBILITY_THRESHOLD", "7")
    cand = make_candidate(db_session)
    mock_call_llm(
        monkeypatch,
        {
            "technique_summary": "x",
            "feasibility_score": 6,
            "feasibility_reasoning": "y",
            "rejected": False,
        },
    )

    result = analyze(cand, hardware_profile)

    assert result.rejected is True


def test_analyze_threshold_explicit_param(db_session, hardware_profile, monkeypatch):
    cand = make_candidate(db_session)
    mock_call_llm(
        monkeypatch,
        {
            "technique_summary": "x",
            "feasibility_score": 7,
            "feasibility_reasoning": "y",
            "rejected": False,
        },
    )

    result = analyze(cand, hardware_profile, threshold=8)

    assert result.rejected is True


def test_analyze_rejected_flag_overrides_high_score(db_session, hardware_profile, monkeypatch):
    cand = make_candidate(db_session)
    mock_call_llm(
        monkeypatch,
        {
            "technique_summary": "x",
            "feasibility_score": 9,
            "feasibility_reasoning": "still a no",
            "rejected": True,
        },
    )

    result = analyze(cand, hardware_profile)

    assert result.rejected is True
    db_session.expire_all()
    assert db_session.get(Candidate, cand.id).status == "rejected"


def test_analyze_valid_response_single_call(db_session, hardware_profile, monkeypatch):
    """A valid first response must not trigger the repair retry."""
    cand = make_candidate(db_session)
    calls = {"n": 0}

    def fake(prompt, system=None, model=None, **kwargs):
        calls["n"] += 1
        return json.dumps(
            {
                "technique_summary": "x",
                "feasibility_score": 8,
                "feasibility_reasoning": "y",
                "rejected": False,
            }
        )

    monkeypatch.setattr(analyst_module, "call_llm", fake)

    result = analyze(cand, hardware_profile)

    assert calls["n"] == 1
    assert result.rejected is False


def test_analyze_repairs_malformed_response(db_session, hardware_profile, monkeypatch):
    """A malformed first response is retried once with a strict JSON repair prompt."""
    cand = make_candidate(db_session)
    calls = {"n": 0, "second_prompt": None}
    payload = {
        "technique_summary": "fixed summary",
        "feasibility_score": 6,
        "feasibility_reasoning": "fits",
        "rejected": False,
    }

    def fake(prompt, system=None, model=None, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return "totally not json"
        calls["second_prompt"] = prompt
        return json.dumps(payload)

    monkeypatch.setattr(analyst_module, "call_llm", fake)

    result = analyze(cand, hardware_profile)

    assert calls["n"] == 2
    assert result.rejected is False
    assert result.feasibility_score == 6
    assert result.technique_summary == "fixed summary"
    assert "not valid JSON" in calls["second_prompt"]
    db_session.expire_all()
    assert db_session.get(Candidate, cand.id).status == "analyzed"


def test_analyze_repair_failure_still_rejects(db_session, hardware_profile, monkeypatch, caplog):
    """If the repair also fails to parse, the candidate is still rejected loudly."""
    cand = make_candidate(db_session)
    calls = {"n": 0}

    def fake(prompt, system=None, model=None, **kwargs):
        calls["n"] += 1
        return "still not json"

    monkeypatch.setattr(analyst_module, "call_llm", fake)

    with caplog.at_level("WARNING"):
        result = analyze(cand, hardware_profile)

    assert calls["n"] == 2
    assert result.rejected is True
    assert result.feasibility_score == 0
    assert "malformed LLM response" in caplog.text
    db_session.expire_all()
    assert db_session.get(Candidate, cand.id).status == "rejected"


def test_analyze_truncates_long_raw_text(db_session, hardware_profile, monkeypatch):
    """Raw text longer than the 6000-char cap is truncated in the prompt."""
    cand = make_candidate(db_session, raw_text="word " * 4000)
    calls = mock_call_llm(
        monkeypatch,
        {
            "technique_summary": "x",
            "feasibility_score": 6,
            "feasibility_reasoning": "y",
            "rejected": False,
        },
    )

    analyze(cand, hardware_profile)

    assert "[truncated]" in calls["prompt"]

