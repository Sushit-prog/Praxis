"""Tests for LLM usage/cost recording in the litellm wrapper (Task #2)."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from praxis import llm as llm_module
from praxis.db import LLMUsage


def _usage_fake_completion(**kwargs):
    """A litellm-shaped response carrying usage, cost, and content."""
    return {
        "choices": [{"message": {"content": "ok"}}],
        "model": "groq/llama-3.1-8b-instant",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "_hidden_params": {"response_cost": 0.000123, "response_ms": 88},
    }


def _plain_fake_completion(**kwargs):
    """A response with no usage info (what existing tests inject)."""
    return {"choices": [{"message": {"content": "fake model output"}}]}


def test_call_llm_records_usage(db_session):
    from praxis.llm import call_llm

    result = call_llm(
        "hello", completion=_usage_fake_completion, stage="analyst", candidate_id=7
    )

    assert result == "ok"
    rows = db_session.scalars(select(LLMUsage)).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.model == "groq/llama-3.1-8b-instant"
    assert row.stage == "analyst"
    assert row.candidate_id == 7
    assert row.prompt_tokens == 10
    assert row.completion_tokens == 5
    assert row.total_tokens == 15
    assert row.cost_usd == pytest.approx(0.000123)
    assert isinstance(row.latency_ms, int) and row.latency_ms >= 0


def test_call_llm_without_usage_skips_recording(db_session):
    from praxis.llm import call_llm

    assert call_llm("hello", completion=_plain_fake_completion) == "fake model output"
    assert db_session.scalars(select(LLMUsage)).all() == []


def test_call_llm_records_failed_attempt(db_session):
    """A raising completion is recorded as an error row and re-raised."""
    from praxis.llm import call_llm

    def failing_completion(**kwargs):
        raise RuntimeError("rate limited")

    with pytest.raises(RuntimeError):
        call_llm("hello", completion=failing_completion, stage="analyst", candidate_id=3)

    rows = db_session.scalars(select(LLMUsage)).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.error == "rate limited"
    assert row.stage == "analyst"
    assert row.candidate_id == 3
    assert row.total_tokens is None
    assert row.cost_usd is None
    assert isinstance(row.latency_ms, int) and row.latency_ms >= 0


def test_recording_failure_does_not_break_call(db_session, monkeypatch):
    from praxis.llm import call_llm

    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(llm_module, "get_session", boom)

    assert call_llm("hello", completion=_usage_fake_completion) == "ok"


def test_response_usage_object_style():
    """Real litellm responses expose usage as an object, not a dict."""
    from praxis.llm import _response_usage

    class Usage:
        prompt_tokens = 2
        completion_tokens = 3
        total_tokens = 5

    class Response:
        usage = Usage()

    assert _response_usage(Response()) == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
    }
    assert _response_usage(object()) is None


def test_cost_prefers_hidden_params():
    from praxis.llm import _response_cost

    response = {"model": "m", "usage": {}, "_hidden_params": {"response_cost": "0.05"}}
    assert _response_cost(response) == pytest.approx(0.05)


def test_cost_falls_back_to_completion_cost(monkeypatch):
    from praxis.llm import _response_cost

    response = {"model": "m", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    monkeypatch.setattr(llm_module, "_completion_cost", lambda **kwargs: 0.05)
    assert _response_cost(response) == pytest.approx(0.05)


def test_cost_none_when_pricing_unknown(monkeypatch):
    from praxis.llm import _response_cost

    response = {"model": "m", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    def boom(**kwargs):
        raise RuntimeError("unknown model")

    monkeypatch.setattr(llm_module, "_completion_cost", boom)
    assert _response_cost(response) is None


def test_usage_summary_aggregates(db_session):
    from praxis.db import LLMUsage, usage_summary

    db_session.add_all(
        [
            LLMUsage(
                model="m1",
                stage="analyst",
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                cost_usd=0.001,
                latency_ms=10,
            ),
            LLMUsage(
                model="m1",
                stage="analyst",
                prompt_tokens=200,
                completion_tokens=50,
                total_tokens=250,
                cost_usd=0.002,
                latency_ms=20,
            ),
            LLMUsage(
                model="m2",
                stage="architect",
                prompt_tokens=300,
                completion_tokens=100,
                total_tokens=400,
                cost_usd=0.003,
                latency_ms=30,
            ),
        ]
    )
    db_session.commit()

    summary = usage_summary(session=db_session)

    assert summary.totals.calls == 3
    assert summary.totals.total_tokens == 800
    assert summary.totals.prompt_tokens == 600
    assert summary.totals.completion_tokens == 200
    assert summary.totals.cost_usd == pytest.approx(0.006)
    assert summary.recent.calls == 3
    assert summary.by_stage["analyst"].calls == 2
    assert summary.by_stage["analyst"].total_tokens == 400
    assert summary.by_stage["architect"].cost_usd == pytest.approx(0.003)
    assert summary.by_model["m1"].calls == 2
    assert summary.by_model["m2"].total_tokens == 400
