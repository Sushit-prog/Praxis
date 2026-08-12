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


def _counting_completion(calls):
    """A completion that counts invocations and returns a canned usage response."""
    def fake(**kwargs):
        calls["n"] += 1
        return {
            "choices": [{"message": {"content": "cached answer"}}],
            "model": "groq/llama-3.1-8b-instant",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "_hidden_params": {"response_cost": 0.0001},
        }

    return fake


def test_llm_cache_hit_skips_completion(db_session):
    from praxis.llm import call_llm

    calls = {"n": 0}
    completion = _counting_completion(calls)

    first = call_llm("same prompt", system="same system", completion=completion)
    second = call_llm("same prompt", system="same system", completion=completion)

    assert first == second == "cached answer"
    assert calls["n"] == 1  # second call served from the cache
    rows = db_session.scalars(select(LLMUsage)).all()
    assert len(rows) == 2
    miss, hit = rows
    assert miss.cached is False
    assert miss.cost_usd == pytest.approx(0.0001)
    assert hit.cached is True
    assert hit.cost_usd is None
    assert hit.total_tokens is None


def test_llm_cache_distinct_inputs_miss(db_session):
    from praxis.llm import call_llm

    calls = {"n": 0}
    completion = _counting_completion(calls)

    call_llm("prompt one", completion=completion)
    call_llm("prompt two", completion=completion)

    assert calls["n"] == 2


def test_llm_cache_disabled_by_env(db_session, monkeypatch):
    from praxis.llm import call_llm

    monkeypatch.setenv("PRAXIS_LLM_CACHE", "0")
    calls = {"n": 0}
    completion = _counting_completion(calls)

    call_llm("same prompt", completion=completion)
    call_llm("same prompt", completion=completion)

    assert calls["n"] == 2


def test_llm_cache_seeded_row_served_without_completion(db_session):
    from praxis.db import LLMCache
    from praxis.llm import _cache_key, _resolve_model, call_llm

    model = _resolve_model(None)
    key = _cache_key(model, "sys", "prompt")
    db_session.add(LLMCache(key=key, model=model, response="stored"))
    db_session.commit()

    calls = {"n": 0}
    completion = _counting_completion(calls)

    result = call_llm("prompt", system="sys", completion=completion)

    assert result == "stored"
    assert calls["n"] == 0


def test_llm_cache_miss_persists_row(db_session):
    from praxis.db import LLMCache
    from praxis.llm import call_llm

    call_llm("prompt", system="sys", completion=_counting_completion({"n": 0}))

    rows = db_session.scalars(select(LLMCache)).all()
    assert len(rows) == 1
    assert rows[0].response == "cached answer"
    assert rows[0].model == "groq/llama-3.1-8b-instant"


def test_llm_cache_key_differs_on_system_change(db_session):
    from praxis.llm import _cache_key, _resolve_model

    model = _resolve_model(None)
    assert _cache_key(model, "system A", "prompt") != _cache_key(model, "system B", "prompt")
    assert _cache_key(model, "sys", "prompt A") != _cache_key(model, "sys", "prompt B")


def test_llm_falls_back_to_next_model_on_failure(db_session, monkeypatch):
    from praxis.llm import _resolve_model, call_llm

    primary = _resolve_model(None)
    monkeypatch.setenv("PRAXIS_FALLBACK_MODELS", "fallback-model")
    models_used = []

    def fake_completion(**kwargs):
        models_used.append(kwargs["model"])
        if kwargs["model"] == primary:
            raise RuntimeError("rate limited")
        return {
            "choices": [{"message": {"content": "recovered"}}],
            "model": "fallback-model",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "_hidden_params": {"response_cost": 0.0001},
        }

    result = call_llm("prompt", completion=fake_completion)

    assert result == "recovered"
    assert models_used == [primary, "fallback-model"]
    rows = db_session.scalars(select(LLMUsage)).all()
    assert len(rows) == 2  # failed attempt + successful fallback call
    errors = [r for r in rows if r.error]
    successes = [r for r in rows if r.error is None]
    assert errors[0].model == primary
    assert errors[0].error == "rate limited"
    assert successes[0].model == "fallback-model"
    assert successes[0].cost_usd == pytest.approx(0.0001)


def test_llm_fallback_result_served_from_cache(db_session, monkeypatch):
    """A cached fallback response is served when the primary is still down."""
    from praxis.llm import _resolve_model, call_llm

    primary = _resolve_model(None)
    monkeypatch.setenv("PRAXIS_FALLBACK_MODELS", "fb-model")
    calls = {"n": 0}

    def fake_completion(**kwargs):
        calls["n"] += 1
        if kwargs["model"] == primary:
            raise RuntimeError("rate limited")
        return {
            "choices": [{"message": {"content": "recovered"}}],
            "model": "fb-model",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "_hidden_params": {"response_cost": 0.0001},
        }

    assert call_llm("prompt", completion=fake_completion) == "recovered"
    # Second call: primary fails again, but the fallback response is cached,
    # so no second fallback completion call happens.
    assert call_llm("prompt", completion=fake_completion) == "recovered"
    assert calls["n"] == 3  # primary fail + fallback (call 1); primary fail only (call 2)
    hits = db_session.scalars(select(LLMUsage).where(LLMUsage.cached.is_(True))).all()
    assert len(hits) == 1


def test_llm_all_models_fail_raises(db_session, monkeypatch):
    from praxis.llm import call_llm

    monkeypatch.setenv("PRAXIS_FALLBACK_MODELS", "fb1, fb2")
    calls = {"n": 0}

    def fake_completion(**kwargs):
        calls["n"] += 1
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        call_llm("prompt", completion=fake_completion)

    assert calls["n"] == 3  # primary + two fallbacks
    rows = db_session.scalars(select(LLMUsage)).all()
    assert len(rows) == 3  # one failure row per attempted model
    assert all(r.error == "boom" for r in rows)


def test_invalidate_llm_cache_deletes_row(db_session):
    from praxis.db import LLMCache
    from praxis.llm import call_llm, invalidate_llm_cache

    call_llm("prompt", system="sys", completion=_counting_completion({"n": 0}))
    assert len(db_session.scalars(select(LLMCache)).all()) == 1

    invalidate_llm_cache("prompt", system="sys")

    assert db_session.scalars(select(LLMCache)).all() == []


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
            LLMUsage(model="m1", stage="analyst", cached=True),
        ]
    )
    db_session.commit()

    summary = usage_summary(session=db_session)

    assert summary.totals.calls == 3  # cached hit excluded from real-call counts
    assert summary.totals.cached_hits == 1
    assert summary.totals.total_tokens == 800
    assert summary.totals.prompt_tokens == 600
    assert summary.totals.completion_tokens == 200
    assert summary.totals.cost_usd == pytest.approx(0.006)
    assert summary.recent.calls == 3
    assert summary.recent.cached_hits == 1
    assert summary.by_stage["analyst"].calls == 2
    assert summary.by_stage["analyst"].total_tokens == 400
    assert summary.by_stage["architect"].cost_usd == pytest.approx(0.003)
    assert summary.by_model["m1"].calls == 2
    assert summary.by_model["m2"].total_tokens == 400
