"""Tests for health-aware provider routing: exhaustion classification, cooldowns, instant switch."""

from __future__ import annotations

from praxis import providers as providers_module
from praxis.db import get_provider_health, provider_health_rows
from praxis.providers import (
    JOB_CODER,
    JOB_PIPELINE,
    ExhaustionSignal,
    ProviderPool,
    classify_auth_failure,
    classify_exhaustion,
    order_models,
    provider_of,
    scan_exhaustion,
)


def test_provider_of_extracts_prefix():
    assert provider_of("groq/llama-3.1-8b-instant") == "groq"
    assert provider_of("openrouter/openai/gpt-4o-mini") == "openrouter"
    assert provider_of("cerebras/llama-3.3-70b") == "cerebras"
    assert provider_of("bare-model") == "bare-model"
    assert provider_of("") == ""


class _FakeError(Exception):
    def __init__(self, message="boom", status_code=None):
        super().__init__(message)
        self.status_code = status_code


def test_classify_exhaustion_rate_limit():
    assert classify_exhaustion(_FakeError("Rate limit exceeded")) == ExhaustionSignal.RATE_LIMIT
    assert classify_exhaustion(_FakeError("too many requests")) == ExhaustionSignal.RATE_LIMIT
    assert classify_exhaustion(_FakeError("429 gotcha")) == ExhaustionSignal.RATE_LIMIT
    assert classify_exhaustion(_FakeError("http 429")) == ExhaustionSignal.RATE_LIMIT
    assert classify_exhaustion(_FakeError("nope", status_code=429)) == ExhaustionSignal.RATE_LIMIT


def test_classify_exhaustion_quota_and_context():
    assert classify_exhaustion(_FakeError("insufficient_quota")) == ExhaustionSignal.QUOTA
    assert classify_exhaustion(_FakeError("exceeded your current quota")) == ExhaustionSignal.QUOTA
    assert classify_exhaustion(_FakeError("quota", status_code=402)) == ExhaustionSignal.QUOTA
    assert classify_exhaustion(_FakeError("context window exceeded")) == ExhaustionSignal.CONTEXT
    assert classify_exhaustion(_FakeError("maximum context length")) == ExhaustionSignal.CONTEXT


def test_classify_exhaustion_transient_is_none():
    assert classify_exhaustion(_FakeError("connection reset")) is None
    assert classify_exhaustion(_FakeError("500 internal error", status_code=500)) is None
    assert classify_exhaustion(_FakeError("api key invalid", status_code=401)) is None


def test_classify_exhaustion_by_class_name():
    class FakeRateLimitError(Exception):
        pass

    assert classify_exhaustion(FakeRateLimitError("nope")) == ExhaustionSignal.RATE_LIMIT


def test_scan_exhaustion_text():
    assert scan_exhaustion("HTTP 429 rate limit exceeded") == ExhaustionSignal.RATE_LIMIT
    assert scan_exhaustion("quota exceeded for this model") == ExhaustionSignal.QUOTA
    assert scan_exhaustion("maximum context length reached") == ExhaustionSignal.CONTEXT
    assert scan_exhaustion("some unrelated error") is None
    assert scan_exhaustion("") is None
    assert scan_exhaustion(None) is None


def test_order_models_respects_provider_preference(monkeypatch):
    monkeypatch.delenv("PRAXIS_PROVIDERS", raising=False)
    models = ["openrouter/openai/gpt-4o-mini", "groq/llama-3.1-8b-instant", "cerebras/x"]
    ordered = order_models(models)
    assert ordered[0] == "groq/llama-3.1-8b-instant"  # groq first by default


def test_order_models_env_override(monkeypatch):
    monkeypatch.setenv("PRAXIS_PROVIDERS", "cerebras,openrouter,groq")
    models = ["openrouter/openai/gpt-4o-mini", "groq/llama-3.1-8b-instant", "cerebras/x"]
    assert order_models(models) == [
        "cerebras/x",
        "openrouter/openai/gpt-4o-mini",
        "groq/llama-3.1-8b-instant",
    ]


def test_pool_default_order():
    pool = ProviderPool(JOB_PIPELINE)
    assert pool.ordered_providers() == ["groq", "openrouter", "cerebras"]


def test_pool_cooldown_skip_and_persist(db_session, monkeypatch):
    monkeypatch.setenv("PRAXIS_PROVIDER_COOLDOWN_S", "60")
    pool = ProviderPool(JOB_PIPELINE)
    now = providers_module._wall_now()
    from datetime import timedelta

    pool.mark_exhausted("groq", ExhaustionSignal.RATE_LIMIT)
    assert pool.is_cooling_down("groq", now=now)
    assert pool.healthy(["groq", "openrouter"]) == ["openrouter"]
    assert pool.healthy_models(["groq/a", "openrouter/b"]) == ["openrouter/b"]

    # Persisted: a fresh pool instance (simulating a restart) still sees it.
    reloaded = ProviderPool(JOB_PIPELINE)
    assert reloaded.is_cooling_down("groq", now=now)

    # Cooldown expiry clears the skip.
    expired = now + timedelta(seconds=61)
    assert not pool.is_cooling_down("groq", now=expired)

    pool.mark_healthy("groq")
    assert not pool.is_cooling_down("groq", now=now)
    recovered = get_provider_health(JOB_PIPELINE, "groq", session=db_session)
    assert recovered is not None and recovered.state == "healthy"


def test_pool_rows_recorded_for_observability(db_session):
    pool = ProviderPool(JOB_CODER)
    pool.mark_exhausted("cerebras", ExhaustionSignal.QUOTA, detail="insufficient_quota")
    rows = provider_health_rows(session=db_session)
    assert len(rows) == 1
    row = rows[0]
    assert row.job == JOB_CODER
    assert row.provider == "cerebras"
    assert row.state == "cooling_down"
    assert row.last_signal == "quota: insufficient_quota"
    assert row.cooldown_until is not None


def test_pool_jobs_are_independent(db_session):
    pipeline = ProviderPool(JOB_PIPELINE)
    coder = ProviderPool(JOB_CODER)
    pipeline.mark_exhausted("groq", ExhaustionSignal.RATE_LIMIT)
    assert pipeline.is_cooling_down("groq")
    assert not coder.is_cooling_down("groq")


def test_first_healthy_skips_cooling(db_session):
    pool = ProviderPool(JOB_PIPELINE)
    pool.mark_exhausted("groq", ExhaustionSignal.RATE_LIMIT)
    assert pool.first_healthy() == "openrouter"


def test_classify_auth_failure_variants():
    """401, AuthenticationError, and the BadRequestError invalid_api_key variant."""
    assert classify_auth_failure(_FakeError("Unauthorized", status_code=401))
    assert classify_auth_failure(_FakeError("invalid_api_key: bad key", status_code=400))
    AuthenticationError = type("AuthenticationError", (Exception,), {})
    assert classify_auth_failure(AuthenticationError("bad key"))
    assert classify_auth_failure(_FakeError("Incorrect API key provided"))
    assert not classify_auth_failure(_FakeError("connection reset by peer"))
    assert not classify_auth_failure(_FakeError("boom", status_code=500))
    assert not classify_auth_failure(_FakeError("429 too many requests"))
    assert classify_exhaustion(_FakeError("invalid_api_key", status_code=401)) is None
