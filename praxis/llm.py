"""Thin wrapper around litellm for all agent LLM calls.

Every completion that reports usage is recorded to the ``llm_usage`` table
(tokens, estimated USD cost, latency, stage, candidate) so spend can be
audited via ``praxis usage``. Recording is best-effort by design: a failed
write logs a warning and never breaks the call itself.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections.abc import Callable
from typing import Any

from litellm import completion as _default_completion
from litellm import completion_cost as _completion_cost

from praxis.db import LLMCache, LLMUsage, get_session

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "groq/llama-3.1-8b-instant"

# Env toggle for the response cache; caching is on unless set to 0/false/no/off.
CACHE_ENV = "PRAXIS_LLM_CACHE"


def _resolve_model(model: str | None) -> str:
    return model or os.environ.get("PRAXIS_MODEL") or DEFAULT_MODEL


class LLMClient:
    """Callable-completion wrapper; inject a fake for tests."""

    def __init__(
        self,
        completion: Callable[..., Any] | None = None,
        model: str | None = None,
    ) -> None:
        self._completion = completion or _default_completion
        self._model = _resolve_model(model)

    def call(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        *,
        stage: str | None = None,
        candidate_id: int | None = None,
    ) -> str:
        model = _resolve_model(model) or self._model
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                *([{"role": "system", "content": system}] if system else []),
                {"role": "user", "content": prompt},
            ],
        }
        cache_enabled = _cache_enabled()
        cache_key = _cache_key(model, system, prompt) if cache_enabled else None
        if cache_key is not None:
            cached = _cache_get(cache_key)
            if cached is not None:
                logger.debug("llm: cache hit for model=%s", model)
                _record_cache_hit(model=model, stage=stage, candidate_id=candidate_id)
                return cached

        started = time.monotonic()
        try:
            response = self._completion(**kwargs)
        except Exception as exc:  # noqa: BLE001 - record the attempt, then propagate
            _record_failure(
                exc,
                model=model,
                stage=stage,
                candidate_id=candidate_id,
                latency_ms=_elapsed_ms(started),
            )
            raise
        latency_ms = _elapsed_ms(started)
        _record_usage(
            response,
            model=model,
            stage=stage,
            candidate_id=candidate_id,
            latency_ms=latency_ms,
        )
        content = response["choices"][0]["message"]["content"]
        if cache_key is not None:
            _cache_put(cache_key, model=model, response=content)
        return content


_client: LLMClient | None = None


def get_client() -> LLMClient:
    """Return the shared module-level client (lazily created)."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def call_llm(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    *,
    stage: str | None = None,
    candidate_id: int | None = None,
    completion: Callable[..., Any] | None = None,
) -> str:
    """Call an LLM, optionally injecting a completion function for tests.

    ``stage`` (e.g. ``\"analyst\"`` or ``\"architect\"``) and ``candidate_id`` are
    recorded alongside the call so spend can be attributed per stage and per
    candidate.
    """
    client = LLMClient(completion=completion) if completion else get_client()
    return client.call(
        prompt, system=system, model=model, stage=stage, candidate_id=candidate_id
    )


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------


def _cache_enabled() -> bool:
    """Response caching is on unless PRAXIS_LLM_CACHE is 0/false/no/off."""
    raw = os.environ.get(CACHE_ENV)
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _cache_key(model: str, system: str | None, prompt: str) -> str:
    """sha256 over model + system + prompt: any change is a fresh key."""
    material = f"{model}\0{system or ''}\0{prompt}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> str | None:
    """Return the cached response for a key, or None on miss/error."""
    try:
        session = get_session()
        try:
            row = session.get(LLMCache, key)
            return row.response if row is not None else None
        finally:
            session.close()
    except Exception as exc:  # noqa: BLE001 - a cache failure is a miss, never an error
        logger.debug("llm: cache read failed: %s", exc)
        return None


def _cache_put(key: str, *, model: str, response: str) -> None:
    """Store a response, upserting on the primary key; best-effort only."""
    try:
        session = get_session()
        try:
            session.merge(LLMCache(key=key, model=model, response=response))
            session.commit()
        finally:
            session.close()
    except Exception as exc:  # noqa: BLE001 - observability/caching must never break a call
        logger.debug("llm: cache write failed: %s", exc)


def invalidate_llm_cache(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
) -> None:
    """Delete the cached response for an input, if one exists.

    Callers that determine the cached content is unusable (e.g. the Analyst
    failing to parse its JSON) use this so a transiently bad response is not
    frozen in the cache and re-served on every later run.
    """
    key = _cache_key(_resolve_model(model), system, prompt)
    try:
        session = get_session()
        try:
            row = session.get(LLMCache, key)
            if row is not None:
                session.delete(row)
                session.commit()
        finally:
            session.close()
    except Exception as exc:  # noqa: BLE001 - best-effort invalidation
        logger.debug("llm: cache invalidation failed: %s", exc)


def _record_cache_hit(*, model: str, stage: str | None, candidate_id: int | None) -> None:
    """Record a cache hit as a zero-token usage row so savings are visible."""
    try:
        session = get_session()
        try:
            session.add(
                LLMUsage(model=model, stage=stage, candidate_id=candidate_id, cached=True)
            )
            session.commit()
        finally:
            session.close()
    except Exception as exc:  # noqa: BLE001 - observability must never break a call
        logger.debug("llm: failed to record cache hit: %s", exc)


# ---------------------------------------------------------------------------
# Usage extraction and recording
# ---------------------------------------------------------------------------


def _elapsed_ms(started: float) -> int:
    """Wall-clock milliseconds since ``started`` (time.monotonic)."""
    return round((time.monotonic() - started) * 1000)


def _as_int(value: Any) -> int | None:
    """Coerce a token count to int, tolerating absent/garbage values."""
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _response_usage(response: Any) -> dict[str, int | None] | None:
    """Extract token counts from a litellm ModelResponse or a plain dict.

    Returns None when the response carries no usage info at all (e.g. an
    injected fake in tests), in which case nothing is recorded.
    """
    usage = (
        response.get("usage")
        if isinstance(response, dict)
        else getattr(response, "usage", None)
    )
    if usage is None:
        return None

    if isinstance(usage, dict):
        get = usage.get
    else:

        def get(key: str) -> Any:
            return getattr(usage, key, None)

    tokens = {
        "prompt_tokens": _as_int(get("prompt_tokens")),
        "completion_tokens": _as_int(get("completion_tokens")),
        "total_tokens": _as_int(get("total_tokens")),
    }
    if all(value is None for value in tokens.values()):
        return None
    return tokens


def _response_cost(response: Any) -> float | None:
    """Estimated USD cost of a completed call, or None when unknown.

    Prefers litellm's auto-injected ``_hidden_params[\"response_cost\"]`` and
    falls back to ``litellm.completion_cost``. A model missing from litellm's
    pricing map (or a fake response in tests) yields None rather than raising.
    """
    hidden = (
        response.get("_hidden_params")
        if isinstance(response, dict)
        else getattr(response, "_hidden_params", None)
    )
    if isinstance(hidden, dict) and hidden.get("response_cost") is not None:
        try:
            return float(hidden["response_cost"])
        except (TypeError, ValueError):
            pass
    try:
        cost = _completion_cost(completion_response=response)
        return float(cost) if cost is not None else None
    except Exception:  # noqa: BLE001 - unknown model or non-litellm response
        return None


def _record_failure(
    exc: Exception,
    *,
    model: str,
    stage: str | None,
    candidate_id: int | None,
    latency_ms: int,
) -> None:
    """Persist a failed LLM attempt (rate limit, network error, ...).

    Failed calls carry no token/cost data, but they still matter for
    observability: without this, a run whose calls mostly fail would report
    near-zero spend and hide the failures entirely.
    """
    try:
        session = get_session()
        try:
            session.add(
                LLMUsage(
                    model=model,
                    stage=stage,
                    candidate_id=candidate_id,
                    latency_ms=latency_ms,
                    error=str(exc)[:500],
                )
            )
            session.commit()
        finally:
            session.close()
    except Exception as record_exc:  # noqa: BLE001 - observability must never break the call
        logger.warning("llm: failed to record usage error: %s", record_exc)


def _record_usage(
    response: Any,
    *,
    model: str,
    stage: str | None,
    candidate_id: int | None,
    latency_ms: int,
) -> None:
    """Persist one LLM call to the usage ledger; best-effort only."""
    usage = _response_usage(response)
    if usage is None:
        return
    try:
        session = get_session()
        try:
            session.add(
                LLMUsage(
                    model=model,
                    stage=stage,
                    candidate_id=candidate_id,
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                    total_tokens=usage["total_tokens"],
                    cost_usd=_response_cost(response),
                    latency_ms=latency_ms,
                )
            )
            session.commit()
        finally:
            session.close()
    except Exception as exc:  # noqa: BLE001 - observability must never break the call
        logger.warning("llm: failed to record usage: %s", exc)
