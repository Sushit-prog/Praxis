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
from praxis.providers import (
    AUTH_SIGNAL,
    JOB_PIPELINE,
    AllProvidersCoolingDownError,
    NoWorkingProviderError,
    ProviderPool,
    classify_auth_failure,
    classify_exhaustion,
    order_models,
    provider_of,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "groq/openai/gpt-oss-20b"

# Env toggle for the response cache; caching is on unless set to 0/false/no/off.
CACHE_ENV = "PRAXIS_LLM_CACHE"

# Truncation guard: max_tokens for the first attempt (unset = provider default)
# and for the one retry after a truncated response (default: 2x the first
# attempt when set, else DEFAULT_RETRY_MAX_TOKENS).
MAX_TOKENS_ENV = "PRAXIS_MAX_TOKENS"
MAX_TOKENS_RETRY_ENV = "PRAXIS_MAX_TOKENS_RETRY"
DEFAULT_RETRY_MAX_TOKENS = 8192


class TruncatedOutputError(RuntimeError):
    """The LLM response was truncated even after a higher-max_tokens retry.

    Raised instead of returning (and never caching or persisting) a partial
    output: a blueprint that ends mid-sentence is worse than no blueprint.
    """


def _initial_max_tokens() -> int | None:
    """First-attempt max_tokens from PRAXIS_MAX_TOKENS (None = provider default)."""
    raw = os.environ.get(MAX_TOKENS_ENV)
    if not raw:
        return None
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning("invalid %s=%r; ignoring", MAX_TOKENS_ENV, raw)
        return None


def _retry_max_tokens(first: int | None) -> int:
    """max_tokens for the truncation retry: env override, else 2x first, else default."""
    raw = os.environ.get(MAX_TOKENS_RETRY_ENV)
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            logger.warning("invalid %s=%r; ignoring", MAX_TOKENS_RETRY_ENV, raw)
    if first is not None:
        return first * 2
    return DEFAULT_RETRY_MAX_TOKENS


def _truncation_reason(response: Any) -> str | None:
    """Why a response counts as truncated, or None when it is complete.

    Two signals: ``finish_reason == "length"`` (the model ran out of output
    tokens mid-text), and empty content — reasoning models can spend the
    whole budget on invisible reasoning and return nothing.
    """
    if isinstance(response, dict):
        choices = response.get("choices") or []
        choice = choices[0] if choices else None
    else:
        choices = getattr(response, "choices", None) or []
        choice = choices[0] if choices else None
    if choice is None:
        return None
    if isinstance(choice, dict):
        finish = choice.get("finish_reason")
        content = (choice.get("message") or {}).get("content")
    else:
        finish = getattr(choice, "finish_reason", None)
        content = getattr(getattr(choice, "message", None), "content", None)
    if finish == "length":
        return "finish_reason=length"
    if content is None or (isinstance(content, str) and not content.strip()):
        return "empty content (reasoning consumed the token budget)"
    return None

# Upper bound (seconds) for waiting out provider cooldowns when every provider
# is rate-limited: wait for the earliest cooldown to expire, but never stall a
# run longer than this — afterwards the run aborts with candidates retryable.
MAX_COOLDOWN_WAIT_S = 90.0

# Comma-separated models tried after the primary when it fails (rate limit, outage).
FALLBACKS_ENV = "PRAXIS_FALLBACK_MODELS"


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
            "messages": [
                *([{"role": "system", "content": system}] if system else []),
                {"role": "user", "content": prompt},
            ],
        }
        # Health-aware chain: skip cooling-down providers instantly so an
        # exhausted key never costs a wasted LLM call. The cache is checked
        # over the FULL chain first — a cached response is free even from a
        # provider currently in cooldown.
        pool = _get_pool()
        chain = order_models([model, *_fallback_models()])
        cache_enabled = _cache_enabled()
        if cache_enabled:
            for cached_model in chain:
                attempt_key = _cache_key(cached_model, system, prompt)
                cached = _cache_get(attempt_key)
                if cached is not None:
                    logger.debug("llm: cache hit for model=%s", cached_model)
                    _record_cache_hit(model=cached_model, stage=stage, candidate_id=candidate_id)
                    return cached

        attempt_chain = pool.healthy_models(chain)
        if not attempt_chain:
            # Every provider is rate-limited: wait for the earliest cooldown to
            # expire (bounded) rather than failing the candidate outright — a
            # rate limit is transient, and the candidate should be retried.
            wait_s = pool.seconds_until_recovery([provider_of(m) for m in chain])
            if wait_s is not None:
                bounded = min(wait_s + 0.5, MAX_COOLDOWN_WAIT_S)
                logger.warning(
                    "llm: all providers cooling down; waiting %.0fs for the "
                    "earliest cooldown to expire (bound %.0fs)",
                    bounded,
                    MAX_COOLDOWN_WAIT_S,
                )
                time.sleep(bounded)
            attempt_chain = pool.healthy_models(chain)
        if not attempt_chain:
            raise AllProvidersCoolingDownError(
                f"all LLM providers are cooling down after waiting up to "
                f"{MAX_COOLDOWN_WAIT_S:.0f}s "
                f"({', '.join(provider_of(m) for m in chain)}); "
                "re-run with --resume later — candidates were not marked failed"
            )
        errors: list[Exception] = []
        auth_failures: list[tuple[str, str]] = []
        for attempt_model in attempt_chain:
            kwargs["model"] = attempt_model
            _inject_provider_key(kwargs, attempt_model)
            first_tokens = _initial_max_tokens()
            if first_tokens is not None:
                kwargs["max_tokens"] = first_tokens
            else:
                kwargs.pop("max_tokens", None)
            started = time.monotonic()
            try:
                response = self._completion(**kwargs)
                reason = _truncation_reason(response)
                if reason is not None:
                    # Truncation is not a provider failure: retry the SAME
                    # model once with a higher token budget instead of
                    # failing over or returning a partial output.
                    _record_failure(
                        RuntimeError(f"truncated: {reason}"),
                        model=attempt_model,
                        stage=stage,
                        candidate_id=candidate_id,
                        latency_ms=_elapsed_ms(started),
                    )
                    retry_tokens = _retry_max_tokens(kwargs.get("max_tokens"))
                    logger.warning(
                        "llm: %s output truncated (%s); retrying once with "
                        "max_tokens=%s",
                        attempt_model,
                        reason,
                        retry_tokens,
                    )
                    retry_kwargs = dict(kwargs)
                    retry_kwargs["max_tokens"] = retry_tokens
                    response = self._completion(**retry_kwargs)
                    retry_reason = _truncation_reason(response)
                    if retry_reason is not None:
                        raise TruncatedOutputError(
                            f"LLM output truncated twice from {attempt_model} "
                            f"({retry_reason}) even with max_tokens={retry_tokens}; "
                            "refusing to return or store a partial output. "
                            f"Raise {MAX_TOKENS_RETRY_ENV} or shorten the prompt."
                        )
                    # Usage for the successful retry is recorded by the normal
                    # post-loop path (latency covers both attempts).
                break
            except TruncatedOutputError:
                raise
            except Exception as exc:  # noqa: BLE001 - record attempt, try next model
                errors.append(exc)
                _record_failure(
                    exc,
                    model=attempt_model,
                    stage=stage,
                    candidate_id=candidate_id,
                    latency_ms=_elapsed_ms(started),
                )
                provider = provider_of(attempt_model)
                if classify_auth_failure(exc):
                    # A rejected key is a provider-level failure: mark the
                    # provider unhealthy in the persisted health table and
                    # fail over to the next configured provider instantly.
                    pool.mark_exhausted(provider, AUTH_SIGNAL, detail=str(exc)[:300])
                    auth_failures.append((provider, str(exc).splitlines()[0][:160]))
                    logger.warning(
                        "llm: %s rejected the key (auth failure); switching to next provider",
                        attempt_model,
                    )
                    continue
                signal = classify_exhaustion(exc)
                if signal is not None:
                    pool.mark_exhausted(
                        provider,
                        signal,
                        detail=str(exc)[:300],
                    )
                    logger.warning(
                        "llm: %s exhausted (%s); switching to next provider",
                        attempt_model,
                        signal,
                    )
                else:
                    logger.warning("llm: model %s failed (%s)", attempt_model, exc)
        else:
            if auth_failures and len(auth_failures) == len(attempt_chain):
                # Every configured provider rejected the key: abort the run —
                # retrying cannot help until the keys are fixed.
                provider, reason = auth_failures[-1]
                raise NoWorkingProviderError(
                    f"no working provider: {provider}: {reason}"
                ) from errors[-1]
            raise errors[-1]

        provider = provider_of(attempt_model)
        if pool.is_cooling_down(provider):
            pool.mark_healthy(provider)
        latency_ms = _elapsed_ms(started)
        _record_usage(
            response,
            model=attempt_model,
            stage=stage,
            candidate_id=candidate_id,
            latency_ms=latency_ms,
        )
        content = response["choices"][0]["message"]["content"]
        if cache_enabled:
            _cache_put(
                _cache_key(attempt_model, system, prompt),
                model=attempt_model,
                response=content,
            )
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
    return client.call(prompt, system=system, model=model, stage=stage, candidate_id=candidate_id)


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------


def _cache_enabled() -> bool:
    """Response caching is on unless PRAXIS_LLM_CACHE is 0/false/no/off."""
    raw = os.environ.get(CACHE_ENV)
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _fallback_models() -> list[str]:
    """Models tried after the primary, from PRAXIS_FALLBACK_MODELS (comma-separated)."""
    raw = os.environ.get(FALLBACKS_ENV)
    if not raw:
        return []
    return [model.strip() for model in raw.split(",") if model.strip()]


def _get_pool() -> ProviderPool:
    """Return the shared provider pool for the Analyst/Architect job."""
    global _pool
    if _pool is None:
        _pool = ProviderPool(JOB_PIPELINE)
    return _pool


_pool: ProviderPool | None = None


def _inject_provider_key(kwargs: dict[str, Any], model: str) -> None:
    """When a per-provider API key override is set, inject it into litellm kwargs."""
    key_env = f"PRAXIS_{provider_of(model).upper()}_API_KEY"
    key = os.environ.get(key_env)
    if key:
        kwargs["api_key"] = key


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
            session.add(LLMUsage(model=model, stage=stage, candidate_id=candidate_id, cached=True))
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
        response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
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
