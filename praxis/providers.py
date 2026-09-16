"""Health-aware multi-provider routing with instant failover on exhaustion.

Praxis consumes LLM providers in two separate jobs:

* **Job A — Analyst + Architect** (`praxis/llm.py`).  Models are resolved to
  litellm model strings (``groq/...``, ``openrouter/...``, ``cerebras/...``)
  and called through litellm, which reads the provider key from the Praxis env
  (``GROQ_API_KEY`` etc., with ``PRAXIS_<PROVIDER>_API_KEY`` overrides).
* **Job B — Coder** (`praxis/agents/coder.py`).  The OpenCode CLI routes through
  the local omniroute gateway (declared in ``~/.config/opencode/opencode.json``,
  baseURL ``http://localhost:20128/v1``), which owns the provider keys and does
  backend-level failover internally.  Praxis only rotates the
  ``--model omniroute/<id>`` id it passes and detects exhaustion from the
  subprocess exit output.

Both jobs share the same health registry.  When a provider is exhausted (rate
limit, quota, context window), it is put into a cooldown that is persisted in
the ``provider_health`` table, and routing instantly skips it — no wasted LLM
latency re-trying a dead key within a batch or across a ``--resume``.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta

from praxis.db import get_provider_health, get_session, set_provider_health

logger = logging.getLogger(__name__)

# Jobs that consume providers; each has its own independent health state.
JOB_PIPELINE = "pipeline"
JOB_CODER = "coder"

# Env toggles for pool ordering + cooldown tuning.
PRAXIS_PROVIDERS_ENV = "PRAXIS_PROVIDERS"
PRAXIS_CODER_MODELS_ENV = "PRAXIS_CODER_MODELS"
COOLDOWN_ENV = "PRAXIS_PROVIDER_COOLDOWN_S"
DEFAULT_COOLDOWN_S = 60

# litellm model strings look like "<provider>/<model>"; this regex extracts the
# provider prefix that owns the key that litellm auto-reads from the env.
PROVIDER_PREFIX_RE = re.compile(r"^([^/]+)")

# Substring/status-code signals that identify a provider as exhausted rather
# than merely transiently broken (network, 5xx, auth, ...). Lossy by design: a
# false positive costs a short cooldown, a false negative wastes a retry.
RATE_LIMIT_MARKERS = re.compile(
    r"rate limit|rate_limit|too many requests|429|http.{0,3}429", re.IGNORECASE
)
QUOTA_MARKERS = re.compile(
    r"insufficient_quota|quota|402|exceeded.*(?:budget|limit)s?", re.IGNORECASE
)
CONTEXT_MARKERS = re.compile(
    r"context.?window|context.?length|context_length_exceeded|maximum context", re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Exhaustion classification
# ---------------------------------------------------------------------------


class ExhaustionSignal:
    """Named signals returned by ``classify_exhaustion``.

    ``rate_limit``/``quota`` mean the provider's key is spent and the pool
    should skip it for the cooldown window; ``context`` means the model cannot
    hold the prompt (usually per-request, still worth a short cooldown).
    """

    RATE_LIMIT = "rate_limit"
    QUOTA = "quota"
    CONTEXT = "context"

    _ALL = (RATE_LIMIT, QUOTA, CONTEXT)


def is_exhaustion_signal(signal: str | None) -> bool:
    return signal in ExhaustionSignal._ALL


def classify_exhaustion(exc: Exception) -> str | None:
    """Classify an exception from Job A (litellm) as a provider-exhaustion signal.

    Returns an :class:`ExhaustionSignal` value, or None when the failure looks
    transient (network error, 5xx, auth) rather than "this provider is spent".
    The caller puts an exhausted provider into cooldown and moves on instantly.
    """
    message = str(exc)
    status = getattr(exc, "status_code", None)
    exc_class = type(exc).__name__

    if status == 429 or RATE_LIMIT_MARKERS.search(message):
        return ExhaustionSignal.RATE_LIMIT
    if status == 402 or QUOTA_MARKERS.search(message):
        return ExhaustionSignal.QUOTA
    if CONTEXT_MARKERS.search(message) or "context_window" in exc_class.lower():
        return ExhaustionSignal.CONTEXT

    # Fall back on the exception type name (catches litellm's
    # RateLimitError/InsufficientQuotaError/ContextWindowExceededError in any
    # version without importing symbols that may not exist).
    lowered = exc_class.lower()
    if "ratelimit" in lowered or "ratelimit" in lowered.replace("_", ""):
        return ExhaustionSignal.RATE_LIMIT
    if "quota" in lowered or "insufficient" in lowered:
        return ExhaustionSignal.QUOTA
    if "context" in lowered:
        return ExhaustionSignal.CONTEXT
    return None


def scan_exhaustion(output: str | None) -> str | None:
    """Scan OpenCode subprocess output (Job B) for an exhaustion signal.

    OpenCode exits non-zero on errors, but the error strings it prints come in
    many formats; this is a lossy text scan equivalent to the exception
    classifier above.
    """
    if not output:
        return None
    if RATE_LIMIT_MARKERS.search(output):
        return ExhaustionSignal.RATE_LIMIT
    if QUOTA_MARKERS.search(output):
        return ExhaustionSignal.QUOTA
    if CONTEXT_MARKERS.search(output):
        return ExhaustionSignal.CONTEXT
    return None


def provider_of(model: str) -> str:
    """Return the provider prefix of a litellm/opencode model string.

    ``groq/llama-3.1-8b-instant`` -> ``groq``; a bare model id returns itself
    (treated as its own provider so it can still be routed/cooldowned).
    """
    match = PROVIDER_PREFIX_RE.match(model or "")
    return match.group(1) if match else model


# ---------------------------------------------------------------------------
# Health registry / provider pool
# ---------------------------------------------------------------------------


class ProviderPool:
    """Health-aware ordered pool of providers for one job.

    ``pick()`` returns the first healthy provider (an exhausted one is in
    cooldown and skipped instantly). Callers use the returned provider both to
    select a model and to handle cooldown bookkeeping via ``mark_exhausted`` /
    ``mark_healthy``.
    """

    def __init__(self, job: str, *, cooldown_s: int | None = None) -> None:
        self.job = job
        self._cooldown_s = cooldown_s

    # -- cooldown config ----------------------------------------------------

    @property
    def cooldown_s(self) -> int:
        return self._cooldown_s if self._cooldown_s is not None else _resolve_cooldown_s()

    # -- queries ------------------------------------------------------------

    def cooldown_until(self, provider: str) -> datetime | None:
        """Return the persisted cooldown deadline (naive UTC) or None."""
        row = self._row(provider)
        if row is None or row.cooldown_until is None:
            return None
        return row.cooldown_until

    def is_cooling_down(self, provider: str, *, now=None) -> bool:
        """True when the provider is in cooldown as of ``now`` (monotonic-safe via wall clock)."""
        until = self.cooldown_until(provider)
        if until is None:
            return False
        now = now or _wall_now()
        return until > now

    def healthy(self, providers: list[str]) -> list[str]:
        """Return ``providers`` (in given order) minus any currently in cooldown."""
        return [p for p in providers if not self.is_cooling_down(p)]

    def healthy_models(self, models: list[str]) -> list[str]:
        """Return model strings whose provider is not currently in cooldown."""
        return [m for m in models if not self.is_cooling_down(provider_of(m))]

    def ordered_providers(self) -> list[str]:
        """Providers for this job in preference order from env (Job A) / configured models (Job B).

        For the pipeline the order comes from ``PRAXIS_PROVIDERS`` (comma
        separated); for the coder it comes from ``PRAXIS_CODER_MODELS``
        (comma-separated ``provider/model`` ids, mapping to their prefixes).
        Falls back to a sensible default when unset so tests and fresh installs
        behave deterministically.
        """
        if self.job == JOB_CODER:
            env = PRAXIS_CODER_MODELS_ENV
        else:
            env = PRAXIS_PROVIDERS_ENV
        return _provider_order(_env_list(env)) or _default_providers()

    def first_healthy(self, providers: list[str] | None = None) -> str | None:
        """Return the first provider from the pool that is not in cooldown, or None."""
        for provider in self.ordered_providers():
            if providers is not None and provider not in providers:
                continue
            if not self.is_cooling_down(provider):
                return provider
        return None

    # -- mutations ----------------------------------------------------------

    def mark_exhausted(self, provider: str, signal: str, *, detail: str | None = None) -> None:
        """Put a provider into cooldown for the configured window (persisted)."""
        until = _wall_now() + timedelta(seconds=self.cooldown_s)
        note = f"{signal}: {detail}" if detail else signal
        session = get_session()
        try:
            set_provider_health(
                self.job,
                provider,
                state="cooling_down",
                cooldown_until=until,
                last_signal=note,
                session=session,
            )
        except Exception as exc:  # noqa: BLE001 - health must never break a call
            logger.warning(
                "providers: failed to persist cooldown for %s/%s: %s", self.job, provider, exc
            )
        finally:
            session.close()
        logger.warning(
            "[providers] %s cooling down (%s) until %s; instant-switching away",
            provider,
            note,
            until.strftime("%H:%M:%S"),
        )

    def mark_healthy(self, provider: str) -> None:
        session = get_session()
        try:
            set_provider_health(
                self.job,
                provider,
                state="healthy",
                cooldown_until=None,
                last_signal=None,
                session=session,
            )
        except Exception as exc:  # noqa: BLE001 - health must never break a call
            logger.warning(
                "providers: failed to persist healthy state for %s/%s: %s", self.job, provider, exc
            )
        finally:
            session.close()

    # -- internals ----------------------------------------------------------

    def _row(self, provider: str):
        session = get_session()
        try:
            return get_provider_health(self.job, provider, session=session)
        except Exception:  # noqa: BLE001 - a read failure means "assume healthy"
            logger.debug(
                "providers: health read failed for %s/%s", self.job, provider, exc_info=True
            )
            return None
        finally:
            session.close()


def _resolve_cooldown_s() -> int:
    raw = _env_first(COOLDOWN_ENV)
    if raw is not None:
        try:
            return max(1, int(raw))
        except ValueError:
            logger.warning("invalid %s=%r; using default", COOLDOWN_ENV, raw)
    return DEFAULT_COOLDOWN_S


def _env_first(name: str) -> str | None:
    import os

    return os.environ.get(name)


def _env_list(name: str) -> list[str]:
    import os

    raw = os.environ.get(name)
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def order_models(models: list[str], providers: list[str] | None = None) -> list[str]:
    """Stable-sort litellm model strings by provider preference order.

    Providers not in the preference list sort after all listed ones, keeping
    their original relative order. Used by Job A to try the preferred provider
    first while still honoring configured fallbacks.
    """
    order = providers or _provider_order(_env_list(PRAXIS_PROVIDERS_ENV)) or _default_providers()
    rank = {provider: i for i, provider in enumerate(order)}

    def _rank_of(model: str) -> int:
        return rank.get(provider_of(model), len(order))

    return sorted(models, key=_rank_of)


def _provider_order(entries: list[str]) -> list[str]:
    """Map a list of models/ids to distinct provider prefixes while preserving order."""
    seen: list[str] = []
    for entry in entries:
        provider = provider_of(entry)
        if provider not in seen:
            seen.append(provider)
    return seen


def _default_providers() -> list[str]:
    """Default preference order matching the documented 3-key setup."""
    return ["groq", "openrouter", "cerebras"]


def _wall_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)
