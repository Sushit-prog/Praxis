"""`praxis doctor`: pre-flight checks so a fresh clone works with real keys.

Checks (in order): .env found and loaded; at least one provider key
configured; for each provider with a key — key presence (only the length is
printed, never the value), an authenticated GET of the provider's models
endpoint, and that the configured ``PRAXIS_MODEL`` exists in that provider's
list; the database path writable with a creatable schema; and the
``PRAXIS_CODER`` status. Every failure carries a one-line fix hint, and the
CLI exits non-zero when any check fails.

With ``--deep``, one extra check per PRAXIS_DESIGN_MODEL chain entry sends a
1-token completion and reports real failures (payment required, model
unavailable, auth) with a fix hint; plain `praxis doctor` is unchanged.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

import requests

# Job A providers and their cheap authenticated models endpoints.
PROVIDER_ORDER = ("groq", "openrouter", "cerebras")
PROVIDER_MODELS_ENDPOINTS = {
    "groq": "https://api.groq.com/openai/v1/models",
    "openrouter": "https://openrouter.ai/api/v1/models",
    "cerebras": "https://api.cerebras.ai/v1/models",
}
MODELS_TIMEOUT_S = 10

# --deep probe: the smallest possible chat completion per chain entry.
DEEP_PROBE_TIMEOUT_S = 30
DEEP_PROBE_MAX_TOKENS = 1
DEEP_PROBE_PROMPT = "ping"


@dataclass
class CheckResult:
    """One checklist line: ok/fail/skip, a detail string, and a fix hint."""

    name: str
    ok: bool = True
    detail: str = ""
    hint: str = ""
    skipped: bool = False


def _provider_key(provider: str) -> str | None:
    """Resolve the API key for a provider: PRAXIS_<P>_API_KEY, else <P>_API_KEY."""
    override = os.environ.get(f"PRAXIS_{provider.upper()}_API_KEY")
    if override:
        return override
    return os.environ.get(f"{provider.upper()}_API_KEY") or None


@dataclass
class ModelsFetchResult:
    """Outcome of a models-endpoint probe: ids, or a classified failure."""

    model_ids: list[str] | None = None
    error_kind: str | None = None  # "network" | "auth" | "http"
    error_detail: str = ""


def _fetch_model_ids(provider: str, key: str) -> ModelsFetchResult:
    """Authenticated GET of the provider's models endpoint.

    Returns the model ids on success, or a classified failure (network = the
    provider could not be reached at all; auth = it answered and rejected the
    key; http = any other error status) so the fix hint can tell the two
    apart.
    """
    try:
        resp = requests.get(
            PROVIDER_MODELS_ENDPOINTS[provider],
            headers={"Authorization": f"Bearer {key}"},
            timeout=MODELS_TIMEOUT_S,
        )
    except requests.RequestException as exc:
        return ModelsFetchResult(
            error_kind="network",
            error_detail=f"could not reach the provider ({type(exc).__name__})",
        )
    if resp.status_code == 200:
        try:
            data = resp.json().get("data", [])
        except ValueError:
            return ModelsFetchResult(
                error_kind="http", error_detail="unexpected (non-JSON) response body"
            )
        return ModelsFetchResult(model_ids=[str(m.get("id", "")) for m in data])
    if resp.status_code in (401, 403):
        return ModelsFetchResult(
            error_kind="auth", error_detail=f"key rejected (HTTP {resp.status_code})"
        )
    return ModelsFetchResult(error_kind="http", error_detail=f"HTTP {resp.status_code}")


def _check_env() -> CheckResult:
    from praxis.cli import _load_env

    path = _load_env()
    if path:
        return CheckResult(".env", True, f"found and loaded from {path}")
    return CheckResult(
        ".env",
        False,
        "not found",
        hint="copy .env.example to .env and fill in your keys "
        "(or export the provider keys in your shell)",
    )


def _check_provider(
    provider: str, model_id: str | None, fetched: ModelsFetchResult
) -> list[CheckResult]:
    key = _provider_key(provider)
    if not key:
        return [
            CheckResult(
                f"provider {provider}",
                skipped=True,
                detail="no key configured (PRAXIS_"
                f"{provider.upper()}_API_KEY or {provider.upper()}_API_KEY)",
            )
        ]

    results: list[CheckResult] = [
        CheckResult(
            f"provider {provider}",
            True,
            f"key present ({len(key)} chars, never printed)",
        )
    ]

    result = fetched
    if result.error_kind is not None:
        hints = {
            "network": "could not reach the provider: check network/VPN/proxy",
            "auth": "check the key value for this provider",
            "http": "the provider answered with an error; check its status page or try again later",
        }
        results.append(
            CheckResult(
                f"provider {provider} models endpoint",
                False,
                result.error_detail,
                hint=hints[result.error_kind],
            )
        )
        return results

    ids = result.model_ids or []
    results.append(
        CheckResult(
            f"provider {provider} models endpoint",
            True,
            f"authenticated ok ({len(ids)} models)",
        )
    )

    if model_id is not None:
        if model_id in ids:
            results.append(
                CheckResult(
                    f"provider {provider} model",
                    True,
                    f"configured model {model_id!r} is available",
                )
            )
        else:
            results.append(
                CheckResult(
                    f"provider {provider} model",
                    False,
                    f"configured model {model_id!r} is not in the provider's list",
                    hint="set PRAXIS_MODEL to a model id this provider serves "
                    "(see the models endpoint output or the provider docs)",
                )
            )
    return results


def _check_db() -> CheckResult:
    from sqlalchemy import text

    from praxis.db import _db_url, get_engine, init_db

    url = _db_url()
    engine = None
    try:
        engine = get_engine()
        init_db(engine)  # idempotent: creates missing tables only
        with engine.connect() as conn:
            conn.execute(text("CREATE TABLE IF NOT EXISTS _praxis_doctor_probe (id INTEGER)"))
            conn.execute(text("DROP TABLE _praxis_doctor_probe"))
            conn.commit()
    except Exception as exc:  # noqa: BLE001 - a broken DB is the finding, not a crash
        return CheckResult(
            "database",
            False,
            f"{url}: {type(exc).__name__}: {exc}",
            hint="check that the PRAXIS_DB_PATH/PRAXIS_DB_URL location exists and is writable",
        )
    finally:
        if engine is not None:
            engine.dispose()
    return CheckResult("database", True, f"schema creatable and writable at {url}")


def _check_coder() -> CheckResult:
    from praxis.agents.coder import CODER_MODE_ENV, resolve_coder_mode

    raw = os.environ.get(CODER_MODE_ENV)
    mode = resolve_coder_mode()
    if raw is not None and raw.strip().lower() not in ("off", "opencode"):
        return CheckResult(
            f"PRAXIS_CODER ({CODER_MODE_ENV}={raw!r})",
            False,
            "invalid value",
            hint="set PRAXIS_CODER to off or opencode",
        )
    if mode == "opencode":
        if shutil.which("opencode") is None:
            return CheckResult(
                "PRAXIS_CODER",
                False,
                "opencode mode, but the `opencode` CLI is not on PATH",
                hint="install opencode (or set PRAXIS_CODER=off)",
            )
        return CheckResult("PRAXIS_CODER", True, "opencode mode, `opencode` CLI found")
    return CheckResult(
        "PRAXIS_CODER", True, "off (default) — the pipeline stops at the blueprint"
    )


def _check_design_chain(
    fetched_by_provider: dict[str, ModelsFetchResult],
) -> list[CheckResult]:
    """Verify each PRAXIS_DESIGN_MODEL chain entry exists at its provider.

    The chain (comma-separated provider/model ids) fails over on rate limits,
    so a typo'd entry silently weakens the design run; each entry gets its own
    warning line. Providers without a configured key are skipped (the
    per-provider checks above already report that). Model ids are shared with
    the per-provider checks (one fetch per provider).
    """
    from praxis.design import resolve_design_model_chain

    results: list[CheckResult] = []
    for entry in resolve_design_model_chain(None):
        provider, _, model_id = entry.partition("/")
        key = _provider_key(provider)
        if not key:
            results.append(
                CheckResult(
                    f"design chain {provider}",
                    skipped=True,
                    detail=f"{entry}: no {provider.upper()}_API_KEY configured; cannot verify",
                )
            )
            continue
        fetched = fetched_by_provider.get(provider)
        if fetched is None or fetched.error_kind is not None or not fetched.model_ids:
            results.append(
                CheckResult(
                    f"design chain {provider}",
                    skipped=True,
                    detail=f"{entry}: models endpoint unavailable; cannot verify",
                )
            )
            continue
        if model_id in fetched.model_ids:
            results.append(
                CheckResult(
                    f"design chain {provider}",
                    True,
                    f"{entry} is available",
                )
            )
        else:
            results.append(
                CheckResult(
                    f"design chain {provider}",
                    False,
                    f"{entry} does not exist at {provider}",
                    hint="fix or remove the entry in PRAXIS_DESIGN_MODEL — a bad chain "
                    "entry silently weakens rate-limit failover",
                )
            )
    return results


# ---------------------------------------------------------------------------
# --deep: live 1-token completion per design-chain entry
# ---------------------------------------------------------------------------


def _classify_deep_failure(status_code: int | None, message: str) -> tuple[str, str]:
    """(short label, fix hint) for a failed deep probe."""
    lowered = message.lower()
    if status_code == 401 or status_code == 403 or "auth" in lowered or "api key" in lowered:
        return "auth", "check the provider API key in .env"
    if status_code == 402 or "quota" in lowered or "billing" in lowered or "payment" in lowered:
        return (
            "payment required",
            "billing rejected the request — check the account's plan/balance",
        )
    if (
        status_code == 404
        or "does not exist" in lowered
        or "not found" in lowered
        or "decommissioned" in lowered
    ):
        return (
            "model unavailable",
            "fix or remove the entry in PRAXIS_DESIGN_MODEL (see the provider's model list)",
        )
    if status_code == 429 or "rate limit" in lowered:
        return (
            "rate limited",
            "the probe was throttled; the entry works — retry the doctor later",
        )
    return "request failed", "check the provider's status page and the model id spelling"


def _completion_for_probe():
    """The completion callable used by deep probes (indirection for tests)."""
    from litellm import completion as _completion

    return _completion


def _probe_chain_entry(entry: str) -> CheckResult:
    """Send a 1-token completion to one chain entry; classify the outcome."""
    from praxis.llm import _inject_provider_key

    completion = _completion_for_probe()
    provider, _, model_id = entry.partition("/")
    kwargs: dict = {
        "model": entry,
        "messages": [{"role": "user", "content": DEEP_PROBE_PROMPT}],
        "max_tokens": DEEP_PROBE_MAX_TOKENS,
        "timeout": DEEP_PROBE_TIMEOUT_S,
    }
    _inject_provider_key(kwargs, entry)
    try:
        completion(**kwargs)
    except Exception as exc:  # noqa: BLE001 - any failure is the finding
        status_code = getattr(exc, "status_code", None)
        message = str(exc)[:300]
        kind, hint = _classify_deep_failure(status_code, message)
        return CheckResult(
            f"design chain probe {provider}",
            False,
            f"{entry}: {kind} ({message.splitlines()[0][:120]})",
            hint=hint,
        )
    return CheckResult(
        f"design chain probe {provider}",
        True,
        f"{entry}: 1-token completion succeeded",
    )


def _check_design_chain_deep() -> list[CheckResult]:
    """Live-probe every PRAXIS_DESIGN_MODEL chain entry with a 1-token call."""
    from praxis.design import resolve_design_model_chain

    results: list[CheckResult] = []
    for entry in resolve_design_model_chain(None):
        provider, _, model_id = entry.partition("/")
        if not _provider_key(provider):
            results.append(
                CheckResult(
                    f"design chain probe {provider}",
                    skipped=True,
                    detail=f"{entry}: no {provider.upper()}_API_KEY configured; cannot probe",
                )
            )
            continue
        results.append(_probe_chain_entry(entry))
    return results


def run_doctor_checks(*, deep: bool = False) -> list[CheckResult]:
    """Run every pre-flight check and return the checklist results in order."""
    from praxis.llm import _resolve_model

    results: list[CheckResult] = [_check_env()]

    model = _resolve_model(None)  # PRAXIS_MODEL or the code default
    # Only the provider that owns the configured model gets a membership check.
    model_provider, _, model_id = model.partition("/")
    keys_configured = any(_provider_key(p) for p in PROVIDER_ORDER)
    if not keys_configured:
        results.append(
            CheckResult(
                "provider keys",
                False,
                "no provider key configured",
                hint="add GROQ_API_KEY= (and/or OPENROUTER_API_KEY=, CEREBRAS_API_KEY=) "
                "to .env — see .env.example",
            )
        )
    else:
        results.append(CheckResult("provider keys", True, "at least one key configured"))

    # One models-endpoint fetch per provider, shared by the per-provider and
    # design-chain checks.
    fetched_by_provider: dict[str, ModelsFetchResult] = {}
    for provider in PROVIDER_ORDER:
        key = _provider_key(provider)
        if key:
            fetched_by_provider[provider] = _fetch_model_ids(provider, key)
        membership_id = model_id if (provider == model_provider and model_id) else None
        results.extend(
            _check_provider(
                provider,
                membership_id,
                fetched_by_provider.get(provider)
                or ModelsFetchResult(error_kind="skip", error_detail="no key"),
            )
        )

    results.extend(_check_design_chain(fetched_by_provider))
    if deep:
        results.extend(_check_design_chain_deep())
    results.append(_check_db())
    results.append(_check_coder())
    return results
