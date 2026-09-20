"""`praxis doctor`: pre-flight checks so a fresh clone works with real keys.

Checks (in order): .env found and loaded; at least one provider key
configured; for each provider with a key — key presence (only the length is
printed, never the value), an authenticated GET of the provider's models
endpoint, and that the configured ``PRAXIS_MODEL`` exists in that provider's
list; the database path writable with a creatable schema; and the
``PRAXIS_CODER`` status. Every failure carries a one-line fix hint, and the
CLI exits non-zero when any check fails.
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
    "cerebras": "https://api.cerebras.io/v1/models",
}
MODELS_TIMEOUT_S = 10


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


def _fetch_model_ids(provider: str, key: str) -> tuple[list[str] | None, str | None]:
    """Authenticated GET of the provider's models endpoint.

    Returns ``(model_ids, None)`` on success or ``(None, reason)`` on failure.
    """
    try:
        resp = requests.get(
            PROVIDER_MODELS_ENDPOINTS[provider],
            headers={"Authorization": f"Bearer {key}"},
            timeout=MODELS_TIMEOUT_S,
        )
    except requests.RequestException as exc:
        return None, f"network error ({type(exc).__name__})"
    if resp.status_code == 200:
        try:
            data = resp.json().get("data", [])
        except ValueError:
            return None, "unexpected (non-JSON) response body"
        return [str(m.get("id", "")) for m in data], None
    if resp.status_code in (401, 403):
        return None, f"key rejected (HTTP {resp.status_code})"
    return None, f"HTTP {resp.status_code}"


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


def _check_provider(provider: str, model_id: str | None) -> list[CheckResult]:
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

    ids, error = _fetch_model_ids(provider, key)
    if error is not None:
        results.append(
            CheckResult(
                f"provider {provider} models endpoint",
                False,
                error,
                hint="check the key value for this provider, or your network/proxy",
            )
        )
        return results

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


def run_doctor_checks() -> list[CheckResult]:
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

    for provider in PROVIDER_ORDER:
        membership_id = model_id if (provider == model_provider and model_id) else None
        results.extend(_check_provider(provider, membership_id))

    results.append(_check_db())
    results.append(_check_coder())
    return results
