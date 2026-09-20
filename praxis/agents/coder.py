"""Coder: turn a blueprint's first phase into a draft prototype via OpenCode CLI."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

from praxis.db import Blueprint, Candidate, get_session
from praxis.providers import JOB_CODER, ProviderPool, provider_of, scan_exhaustion

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 600
TIMEOUT_ENV = "PRAXIS_CODER_TIMEOUT_S"
SCRATCH_ROOT_ENV = "PRAXIS_SCRATCH_ROOT"
DEFAULT_SCRATCH_ROOT = Path("./scratch")

DEFAULT_MAX_FAILURES = 2
MAX_FAILURES_ENV = "PRAXIS_CODER_MAX_FAILURES"
DEFAULT_COOLDOWN_S = 300.0
COOLDOWN_ENV = "PRAXIS_CODER_COOLDOWN_S"

# Comma-separated opencode model ids (provider/model) tried in order on
# provider exhaustion; empty means "use opencode's default model, no rotation".
CODER_MODELS_ENV = "PRAXIS_CODER_MODELS"
DEFAULT_PROVIDER_RETRIES = 3
PROVIDER_RETRIES_ENV = "PRAXIS_CODER_PROVIDER_RETRIES"

# Extra flags passed to `opencode run` after the subcommand. Defaults to
# `--auto` (stock opencode's non-interactive/auto-approve flag); the fork build
# on this machine drops unknown flags, so set it to empty in the project env.
OPENCODE_FLAGS_ENV = "PRAXIS_CODER_OPENCODE_FLAGS"
DEFAULT_OPENCODE_FLAGS = "--auto"

# How much of opencode's stderr is logged on failure: the first lines (often
# the config/banner) and the tail (where the actual error line lives).
_STDERR_HEAD_LINES = 10
_STDERR_TAIL_LINES = 15

# Coder stage mode: "off" (default) ends the pipeline at the blueprint (the
# exportable artifact); "opencode" drafts prototypes via the OpenCode CLI.
# Opt in per run with `praxis run --prototype`, or persistently with env.
CODER_MODE_ENV = "PRAXIS_CODER"
DEFAULT_CODER_MODE = "off"
VALID_CODER_MODES = ("off", "opencode")


def resolve_coder_mode(prototype: bool | None = None) -> str:
    """Resolve the Coder stage mode from the --prototype flag and PRAXIS_CODER.

    Order: an explicit --prototype decision (True/False) wins; otherwise the
    PRAXIS_CODER env var (off|opencode); otherwise "off". Invalid env values
    log a warning and fall back to the default, so a typo never silently
    launches the OpenCode subprocess.
    """
    if prototype is not None:
        return "opencode" if prototype else "off"
    raw = os.environ.get(CODER_MODE_ENV)
    if raw is None:
        return DEFAULT_CODER_MODE
    mode = raw.strip().lower()
    if mode in VALID_CODER_MODES:
        return mode
    logger.warning("invalid %s=%r; using default %r", CODER_MODE_ENV, raw, DEFAULT_CODER_MODE)
    return DEFAULT_CODER_MODE


class _CircuitBreaker:
    """Fail-fast guard around the OpenCode subprocess.

    After ``max_failures`` consecutive failures the circuit opens and further
    calls return without touching the subprocess until ``cooldown_s`` elapses,
    at which point one trial attempt is allowed (half-open). This stops a
    runaway or persistently-broken OpenCode from burning time on every
    candidate in a batch.
    """

    def __init__(self, max_failures: int, cooldown_s: float) -> None:
        self.max_failures = max_failures
        self.cooldown_s = cooldown_s
        self._failures = 0
        self._open_until = 0.0
        self._trial = False  # True while a half-open trial attempt is in flight

    def allow(self) -> bool:
        if self._failures >= self.max_failures:
            if time.monotonic() >= self._open_until:
                self._failures = 0  # half-open: permit one trial attempt
                self._trial = True
                return True
            return False
        # Closed path: clear any stale trial flag left by an exception that
        # escaped between allow() and record_* (e.g. opencode missing), so a
        # later normal failure cannot be misclassified as a trial failure.
        self._trial = False
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._trial = False

    def record_failure(self) -> None:
        if self._trial:
            # A half-open trial failed: re-open the circuit immediately with a
            # fresh cooldown instead of quietly closing it (one more failure
            # must not be needed to re-trip).
            self._trial = False
            self._failures = self.max_failures
            self._open_until = time.monotonic() + self.cooldown_s
            return
        self._failures += 1
        if self._failures >= self.max_failures:
            self._open_until = time.monotonic() + self.cooldown_s


def _resolve_max_failures() -> int:
    raw = os.environ.get(MAX_FAILURES_ENV)
    if raw is not None:
        try:
            return max(1, int(raw))
        except ValueError:
            logger.warning("invalid %s=%r; using default", MAX_FAILURES_ENV, raw)
    return DEFAULT_MAX_FAILURES


def _resolve_cooldown() -> float:
    raw = os.environ.get(COOLDOWN_ENV)
    if raw is not None:
        try:
            return max(0.0, float(raw))
        except ValueError:
            logger.warning("invalid %s=%r; using default", COOLDOWN_ENV, raw)
    return DEFAULT_COOLDOWN_S


_breaker: _CircuitBreaker | None = None


def _get_breaker() -> _CircuitBreaker:
    """Return the module-level circuit breaker (lazily built from env)."""
    global _breaker
    if _breaker is None:
        _breaker = _CircuitBreaker(_resolve_max_failures(), _resolve_cooldown())
    return _breaker


PHASED_PLAN_HEADING = "phased build plan"
MILESTONE_RE = re.compile(r"^\s*\d+[.)]\s")
SUBITEM_RE = re.compile(r"^\s*[-*]\s")


def _resolve_timeout(timeout: float | None) -> float:
    if timeout is not None:
        return timeout
    raw = os.environ.get(TIMEOUT_ENV)
    if raw is not None:
        try:
            return float(raw)
        except ValueError:
            logger.warning("invalid %s=%r; using default", TIMEOUT_ENV, raw)
    return DEFAULT_TIMEOUT_S


def _resolve_scratch_root(scratch_root: Path | None) -> Path:
    if scratch_root is not None:
        return scratch_root
    raw = os.environ.get(SCRATCH_ROOT_ENV)
    if raw:
        return Path(raw)
    return DEFAULT_SCRATCH_ROOT


def _resolve_provider_retries() -> int:
    raw = os.environ.get(PROVIDER_RETRIES_ENV)
    if raw is not None:
        try:
            return max(1, int(raw))
        except ValueError:
            logger.warning("invalid %s=%r; using default", PROVIDER_RETRIES_ENV, raw)
    return DEFAULT_PROVIDER_RETRIES


def _resolve_coder_models() -> list[str]:
    """Comma-separated opencode model ids (provider/model); [] means use opencode default."""
    raw = os.environ.get(CODER_MODELS_ENV)
    if not raw:
        return []
    return [model.strip() for model in raw.split(",") if model.strip()]


def _first_paragraph(lines: list[str]) -> str:
    para: list[str] = []
    for line in lines:
        if not line.strip():
            if para:
                break
            continue
        para.append(line)
    return "\n".join(para).strip()


def _first_milestone(section: list[str]) -> str:
    collected: list[str] = []
    found = False
    for line in section:
        if not found:
            if MILESTONE_RE.match(line) or SUBITEM_RE.match(line):
                found = True
                collected.append(line)
        elif MILESTONE_RE.match(line):
            break
        else:
            collected.append(line)
    if not collected:
        return _first_paragraph(section)
    return "\n".join(collected).strip()


def _extract_first_phase(md: str) -> str:
    """Return the first milestone from a blueprint's Phased Build Plan."""
    if not md.strip():
        return ""
    lines = md.splitlines()
    start = None
    for i, line in enumerate(lines):
        lowered = line.strip().lower()
        if lowered.startswith("##") and PHASED_PLAN_HEADING in lowered:
            start = i
            break
    if start is None:
        return _first_paragraph(lines)
    section: list[str] = []
    for line in lines[start + 1 :]:
        if line.strip().lower().startswith("##"):
            break
        section.append(line)
    return _first_milestone(section)


def _resolve_opencode_flags() -> list[str]:
    """Flags for `opencode run`, from PRAXIS_CODER_OPENCODE_FLAGS (default --auto).

    An unset variable yields ``--auto`` (stock opencode's non-interactive
    flag). An explicitly empty value yields no flags — some opencode builds
    treat ``--auto`` as an unknown argument and exit 1.
    """
    raw = os.environ.get(OPENCODE_FLAGS_ENV)
    if raw is None:
        return [DEFAULT_OPENCODE_FLAGS]
    return [flag for flag in raw.split() if flag]


def _build_coder_prompt(blueprint: Blueprint) -> str:
    first_phase = _extract_first_phase(blueprint.blueprint_md or "")
    return (
        "You are drafting a working prototype inside a fresh, empty directory.\n\n"
        f"Feasibility score of the parent blueprint: {blueprint.feasibility_score}\n\n"
        "First phase to implement (from the blueprint's Phased Build Plan):\n"
        f"{first_phase}\n\n"
        "Build a minimal, self-contained, working implementation of this phase in "
        "this directory. Keep the scope to exactly this phase - do not build ahead "
        "into later milestones. Write real code, add a short README explaining how "
        "to run it, and keep dependencies minimal."
    )


def _invoke_opencode(
    prompt: str,
    cwd: Path,
    timeout: float,
    model: str | None = None,
) -> subprocess.CompletedProcess:
    """Run OpenCode CLI in non-interactive mode against a scratch directory."""
    if os.name == "nt":
        cmd = ["cmd", "/c", "opencode", "run", *_resolve_opencode_flags()]
    else:
        cmd = ["opencode", "run", *_resolve_opencode_flags()]
    if model:
        cmd += ["--model", model]
    cmd.append(prompt)
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _persist_status(
    blueprint: Blueprint,
    candidate_status: str,
    prototype_path: str | None,
) -> None:
    session = get_session()
    try:
        stored = session.get(Blueprint, blueprint.id) if blueprint.id is not None else None
        if stored is None:
            stored = blueprint
            session.add(stored)
        if prototype_path is not None:
            stored.prototype_path = prototype_path
        if blueprint.candidate_id is not None:
            candidate = session.get(Candidate, blueprint.candidate_id)
            if candidate is not None:
                candidate.status = candidate_status
            else:
                logger.warning(
                    "coder: candidate %s not found; status not updated",
                    blueprint.candidate_id,
                )
        session.commit()
    finally:
        session.close()


def _stderr_summary(output) -> str:
    """First lines + tail of opencode's stderr for a failure log.

    The real error line is almost always at the very end of stderr, while the
    first lines show what opencode was doing; logging only a byte-limited tail
    used to cut the error off entirely for long, single-line JSON errors.
    """
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    lines = [line for line in (output or "").splitlines() if line.strip()]
    if not lines:
        return "(stderr empty)"
    limit = _STDERR_HEAD_LINES + _STDERR_TAIL_LINES
    if len(lines) <= limit:
        return "\n".join(lines)
    omitted = len(lines) - limit
    head = lines[:_STDERR_HEAD_LINES]
    tail = lines[-_STDERR_TAIL_LINES:]
    return "\n".join([*head, f"... ({omitted} lines omitted) ...", *tail])


def _classify_opencode_failure(exc_or_proc) -> str | None:
    """Return an exhaustion signal for a failed opencode run (exit/timeout), or None."""
    output = (
        f"{getattr(exc_or_proc, 'stdout', '') or ''}\n{getattr(exc_or_proc, 'stderr', '') or ''}"
    )
    return scan_exhaustion(output)


def draft_prototype(
    blueprint: Blueprint,
    scratch_root: Path | None = None,
    timeout: float | None = None,
) -> Path | None:
    """Draft a prototype from the blueprint's first phase via OpenCode CLI.

    When ``PRAXIS_CODER_MODELS`` lists one or more ``provider/model`` ids, the
    models are tried in order; a provider that reports rate-limit/quota/context
    exhaustion goes into a persisted cooldown and the next model is tried
    instantly (no backoff wait). Generic failures (network, other non-zero
    exits) still hit the global circuit breaker as before.
    """
    timeout = _resolve_timeout(timeout)
    root = _resolve_scratch_root(scratch_root)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    scratch = root / f"proto-{blueprint.candidate_id}-{timestamp}"
    scratch.mkdir(parents=True, exist_ok=True)

    prompt = _build_coder_prompt(blueprint)
    breaker = _get_breaker()
    if not breaker.allow():
        logger.warning(
            "coder: circuit open; skipping opencode for blueprint %s (max %d failures, "
            "cooldown %ss)",
            blueprint.id,
            breaker.max_failures,
            breaker.cooldown_s,
        )
        _persist_status(blueprint, "prototype_failed", None)
        return None

    pool = ProviderPool(JOB_CODER)
    models = _resolve_coder_models()
    if models:
        attempts = pool.healthy_models(models)[: _resolve_provider_retries()]
    else:
        attempts = [None]  # opencode's own default model, legacy single-attempt behavior

    for model in attempts:
        try:
            if model is None:
                proc = _invoke_opencode(prompt, scratch, timeout)
            else:
                proc = _invoke_opencode(prompt, scratch, timeout, model=model)
        except subprocess.TimeoutExpired as exc:
            signal = _classify_opencode_failure(exc)
            if signal is not None and model is not None:
                pool.mark_exhausted(provider_of(model), signal, detail=str(exc)[:300])
                logger.warning("coder: model %s exhausted (%s); switching provider", model, signal)
                continue
            breaker.record_failure()
            logger.warning(
                "coder: opencode timed out after %ss for blueprint %s; stderr:\n%s",
                timeout,
                blueprint.id,
                _stderr_summary(exc.stderr),
            )
            _persist_status(blueprint, "prototype_failed", None)
            return None

        if proc.returncode != 0:
            signal = _classify_opencode_failure(proc)
            output = f"{proc.stdout or ''}\n{proc.stderr or ''}"
            if signal is not None and model is not None:
                pool.mark_exhausted(provider_of(model), signal, detail=output[-300:])
                logger.warning("coder: model %s exhausted (%s); switching provider", model, signal)
                continue
            breaker.record_failure()
            logger.warning(
                "coder: opencode failed for blueprint %s (rc=%s); stderr:\n%s",
                blueprint.id,
                proc.returncode,
                _stderr_summary(proc.stderr),
            )
            _persist_status(blueprint, "prototype_failed", None)
            return None

        breaker.record_success()
        if model is not None:
            pool.mark_healthy(provider_of(model))
        _persist_status(blueprint, "prototyped", str(scratch))
        return scratch

    # Every configured provider was exhausted within the retry bound.
    _persist_status(blueprint, "prototype_failed", None)
    logger.warning("coder: all providers exhausted for blueprint %s", blueprint.id)
    return None


coder = draft_prototype
