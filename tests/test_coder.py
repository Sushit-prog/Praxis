"""Coder agent tests: mocked subprocess.run; success, failure, timeout, prompt scoping."""

from __future__ import annotations

import importlib
import subprocess

import pytest

from praxis.agents.coder import _CircuitBreaker, _extract_first_phase, draft_prototype
from praxis.db import Blueprint, Candidate

coder_module = importlib.import_module("praxis.agents.coder")


@pytest.fixture(autouse=True)
def _isolated_breaker(monkeypatch):
    """Reset the module-level circuit breaker before every test so failure state
    never leaks between tests; default envs keep it permissive (never trips)."""
    monkeypatch.setattr(coder_module, "_breaker", None)
    monkeypatch.setenv("PRAXIS_CODER_MAX_FAILURES", "1000")
    monkeypatch.setenv("PRAXIS_CODER_COOLDOWN_S", "0")
    monkeypatch.setenv("PRAXIS_CODER_OPENCODE_FLAGS", "--auto")


PHASED_PLAN_MD = (
    "# Build a Fine-Tuner \u2014 Blueprint\n\n"
    "## Problem Statement\nMake a small fine-tuner.\n\n"
    "## Proposed Architecture\nCPU-only.\n\n"
    "## Phased Build Plan\n"
    "Start with a data loader that reads JSONL.\n\n"
    "1. Data pipeline: parse, tokenize, and batch samples in plain Python.\n"
    "   - Read JSONL files.\n"
    "   - Tokenize with a small regex tokenizer.\n"
    "2. Training loop: minimal gradient descent on CPU.\n"
    "3. Evaluation harness.\n\n"
    "## Deferred to Later Versions\nGPU support.\n"
)


def make_candidate(session, url="https://example.com/repo", title="A Repo", raw_text="body"):
    cand = Candidate(source="github", url=url, title=title, raw_text=raw_text, status="blueprinted")
    session.add(cand)
    session.commit()
    session.refresh(cand)
    return cand


def make_blueprint(session, candidate, md=PHASED_PLAN_MD):
    bp = Blueprint(candidate_id=candidate.id, feasibility_score=8.0, blueprint_md=md)
    session.add(bp)
    session.commit()
    session.refresh(bp)
    return bp


def fake_completed(returncode=0, stdout="ok", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def mock_subprocess_run(monkeypatch, result):
    """Patch coder.subprocess.run; records call kwargs, returns/raises result."""
    calls = {}

    def fake(cmd, cwd=None, capture_output=None, text=None, timeout=None):
        calls["cmd"] = cmd
        calls["cwd"] = cwd
        calls["capture_output"] = capture_output
        calls["text"] = text
        calls["timeout"] = timeout
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(coder_module.subprocess, "run", fake)
    return calls


def test_draft_prototype_success(db_session, monkeypatch, tmp_path):
    cand = make_candidate(db_session)
    bp = make_blueprint(db_session, cand)
    calls = mock_subprocess_run(monkeypatch, fake_completed())

    path = draft_prototype(bp, scratch_root=tmp_path)

    assert path is not None
    assert path.exists()
    assert path.parent == tmp_path
    assert path.name.startswith(f"proto-{cand.id}-")
    assert calls["cwd"] == str(path)
    assert calls["timeout"] == 600
    assert calls["capture_output"] is True
    assert calls["text"] is True
    assert "opencode" in calls["cmd"]
    assert "run" in calls["cmd"]
    assert "--auto" in calls["cmd"]

    db_session.expire_all()
    assert db_session.get(Candidate, cand.id).status == "prototyped"
    assert db_session.get(Blueprint, bp.id).prototype_path == str(path)


def test_draft_prototype_no_auto_flags(db_session, monkeypatch, tmp_path):
    """Forks whose `opencode run` rejects --auto pass an empty flag list."""
    monkeypatch.setenv("PRAXIS_CODER_OPENCODE_FLAGS", "")
    cand = make_candidate(db_session)
    bp = make_blueprint(db_session, cand)
    calls = mock_subprocess_run(monkeypatch, fake_completed())

    path = draft_prototype(bp, scratch_root=tmp_path)

    assert path is not None
    assert "--auto" not in calls["cmd"]
    assert "opencode" in calls["cmd"]
    assert "run" in calls["cmd"]


def test_draft_prototype_custom_flags(db_session, monkeypatch, tmp_path):
    monkeypatch.setenv("PRAXIS_CODER_OPENCODE_FLAGS", "--print-logs")
    cand = make_candidate(db_session)
    bp = make_blueprint(db_session, cand)
    calls = mock_subprocess_run(monkeypatch, fake_completed())

    path = draft_prototype(bp, scratch_root=tmp_path)

    assert path is not None
    assert "--print-logs" in calls["cmd"]
    assert "--auto" not in calls["cmd"]


def test_draft_prototype_nonzero_exit(db_session, monkeypatch, tmp_path, caplog):
    cand = make_candidate(db_session)
    bp = make_blueprint(db_session, cand)
    mock_subprocess_run(monkeypatch, fake_completed(returncode=1, stderr="boom error"))

    with caplog.at_level("WARNING"):
        path = draft_prototype(bp, scratch_root=tmp_path)

    assert path is None
    assert "boom error" in caplog.text
    db_session.expire_all()
    assert db_session.get(Candidate, cand.id).status == "prototype_failed"
    assert db_session.get(Blueprint, bp.id).prototype_path is None


def test_draft_prototype_timeout(db_session, monkeypatch, tmp_path, caplog):
    cand = make_candidate(db_session)
    bp = make_blueprint(db_session, cand)
    mock_subprocess_run(
        monkeypatch,
        subprocess.TimeoutExpired(cmd=["opencode", "run"], timeout=600),
    )

    with caplog.at_level("WARNING"):
        path = draft_prototype(bp, scratch_root=tmp_path)

    assert path is None
    assert "timed out" in caplog.text
    db_session.expire_all()
    assert db_session.get(Candidate, cand.id).status == "prototype_failed"


def test_draft_prototype_prompt_scoped_to_first_phase(db_session, monkeypatch, tmp_path):
    cand = make_candidate(db_session)
    bp = make_blueprint(db_session, cand)
    captured = {}

    def fake_invoke(prompt, cwd, timeout, model=None):
        captured["prompt"] = prompt
        captured["cwd"] = cwd
        return fake_completed()

    monkeypatch.setattr(coder_module, "_invoke_opencode", fake_invoke)

    path = draft_prototype(bp, scratch_root=tmp_path)

    assert path is not None
    prompt = captured["prompt"]
    assert "Feasibility score of the parent blueprint: 8.0" in prompt
    assert "Data pipeline" in prompt
    assert "Read JSONL files" in prompt
    assert "Training loop" not in prompt
    assert "Evaluation harness" not in prompt


def test_draft_prototype_timeout_from_env(db_session, monkeypatch, tmp_path):
    monkeypatch.setenv("PRAXIS_CODER_TIMEOUT_S", "3")
    cand = make_candidate(db_session)
    bp = make_blueprint(db_session, cand)
    calls = mock_subprocess_run(monkeypatch, fake_completed())

    draft_prototype(bp, scratch_root=tmp_path)

    assert calls["timeout"] == 3


def test_extract_first_phase_numbered_list():
    md = (
        "## Phased Build Plan\n"
        "Intro prose.\n\n"
        "1. First milestone.\n"
        "   - sub bullet.\n"
        "2. Second milestone.\n"
    )
    out = _extract_first_phase(md)
    assert "First milestone" in out
    assert "sub bullet" in out
    assert "Second milestone" not in out
    assert "Intro prose" not in out


def test_extract_first_phase_no_heading_returns_first_paragraph():
    md = "First paragraph line.\n\nSecond paragraph.\n"
    assert _extract_first_phase(md) == "First paragraph line."


def test_extract_first_phase_no_items_returns_section_prose():
    md = "## Phased Build Plan\nProse only here.\n"
    assert _extract_first_phase(md) == "Prose only here."


def test_extract_first_phase_blank():
    assert _extract_first_phase("") == ""
    assert _extract_first_phase("   \n  ") == ""


def test_circuit_breaker_opens_after_max_failures():
    cb = _CircuitBreaker(max_failures=2, cooldown_s=1000)

    assert cb.allow() is True
    cb.record_failure()
    assert cb.allow() is True
    cb.record_failure()
    assert cb.allow() is False  # open


def test_circuit_breaker_reopens_after_cooldown(monkeypatch):
    cb = _CircuitBreaker(max_failures=1, cooldown_s=10)
    now = {"t": 0.0}
    monkeypatch.setattr(coder_module.time, "monotonic", lambda: now["t"])

    cb.record_failure()
    assert cb.allow() is False  # open
    now["t"] = 11.0
    assert cb.allow() is True  # half-open trial attempt


def test_circuit_breaker_success_resets():
    cb = _CircuitBreaker(max_failures=2, cooldown_s=10)

    cb.record_failure()
    cb.record_success()
    assert cb.allow() is True
    cb.record_failure()
    cb.record_failure()
    assert cb.allow() is False


def test_circuit_breaker_failed_trial_reopens(monkeypatch):
    """A half-open trial that fails re-opens the circuit with a fresh cooldown."""
    cb = _CircuitBreaker(max_failures=2, cooldown_s=10)
    now = {"t": 0.0}
    monkeypatch.setattr(coder_module.time, "monotonic", lambda: now["t"])

    cb.record_failure()
    cb.record_failure()
    assert cb.allow() is False  # open
    now["t"] = 11.0
    assert cb.allow() is True  # half-open trial permitted
    cb.record_failure()  # trial fails -> must re-open, not quietly close
    assert cb.allow() is False  # still open (fresh cooldown)
    now["t"] = 22.0
    assert cb.allow() is True  # next trial after the fresh cooldown


def test_draft_prototype_circuit_open_skips_subprocess(db_session, monkeypatch, tmp_path, caplog):
    cand = make_candidate(db_session)
    bp = make_blueprint(db_session, cand)
    calls = {"n": 0}

    def fake_run(cmd, cwd=None, capture_output=None, text=None, timeout=None):
        calls["n"] += 1
        return fake_completed()

    monkeypatch.setattr(coder_module.subprocess, "run", fake_run)

    class _BlockingBreaker:
        max_failures = 2
        cooldown_s = 300.0

        def allow(self):
            return False

    monkeypatch.setattr(coder_module, "_get_breaker", lambda: _BlockingBreaker())

    with caplog.at_level("WARNING"):
        path = draft_prototype(bp, scratch_root=tmp_path)

    assert path is None
    assert calls["n"] == 0
    assert "circuit open" in caplog.text
    db_session.expire_all()
    assert db_session.get(Candidate, cand.id).status == "prototype_failed"


def test_draft_prototype_failures_trip_breaker(db_session, monkeypatch, tmp_path):
    """Consecutive subprocess failures open the env-configured circuit."""
    monkeypatch.setenv("PRAXIS_CODER_MAX_FAILURES", "1")
    monkeypatch.setenv("PRAXIS_CODER_COOLDOWN_S", "1000")
    cand = make_candidate(db_session)
    bp = make_blueprint(db_session, cand)
    calls = {"n": 0}

    def fake_run(cmd, cwd=None, capture_output=None, text=None, timeout=None):
        calls["n"] += 1
        return fake_completed(returncode=1)

    monkeypatch.setattr(coder_module.subprocess, "run", fake_run)

    assert draft_prototype(bp, scratch_root=tmp_path) is None  # fails, circuit opens
    assert draft_prototype(bp, scratch_root=tmp_path) is None  # skipped, no subprocess

    assert calls["n"] == 1


def test_coder_alias():
    from praxis.agents import coder

    assert coder is draft_prototype


def _run_sequence(monkeypatch, sequence):
    """Patch subprocess.run to return each CompletedProcess in order."""
    calls = []

    def fake(cmd, cwd=None, capture_output=None, text=None, timeout=None):
        calls.append(cmd)
        return sequence[min(len(calls) - 1, len(sequence) - 1)]

    monkeypatch.setattr(coder_module.subprocess, "run", fake)
    return calls


def test_draft_prototype_exhaustion_rotates_to_next_model(db_session, monkeypatch, tmp_path):
    """A 429 on the first coder model instantly switches to the next provider."""
    monkeypatch.setenv(
        "PRAXIS_CODER_MODELS",
        "groq/llama-3.1-8b-instant,openrouter/openai/gpt-4o-mini",
    )
    cand = make_candidate(db_session)
    bp = make_blueprint(db_session, cand)
    calls = _run_sequence(
        monkeypatch,
        [
            fake_completed(returncode=1, stderr="HTTP 429 rate limit exceeded"),
            fake_completed(stdout="built a prototype"),
        ],
    )

    path = draft_prototype(bp, scratch_root=tmp_path)

    assert path is not None
    assert len(calls) == 2
    first, second = calls
    assert "--model" in first
    assert "groq/llama-3.1-8b-instant" in first
    assert "--model" in second
    assert "openrouter/openai/gpt-4o-mini" in second
    db_session.expire_all()
    assert db_session.get(Candidate, cand.id).status == "prototyped"


def test_draft_prototype_all_providers_exhausted_fails(db_session, monkeypatch, tmp_path, caplog):
    """When every configured coder model is exhausted, skip breaker tripping per model."""
    monkeypatch.setenv(
        "PRAXIS_CODER_MODELS",
        "groq/llama-3.1-8b-instant,openrouter/openai/gpt-4o-mini",
    )
    cand = make_candidate(db_session)
    bp = make_blueprint(db_session, cand)
    calls = _run_sequence(
        monkeypatch,
        [
            fake_completed(returncode=1, stderr="rate limit exceeded"),
            fake_completed(returncode=1, stderr="insufficient_quota"),
        ],
    )

    with caplog.at_level("WARNING"):
        path = draft_prototype(bp, scratch_root=tmp_path)

    assert path is None
    assert len(calls) == 2
    assert "all providers exhausted" in caplog.text
    db_session.expire_all()
    assert db_session.get(Candidate, cand.id).status == "prototype_failed"


def test_draft_prototype_skips_cooling_provider(db_session, monkeypatch, tmp_path):
    """A provider in cooldown from an earlier model failure is skipped instantly."""
    from datetime import UTC, datetime, timedelta

    monkeypatch.setenv(
        "PRAXIS_CODER_MODELS",
        "groq/llama-3.1-8b-instant,openrouter/openai/gpt-4o-mini",
    )
    from praxis.db import set_provider_health
    from praxis.providers import JOB_CODER, ExhaustionSignal

    set_provider_health(
        JOB_CODER,
        "groq",
        state="cooling_down",
        cooldown_until=datetime.now(UTC).replace(tzinfo=None) + timedelta(seconds=60),
        last_signal=ExhaustionSignal.RATE_LIMIT,
        session=db_session,
    )
    db_session.commit()

    cand = make_candidate(db_session)
    bp = make_blueprint(db_session, cand)
    calls = _run_sequence(monkeypatch, [fake_completed(stdout="ok")])

    path = draft_prototype(bp, scratch_root=tmp_path)

    assert path is not None
    assert len(calls) == 1
    assert "groq/llama-3.1-8b-instant" not in calls[0]
    assert "openrouter/openai/gpt-4o-mini" in calls[0]
