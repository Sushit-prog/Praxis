"""Smoke tests: package imports, config defaults, LLM wrapper, CLI --help."""

from __future__ import annotations

import pytest

import praxis
from praxis.cli import main


def test_package_imports():
    assert praxis.__version__ == "0.1.0"
    from praxis import agents, pipeline  # noqa: F401

    assert agents is not None


def test_config_defaults(hardware_profile):
    assert hardware_profile.cpu_only is True
    assert hardware_profile.ram_gb == 8
    assert hardware_profile.gpu is False
    assert hardware_profile.monthly_budget_usd == 15.0


def test_config_loads_env_override(monkeypatch):
    from praxis.config import load_config

    monkeypatch.setenv("PRAXIS_RAM_GB", "16")
    monkeypatch.setenv("PRAXIS_GPU", "true")
    profile = load_config(path="nonexistent.yaml")
    assert profile.ram_gb == 16
    assert profile.gpu is True


def test_call_llm_with_fake_completion(completion_func):
    from praxis.llm import call_llm

    result = call_llm("hello", completion=completion_func)
    assert result == "fake model output"


def test_db_models_roundtrip(db_engine):
    from sqlalchemy.orm import Session

    from praxis.db import Blueprint, Candidate

    with Session(db_engine) as session:
        cand = Candidate(source="arxiv", url="https://example.com", title="t", raw_text="text")
        session.add(cand)
        session.flush()
        session.add(
            Blueprint(
                candidate_id=cand.id,
                feasibility_score=0.9,
                blueprint_md="# Plan",
            )
        )
        session.commit()
        cand = session.get(Candidate, cand.id)
        assert cand is not None
        assert cand.blueprints[0].feasibility_score == 0.9


def test_run_with_retry_recovers(monkeypatch):
    from praxis.pipeline import run_with_retry

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("transient")
        return "ok"

    monkeypatch.setattr("praxis.pipeline.time.sleep", lambda s: None)
    assert run_with_retry(flaky) == "ok"
    assert calls["n"] == 3


def test_run_with_retry_does_not_retry_not_implemented(monkeypatch):
    from praxis.pipeline import run_with_retry

    calls = {"n": 0}

    def stub():
        calls["n"] += 1
        raise NotImplementedError

    monkeypatch.setattr("praxis.pipeline.time.sleep", lambda s: None)
    with pytest.raises(NotImplementedError):
        run_with_retry(stub)
    assert calls["n"] == 1


def test_run_with_retry_does_not_retry_db_error(monkeypatch):
    """Bug regression: OperationalError is a DB error — fail fast, no retries."""
    from sqlalchemy.exc import OperationalError

    from praxis.pipeline import run_with_retry

    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise OperationalError("SELECT 1", {}, Exception("no such table: candidates"))

    monkeypatch.setattr("praxis.pipeline.time.sleep", lambda s: None)
    with pytest.raises(OperationalError):
        run_with_retry(boom, retries=3)
    assert calls["n"] == 1


def test_run_with_retry_does_not_retry_value_error(monkeypatch):
    from praxis.pipeline import run_with_retry

    calls = {"n": 0}

    def bug():
        calls["n"] += 1
        raise ValueError("bad input")

    monkeypatch.setattr("praxis.pipeline.time.sleep", lambda s: None)
    with pytest.raises(ValueError):
        run_with_retry(bug, retries=3)
    assert calls["n"] == 1


def test_run_with_retry_retries_named_rate_limit_error(monkeypatch):
    """Errors named like litellm rate-limit families are retried with backoff."""
    from praxis.pipeline import run_with_retry

    RateLimitError = type("RateLimitError", (Exception,), {})
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RateLimitError("429 too many requests")
        return "ok"

    monkeypatch.setattr("praxis.pipeline.time.sleep", lambda s: None)
    assert run_with_retry(flaky, retries=3) == "ok"
    assert calls["n"] == 3


def test_run_with_retry_retries_connection_error_but_not_file_error(monkeypatch):
    """OSError-family network errors are transient; config errors are fatal."""
    from praxis.pipeline import run_with_retry

    calls = {"n": 0}

    def conn_fail():
        calls["n"] += 1
        raise ConnectionError("reset by peer")

    monkeypatch.setattr("praxis.pipeline.time.sleep", lambda s: None)
    # Exhausted retries surface as RuntimeError with the original error attached.
    with pytest.raises(RuntimeError) as excinfo:
        run_with_retry(conn_fail, retries=2)
    assert isinstance(excinfo.value.__cause__, ConnectionError)
    assert calls["n"] == 2  # retried once before giving up

    calls["n"] = 0

    def missing_file():
        calls["n"] += 1
        raise FileNotFoundError("hardware_profile.yaml")

    with pytest.raises(FileNotFoundError):
        run_with_retry(missing_file, retries=2)
    assert calls["n"] == 1  # fatal: no retry


def test_load_env_populates_keys_and_respects_real_env(tmp_path, monkeypatch):
    """A temp .env supplies provider keys; real environment variables still win."""
    import os

    from praxis.cli import _load_env

    dotenv_keys = (
        "GROQ_API_KEY",
        "CEREBRAS_API_KEY",
        "PRAXIS_GROQ_API_KEY",
        "PRAXIS_MODEL",
    )
    for key in dotenv_keys:
        monkeypatch.delenv(key, raising=False)  # teardown removes values loaded below
    (tmp_path / ".env").write_text(
        "GROQ_API_KEY=gsk-from-file\n"
        "OPENROUTER_API_KEY=sk-or-from-file\n"
        "CEREBRAS_API_KEY=cs-from-file\n"
        "PRAXIS_GROQ_API_KEY=praxis-override\n"
        "PRAXIS_MODEL=groq/openai/gpt-oss-20b\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-from-shell")  # real env wins

    loaded = _load_env()

    assert loaded is not None and loaded.endswith(".env")
    assert os.environ["GROQ_API_KEY"] == "gsk-from-file"
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-from-shell"
    assert os.environ["CEREBRAS_API_KEY"] == "cs-from-file"
    assert os.environ["PRAXIS_GROQ_API_KEY"] == "praxis-override"
    assert os.environ["PRAXIS_MODEL"] == "groq/openai/gpt-oss-20b"


def test_load_env_without_file_is_noop(tmp_path, monkeypatch):
    from praxis.cli import _load_env

    monkeypatch.chdir(tmp_path)
    assert _load_env() is None


def test_cli_reads_config_after_dotenv(tmp_path, monkeypatch, capsys):
    """PRAXIS_DB_URL set only in .env is honored by the CLI (env loads before config)."""
    import os

    (tmp_path / ".env").write_text(
        f"PRAXIS_DB_URL=sqlite:///{(tmp_path / 'from-env.db').as_posix()}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PRAXIS_DB_URL", raising=False)  # teardown removes loaded value

    rc = main(["status"])

    assert rc == 0
    assert "No candidates" in capsys.readouterr().out
    assert (tmp_path / "from-env.db").exists()
    assert os.environ.get("PRAXIS_DB_URL")  # loaded for the command that just ran


def test_provider_keys_pass_through_to_litellm(monkeypatch):
    """Plain and PRAXIS_-prefixed keys reach the litellm kwargs.

    Plain keys flow through os.environ (litellm auto-reads them there); the
    PRAXIS_<PROVIDER>_API_KEY overrides are injected into the call kwargs.
    """
    from praxis.llm import _inject_provider_key

    monkeypatch.setenv("PRAXIS_GROQ_API_KEY", "praxis-groq")
    kwargs: dict = {}
    _inject_provider_key(kwargs, "groq/openai/gpt-oss-20b")
    assert kwargs["api_key"] == "praxis-groq"

    monkeypatch.setenv("PRAXIS_OPENROUTER_API_KEY", "praxis-or")
    kwargs = {}
    _inject_provider_key(kwargs, "openrouter/openai/gpt-oss-20b")
    assert kwargs["api_key"] == "praxis-or"

    monkeypatch.setenv("PRAXIS_CEREBRAS_API_KEY", "praxis-cs")
    kwargs = {}
    _inject_provider_key(kwargs, "cerebras/llama-3.3-70b")
    assert kwargs["api_key"] == "praxis-cs"

    # No override set -> no injection; litellm reads the plain key from os.environ.
    monkeypatch.delenv("PRAXIS_GROQ_API_KEY", raising=False)
    kwargs = {}
    _inject_provider_key(kwargs, "groq/openai/gpt-oss-20b")
    assert "api_key" not in kwargs


def test_cli_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0


def test_cli_run_subcommand_help():
    with pytest.raises(SystemExit) as excinfo:
        main(["run", "--help"])
    assert excinfo.value.code == 0
