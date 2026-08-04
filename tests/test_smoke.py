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


def test_agent_stubs_raise():
    from praxis.agents import analyze

    with pytest.raises(NotImplementedError):
        analyze(candidate=None)


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


def test_cli_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0


def test_cli_run_subcommand_help():
    with pytest.raises(SystemExit) as excinfo:
        main(["run", "--help"])
    assert excinfo.value.code == 0
