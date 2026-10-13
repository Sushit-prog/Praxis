"""Shared pytest fixtures, including a fake/mock LLM client."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Key-independence: LITELLM_MODE != DEV disables litellm's import-time
# load_dotenv, so preloading it here never reads the project .env (the
# _hide_project_dotenv fixture hides that file from praxis's _load_env too,
# and _hermetic_model_env clears provider keys per test). The preload itself
# keeps litellm's import cost out of the first test that needs it.
os.environ["LITELLM_MODE"] = "PRODUCTION"
import litellm  # noqa: E402, F401

from praxis.config import HardwareProfile  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _deep_isolation(tmp_path_factory):
    """Keep the suite off the developer's real ./praxis.db.

    Any test that does not set its own PRAXIS_DB_URL gets a throwaway
    database instead of the default ledger, and the session fails loudly if
    ./praxis.db (or its WAL sidecars) still changes — some code path opened
    the default ledger and must be fixed to point at a temp DB.
    """
    default_db = Path.cwd() / "praxis.db"

    def _snapshot():
        paths = [default_db, Path(f"{default_db}-wal"), Path(f"{default_db}-shm")]
        return {
            p: (p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else None
            for p in paths
        }

    before = _snapshot()
    suite_db = tmp_path_factory.mktemp("ledger") / "suite.db"
    os.environ["PRAXIS_DB_URL"] = f"sqlite:///{suite_db.as_posix()}"
    yield
    if os.environ.get("PRAXIS_DB_URL") == f"sqlite:///{suite_db.as_posix()}":
        del os.environ["PRAXIS_DB_URL"]
    after = _snapshot()
    if before != after:
        pytest.fail(
            "the default ./praxis.db was opened/modified during the suite; "
            "tests must set PRAXIS_DB_URL (or PRAXIS_DB_PATH) to a temp database"
        )


@pytest.fixture(scope="session", autouse=True)
def _hide_project_dotenv():
    """Keep the repo's own .env out of the test session.

    ``_load_env`` discovers the file via ``dotenv.find_dotenv``; the guard
    makes the project's .env resolve to "not found" so no test can pull real
    provider keys into ``os.environ``. Any other .env (the temp fixtures the
    loader tests write) still loads normally. The real function is restored
    at session end.
    """
    import dotenv

    project_env = Path(__file__).resolve().parents[1] / ".env"
    real_find_dotenv = dotenv.find_dotenv

    def _guarded_find_dotenv(*args, **kwargs):
        path = real_find_dotenv(*args, **kwargs)
        if path and Path(path).resolve() == project_env:
            return ""
        return path

    dotenv.find_dotenv = _guarded_find_dotenv
    yield
    dotenv.find_dotenv = real_find_dotenv


def fake_completion(**kwargs):
    """Stand-in for litellm.completion that returns a canned response."""
    return {"choices": [{"message": {"content": "fake model output"}}]}


@pytest.fixture(autouse=True)
def _hermetic_model_env(monkeypatch):
    """Keep model identity and provider keys out of the tests.

    Tests inject fakes and assert explicit model strings and key values;
    they must not depend on local env files or ambient shell exports. The
    session hides the project .env from ``_load_env`` (litellm's own import
    load is disabled via LITELLM_MODE), but a developer's exported
    PRAXIS_MODEL/PRAXIS_FALLBACK_MODELS or provider keys must not leak into
    which model id the usage ledger records either.
    """
    monkeypatch.delenv("PRAXIS_MODEL", raising=False)
    monkeypatch.delenv("PRAXIS_FALLBACK_MODELS", raising=False)
    # Hardware-profile and design-chain overrides that a project .env or the
    # shell could carry.
    for var in (
        "PRAXIS_CONFIG",
        "PRAXIS_DETECT_HW",
        "PRAXIS_OS",
        "PRAXIS_CPU",
        "PRAXIS_CPU_CORES",
        "PRAXIS_RAM_GB",
        "PRAXIS_USABLE_RAM_GB",
        "PRAXIS_CPU_ONLY",
        "PRAXIS_GPU",
        "PRAXIS_GPU_NAME",
        "PRAXIS_GPU_NOTE",
        "PRAXIS_RAM_NOTE",
        "PRAXIS_STORAGE_FREE_GB",
        "PRAXIS_MONTHLY_BUDGET_USD",
        "PRAXIS_DESIGN_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)
    # Provider keys are never ambient: every test starts keyless and sets
    # its own fake values when it needs them (GROQ / NVIDIA NIM / OpenRouter).
    for key in ("GROQ_API_KEY", "NVIDIA_NIM_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    # Host detection stays off by default so the suite is host-independent;
    # detection tests opt in via PRAXIS_DETECT_HW / detected_host= with a
    # mocked probe (see tests/test_facts.py and tests/test_hardware.py).
    monkeypatch.setenv("PRAXIS_DETECT_HW", "0")
    # The design engine's TPM pacing window is process-global; a fresh window
    # per test keeps paced sleeps (bounded, but up to ~90s) out of the suite.
    # The sleeps themselves are neutralized: rate-limit waiting is covered by
    # dedicated tests that patch praxis.design._sleep and assert the waits.
    import praxis.design as design_module

    design_module._token_window = design_module._TokenWindow()
    monkeypatch.setattr(design_module, "_sleep", lambda _s: None)


@pytest.fixture
def completion_func():
    """Return the fake completion function for injection into call_llm."""
    return fake_completion


@pytest.fixture
def llm_client(completion_func):
    """An LLMClient bound to the fake completion function."""
    from praxis.llm import LLMClient

    return LLMClient(completion=completion_func)


@pytest.fixture
def hardware_profile() -> HardwareProfile:
    return HardwareProfile()


@pytest.fixture
def db_engine(tmp_path):
    """An in-memory SQLite engine with tables created."""
    from sqlalchemy import create_engine

    from praxis.db import Base

    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(db_engine, monkeypatch):
    """A persistent test session; agents' get_session() gets fresh sessions on the same engine."""
    import importlib

    from sqlalchemy.orm import Session

    session = Session(bind=db_engine)

    def fresh_session():
        return Session(bind=db_engine, expire_on_commit=False)

    # Import every module BEFORE patching: a module first imported while
    # praxis.db.get_session is already patched would capture the test session
    # factory permanently via its ``from praxis.db import get_session``
    # binding (monkeypatch could never restore it).
    modules = [
        importlib.import_module(module_name)
        for module_name in (
            "praxis.agents.scout",
            "praxis.agents.analyst",
            "praxis.agents.architect",
            "praxis.agents.coder",
            "praxis.pipeline",
            "praxis.llm",
            "praxis.review",
            "praxis.providers",
            "praxis.design",
            "praxis.discover",
            "praxis.design_io",
            "praxis.export",
        )
    ]
    for module in modules:
        monkeypatch.setattr(module, "get_session", fresh_session)
    yield session
    session.close()
