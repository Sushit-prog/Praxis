"""Shared pytest fixtures, including a fake/mock LLM client."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from praxis.config import HardwareProfile  # noqa: E402


def fake_completion(**kwargs):
    """Stand-in for litellm.completion that returns a canned response."""
    return {"choices": [{"message": {"content": "fake model output"}}]}


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
    """A session bound to the test engine, injected into scout's get_session."""
    import importlib

    from sqlalchemy.orm import Session

    scout_module = importlib.import_module("praxis.agents.scout")
    session = Session(bind=db_engine)
    monkeypatch.setattr(scout_module, "get_session", lambda: session)
    yield session
    session.close()
