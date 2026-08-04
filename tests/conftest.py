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
    """A persistent test session; agents' get_session() gets fresh sessions on the same engine."""
    import importlib

    from sqlalchemy.orm import Session

    session = Session(bind=db_engine)

    def fresh_session():
        return Session(bind=db_engine, expire_on_commit=False)

    for module_name in (
        "praxis.agents.scout",
        "praxis.agents.analyst",
        "praxis.agents.architect",
        "praxis.agents.coder",
    ):
        module = importlib.import_module(module_name)
        monkeypatch.setattr(module, "get_session", fresh_session)
    yield session
    session.close()
