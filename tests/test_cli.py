"""CLI tests for `praxis run`, `praxis status`, and `praxis show`."""

from __future__ import annotations

import pytest

from praxis.cli import main


@pytest.fixture
def seeded_db(tmp_path, monkeypatch):
    """A temp SQLite DB with one prototyped and one rejected candidate."""
    from sqlalchemy.orm import Session

    from praxis.db import Base, Blueprint, Candidate, get_engine

    db_path = tmp_path / "cli.db"
    monkeypatch.setenv("PRAXIS_DB_URL", f"sqlite:///{db_path}")
    engine = get_engine()
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        alpha = Candidate(
            source="arxiv", url="https://a", title="Alpha", raw_text="x", status="prototyped"
        )
        session.add(alpha)
        session.flush()
        session.add(
            Blueprint(
                candidate_id=alpha.id,
                feasibility_score=7.0,
                blueprint_md="# Blueprint Alpha\n\nPhase 1: build it.",
            )
        )
        session.add(
            Candidate(
                source="github", url="https://b", title="Beta", raw_text="y", status="rejected"
            )
        )
        session.commit()
    return db_path


def test_cli_status_counts(seeded_db, capsys):
    rc = main(["status"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Candidate counts by status:" in out
    assert "prototyped: 1" in out
    assert "rejected: 1" in out


def test_cli_show_prints_blueprint(seeded_db, capsys):
    rc = main(["show", "1"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Alpha" in out
    assert "https://a" in out
    assert "# Blueprint Alpha" in out


def test_cli_show_missing_candidate(seeded_db, capsys):
    rc = main(["show", "999"])

    assert rc == 1
    assert "no candidate" in capsys.readouterr().err


def test_cli_show_candidate_without_blueprint(seeded_db, capsys):
    rc = main(["show", "2"])

    assert rc == 1
    assert "no blueprint" in capsys.readouterr().err


def test_cli_status_empty_db(tmp_path, monkeypatch, capsys):
    from sqlalchemy import create_engine

    from praxis.db import Base

    db_path = tmp_path / "empty.db"
    monkeypatch.setenv("PRAXIS_DB_URL", f"sqlite:///{db_path}")
    Base.metadata.create_all(create_engine(f"sqlite:///{db_path}"))

    rc = main(["status"])

    assert rc == 0
    assert "No candidates" in capsys.readouterr().out


def test_cli_run_prints_summary(monkeypatch, capsys):
    from praxis.pipeline import CandidateOutcome, PipelineResult

    fake_result = PipelineResult(
        source="arxiv",
        topic="attention",
        discovered=2,
        analyzed=1,
        rejected=1,
        blueprinted=1,
        prototyped=1,
        candidates=[
            CandidateOutcome(title="Alpha", url="https://a", status="prototyped"),
            CandidateOutcome(title="Beta", url="https://b", status="rejected"),
        ],
    )
    monkeypatch.setattr("praxis.pipeline.run", lambda **kwargs: fake_result)

    rc = main(["run", "--source", "arxiv", "--topic", "attention"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Summary for topic='attention' source=arxiv" in out
    assert "prototyped: 1" in out
    assert "rejected: 1" in out
    assert "Alpha [prototyped]" in out
    assert "Beta [rejected]" in out
