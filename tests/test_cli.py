"""CLI tests for `praxis run`, `praxis status`, `praxis show`, and `praxis eval`."""

from __future__ import annotations

import json

import pytest

from praxis.cli import main


@pytest.fixture
def seeded_db(tmp_path, monkeypatch):
    """A temp SQLite DB with one prototyped, one rejected, usage, and memory rows."""
    from sqlalchemy.orm import Session

    from praxis.db import Base, Blueprint, BuildMemory, Candidate, LLMUsage, get_engine

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
        session.add(
            BuildMemory(
                candidate_id=alpha.id,
                technique="LoRA fine-tuning on CPU",
                decision="approved",
                outcome="prototyped",
            )
        )
        session.add_all(
            [
                LLMUsage(
                    model="groq/llama-3.1-8b-instant",
                    stage="analyst",
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150,
                    cost_usd=0.001,
                    latency_ms=200,
                ),
                LLMUsage(
                    model="groq/llama-3.1-8b-instant",
                    stage="architect",
                    prompt_tokens=300,
                    completion_tokens=100,
                    total_tokens=400,
                    cost_usd=0.002,
                    latency_ms=400,
                ),
                LLMUsage(model="groq/llama-3.1-8b-instant", stage="analyst", cached=True),
            ]
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


def test_cli_memory_lists_entries(seeded_db, capsys):
    rc = main(["memory"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Recent build memory:" in out
    assert "approved (prototyped): LoRA fine-tuning on CPU" in out


def test_cli_memory_empty(tmp_path, monkeypatch, capsys):
    from sqlalchemy import create_engine

    from praxis.db import Base

    db_path = tmp_path / "empty.db"
    monkeypatch.setenv("PRAXIS_DB_URL", f"sqlite:///{db_path}")
    Base.metadata.create_all(create_engine(f"sqlite:///{db_path}"))

    rc = main(["memory"])

    assert rc == 0
    assert "No build memory recorded yet." in capsys.readouterr().out


def test_cli_usage_prints_report(seeded_db, capsys):
    rc = main(["usage"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "LLM usage" in out
    assert "all time: 2 calls (1 from cache)" in out
    assert "550 tokens" in out
    assert "$0.0030" in out
    assert "by stage:" in out
    assert "analyst: 1 calls" in out
    assert "architect: 1 calls" in out
    assert "by model:" in out
    assert "groq/llama-3.1-8b-instant: 2 calls" in out


def test_cli_usage_empty_db(tmp_path, monkeypatch, capsys):
    from sqlalchemy import create_engine

    from praxis.db import Base

    db_path = tmp_path / "empty.db"
    monkeypatch.setenv("PRAXIS_DB_URL", f"sqlite:///{db_path}")
    Base.metadata.create_all(create_engine(f"sqlite:///{db_path}"))

    rc = main(["usage"])

    assert rc == 0
    assert "No LLM usage recorded yet" in capsys.readouterr().out


def test_cli_usage_missing_table(tmp_path, monkeypatch, capsys):
    """A pre-existing DB without the usage table degrades gracefully."""
    from sqlalchemy import create_engine

    from praxis.db import Base

    db_path = tmp_path / "old.db"
    monkeypatch.setenv("PRAXIS_DB_URL", f"sqlite:///{db_path}")
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine, tables=[Base.metadata.tables["candidates"]])

    rc = main(["usage"])

    assert rc == 0
    assert "No LLM usage recorded yet" in capsys.readouterr().out


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


def _write_golden(path, entries):
    path.write_text(json.dumps(entries), encoding="utf-8")


def _passing_report():
    from praxis.eval import AnalystEvalResult, EvalReport

    return EvalReport(
        analyst=[
            AnalystEvalResult(
                fixture_id="cpu-finetune",
                expected_verdict="accept",
                actual_verdict="accept",
                actual_score=7,
                expected_score_min=5,
                expected_score_max=10,
            )
        ]
    )


def test_cli_eval_prints_report(monkeypatch, capsys, tmp_path):
    golden = tmp_path / "golden.json"
    _write_golden(
        golden,
        [
            {
                "id": "cpu-finetune",
                "source": "arxiv",
                "url": "u",
                "title": "t",
                "raw_text": "r",
                "expected_verdict": "accept",
                "score_min": 5,
                "score_max": 10,
            }
        ],
    )
    monkeypatch.setattr("praxis.eval.run_eval", lambda *args, **kwargs: _passing_report())

    rc = main(["eval", "--golden", str(golden)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "OVERALL: PASS" in out
    assert "analyst: 1/1 passed" in out
    assert "cpu-finetune" in out


def test_cli_eval_failure_exit_code(monkeypatch, capsys, tmp_path):
    from praxis.eval import AnalystEvalResult, EvalReport

    golden = tmp_path / "golden.json"
    _write_golden(
        golden,
        [
            {
                "id": "cpu-finetune",
                "source": "arxiv",
                "url": "u",
                "title": "t",
                "raw_text": "r",
                "expected_verdict": "accept",
                "score_min": 5,
                "score_max": 10,
            }
        ],
    )
    failing = EvalReport(
        analyst=[
            AnalystEvalResult(
                fixture_id="cpu-finetune",
                expected_verdict="accept",
                actual_verdict="reject",
                actual_score=2,
                expected_score_min=5,
                expected_score_max=10,
            )
        ]
    )
    monkeypatch.setattr("praxis.eval.run_eval", lambda *args, **kwargs: failing)

    rc = main(["eval", "--golden", str(golden)])

    assert rc == 1
    assert "OVERALL: FAIL" in capsys.readouterr().out


def test_cli_eval_missing_golden_file(capsys):
    rc = main(["eval", "--golden", "does-not-exist.json"])

    assert rc == 1
    assert "not found" in capsys.readouterr().err
