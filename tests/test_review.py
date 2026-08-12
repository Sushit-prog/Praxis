"""Review gate tests: listing pending candidates, approve/reject transitions."""

from __future__ import annotations

from pathlib import Path

import praxis.agents as agents_module
from praxis.db import Blueprint, Candidate


def _borderline_candidate(db_session, *, url="https://x", score=5):
    cand = Candidate(
        source="arxiv",
        url=url,
        title="X",
        raw_text="r",
        status="borderline",
        technique_summary="t",
        feasibility_score=score,
        feasibility_reasoning="close call",
    )
    db_session.add(cand)
    db_session.commit()
    db_session.refresh(cand)
    return cand


def test_pending_candidates_only_borderline(db_session):
    from praxis.review import pending_candidates

    _borderline_candidate(db_session)
    db_session.add(
        Candidate(
            source="arxiv",
            url="https://done",
            title="Done",
            raw_text="z",
            status="prototyped",
        )
    )
    db_session.commit()

    pending = pending_candidates()

    assert len(pending) == 1
    assert pending[0].status == "borderline"
    assert pending[0].feasibility_score == 5


def test_approve_builds_candidate(db_session, monkeypatch, hardware_profile):
    from praxis.review import approve

    cand = _borderline_candidate(db_session)

    def fake_architect(**kwargs):
        bp = Blueprint(
            candidate_id=cand.id, feasibility_score=5.0, blueprint_md="# Plan"
        )
        db_session.add(bp)
        stored = db_session.get(Candidate, cand.id)
        stored.status = "blueprinted"
        db_session.commit()
        db_session.refresh(bp)
        return bp

    def fake_coder(**kwargs):
        stored = db_session.get(Candidate, cand.id)
        stored.status = "prototyped"
        db_session.commit()
        return Path("/tmp/proto")

    monkeypatch.setattr(agents_module, "architect", fake_architect)
    monkeypatch.setattr(agents_module, "coder", fake_coder)

    result = approve(cand.id, config=hardware_profile, retries=1)

    assert result.action == "approved"
    assert result.status == "prototyped"
    assert result.prototype_path == str(Path("/tmp/proto"))
    db_session.expire_all()
    assert db_session.get(Candidate, cand.id).status == "prototyped"


def test_approve_non_borderline_errors(db_session, monkeypatch, hardware_profile):
    from praxis.review import approve

    cand = Candidate(
        source="arxiv",
        url="https://done",
        title="Done",
        raw_text="z",
        status="prototyped",
    )
    db_session.add(cand)
    db_session.commit()
    db_session.refresh(cand)

    def fake_architect(**kwargs):
        raise AssertionError("architect should not run")

    monkeypatch.setattr(agents_module, "architect", fake_architect)

    result = approve(cand.id, config=hardware_profile, retries=1)

    assert result.error is not None
    assert "not awaiting review" in result.error
    db_session.expire_all()
    assert db_session.get(Candidate, cand.id).status == "prototyped"


def test_approve_build_failure_reports_status(db_session, monkeypatch, hardware_profile):
    from praxis.review import approve

    cand = _borderline_candidate(db_session)

    def fake_architect(**kwargs):
        raise RuntimeError("architect boom")

    monkeypatch.setattr(agents_module, "architect", fake_architect)

    result = approve(cand.id, config=hardware_profile, retries=1)

    assert result.status == "failed"
    db_session.expire_all()
    assert db_session.get(Candidate, cand.id).status == "failed"


def test_reject_marks_candidate_rejected(db_session):
    from praxis.review import reject

    cand = _borderline_candidate(db_session)

    result = reject(cand.id)

    assert result.action == "rejected"
    assert result.status == "rejected"
    db_session.expire_all()
    assert db_session.get(Candidate, cand.id).status == "rejected"


def test_reject_non_borderline_errors(db_session):
    from praxis.review import reject

    cand = Candidate(
        source="arxiv", url="https://done", title="Done", raw_text="z", status="analyzed"
    )
    db_session.add(cand)
    db_session.commit()
    db_session.refresh(cand)

    result = reject(cand.id)

    assert result.error is not None
    assert "not awaiting review" in result.error


def test_reviewed_candidate_resumed_skips_analyst(db_session, monkeypatch, hardware_profile):
    """A 'reviewed' candidate picked up by --resume builds without re-analysis."""
    from praxis.pipeline import run

    cand = _borderline_candidate(db_session)
    stored = db_session.get(Candidate, cand.id)
    stored.status = "reviewed"
    db_session.commit()

    def fake_scout(**kwargs):
        return []

    def fake_analyze(**kwargs):
        raise AssertionError("analyst must not run for an approved candidate")

    def fake_architect(**kwargs):
        bp = Blueprint(candidate_id=cand.id, feasibility_score=5.0, blueprint_md="# Plan")
        db_session.add(bp)
        stored = db_session.get(Candidate, cand.id)
        stored.status = "blueprinted"
        db_session.commit()
        db_session.refresh(bp)
        return bp

    def fake_coder(**kwargs):
        stored = db_session.get(Candidate, cand.id)
        stored.status = "prototyped"
        db_session.commit()
        return Path("/tmp/proto")

    monkeypatch.setattr(agents_module, "scout", fake_scout)
    monkeypatch.setattr(agents_module, "analyze", fake_analyze)
    monkeypatch.setattr(agents_module, "architect", fake_architect)
    monkeypatch.setattr(agents_module, "coder", fake_coder)

    result = run("arxiv", "attention", config=hardware_profile, limit=10, retries=1, resume=True)

    assert result.resumed == 1
    assert result.analyzed == 0
    assert result.prototyped == 1
    db_session.expire_all()
    assert db_session.get(Candidate, cand.id).status == "prototyped"


def test_review_cli_list_and_reject(monkeypatch, capsys, db_session):
    from praxis.cli import main

    _borderline_candidate(db_session)

    rc = main(["review"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Candidates awaiting review (borderline):" in out
    assert "[1] X" in out
    assert "score 5" in out

    rc = main(["review", "reject", "1"])
    assert rc == 0
    assert "rejected 1: X" in capsys.readouterr().out
    db_session.expire_all()
    assert db_session.get(Candidate, 1).status == "rejected"


def test_review_cli_approve(monkeypatch, capsys, db_session):
    from praxis.cli import main

    cand = _borderline_candidate(db_session)

    def fake_architect(**kwargs):
        bp = Blueprint(candidate_id=cand.id, feasibility_score=5.0, blueprint_md="# Plan")
        db_session.add(bp)
        stored = db_session.get(Candidate, cand.id)
        stored.status = "blueprinted"
        db_session.commit()
        db_session.refresh(bp)
        return bp

    def fake_coder(**kwargs):
        stored = db_session.get(Candidate, cand.id)
        stored.status = "prototyped"
        db_session.commit()
        return Path("/tmp/proto")

    monkeypatch.setattr(agents_module, "architect", fake_architect)
    monkeypatch.setattr(agents_module, "coder", fake_coder)

    rc = main(["review", "approve", "1"])

    assert rc == 0
    assert "approved 1: X -> prototyped" in capsys.readouterr().out
    db_session.expire_all()
    assert db_session.get(Candidate, 1).status == "prototyped"


def test_review_missing_candidate_errors(monkeypatch, capsys, db_session):
    from praxis.cli import main
    from praxis.review import reject

    rc = main(["review", "approve", "999"])
    assert rc == 1
    assert "no such candidate" in capsys.readouterr().err

    result = reject(999)
    assert result.error == "no such candidate"


def test_review_cli_empty(monkeypatch, capsys, db_session):
    from praxis.cli import main

    rc = main(["review"])

    assert rc == 0
    assert "No candidates awaiting review" in capsys.readouterr().out
