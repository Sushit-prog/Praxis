"""Pipeline tests: full Scout -> Analyst -> Architect -> Coder flow with mocked agents."""

from __future__ import annotations

import pytest
from sqlalchemy import select

import praxis.agents as agents_module
from praxis.agents.analyst import AnalysisResult
from praxis.db import Blueprint, Candidate, LLMUsage
from praxis.pipeline import CandidateOutcome, PipelineResult, format_summary, run


class _FakeCandidate:
    def __init__(self, url, title=None):
        self.url = url
        self.title = title or url


def _analysis_for(url):
    # Score 6: clear of the default threshold (4) plus the borderline margin (1),
    # so the standard flow tests are never accidentally routed as borderline.
    return AnalysisResult(
        technique_summary="t",
        feasibility_score=6,
        feasibility_reasoning="r",
        rejected=url.endswith("-no"),
    )


def test_run_flows_through_all_agents(monkeypatch, hardware_profile):
    candidates = [_FakeCandidate("https://a", "A"), _FakeCandidate("https://b-no", "B")]
    analysis_by_url = {c.url: _analysis_for(c.url) for c in candidates}

    def fake_scout(**kwargs):
        return candidates

    def fake_analyze(**kwargs):
        return analysis_by_url[kwargs["candidate"].url]

    def fake_architect(**kwargs):
        return f"# Blueprint for {kwargs['candidate'].url}"

    def fake_coder(**kwargs):
        return kwargs["blueprint"]

    monkeypatch.setattr(agents_module, "scout", fake_scout)
    monkeypatch.setattr(agents_module, "analyze", fake_analyze)
    monkeypatch.setattr(agents_module, "architect", fake_architect)
    monkeypatch.setattr(agents_module, "coder", fake_coder)

    result = run("arxiv", "attention", config=hardware_profile, limit=10, retries=1)

    assert isinstance(result, PipelineResult)
    assert result.discovered == 2
    assert result.analyzed == 1
    assert result.rejected == 1
    assert result.blueprinted == 1
    assert result.prototyped == 1
    assert result.failed == 0
    assert [o.title for o in result.candidates if o.status == "prototyped"] == ["A"]


def test_run_passes_analysis_to_architect(monkeypatch, hardware_profile):
    cand = _FakeCandidate("https://a", "A")
    analysis = _analysis_for("https://a")

    def fake_scout(**kwargs):
        return [cand]

    def fake_analyze(**kwargs):
        return analysis

    seen = {}

    def fake_architect(**kwargs):
        seen["analysis"] = kwargs["analysis"]
        return "# ok"

    monkeypatch.setattr(agents_module, "scout", fake_scout)
    monkeypatch.setattr(agents_module, "analyze", fake_analyze)
    monkeypatch.setattr(agents_module, "architect", fake_architect)
    monkeypatch.setattr(agents_module, "coder", lambda **kwargs: kwargs["blueprint"])

    run("arxiv", "attention", config=hardware_profile, limit=10, retries=1)

    assert seen["analysis"] is analysis


def test_run_all_rejected_skips_coder(monkeypatch, hardware_profile):
    cand = _FakeCandidate("https://a-no", "A")

    def fake_scout(**kwargs):
        return [cand]

    def fake_analyze(**kwargs):
        return _analysis_for("https://a-no")

    coder_calls = {}

    def fake_coder(**kwargs):
        coder_calls["blueprint"] = kwargs["blueprint"]
        return kwargs["blueprint"]

    monkeypatch.setattr(agents_module, "scout", fake_scout)
    monkeypatch.setattr(agents_module, "analyze", fake_analyze)
    monkeypatch.setattr(agents_module, "coder", fake_coder)

    result = run("arxiv", "attention", config=hardware_profile, limit=10, retries=1)

    assert result.rejected == 1
    assert result.prototyped == 0
    assert coder_calls == {}


def test_run_borderline_routes_separately(db_session, monkeypatch, hardware_profile):
    """A borderline analysis is held for review, not built or rejected."""
    cand = _FakeCandidate("https://a", "A")

    def fake_scout(**kwargs):
        return [cand]

    def fake_analyze(**kwargs):
        return AnalysisResult(
            technique_summary="t",
            feasibility_score=5,
            feasibility_reasoning="r",
            rejected=False,
            borderline=True,
        )

    architect_calls = {"n": 0}

    def fake_architect(**kwargs):
        architect_calls["n"] += 1
        return "# ok"

    monkeypatch.setattr(agents_module, "scout", fake_scout)
    monkeypatch.setattr(agents_module, "analyze", fake_analyze)
    monkeypatch.setattr(agents_module, "architect", fake_architect)
    monkeypatch.setattr(agents_module, "coder", lambda **kwargs: "proto")

    result = run("arxiv", "attention", config=hardware_profile, limit=10, retries=1)

    assert result.borderline == 1
    assert result.analyzed == 0
    assert result.blueprinted == 0
    assert result.candidates[0].status == "borderline"
    assert architect_calls["n"] == 0


def test_run_continues_past_analyst_failure(monkeypatch, hardware_profile):
    ok = _FakeCandidate("https://a", "A")
    bad = _FakeCandidate("https://boom", "Boom")

    def fake_scout(**kwargs):
        return [ok, bad]

    def fake_analyze(**kwargs):
        if kwargs["candidate"].url == "https://boom":
            raise RuntimeError("analyst boom")
        return _analysis_for("https://a")

    def fake_architect(**kwargs):
        return "# ok"

    monkeypatch.setattr(agents_module, "scout", fake_scout)
    monkeypatch.setattr(agents_module, "analyze", fake_analyze)
    monkeypatch.setattr(agents_module, "architect", fake_architect)
    monkeypatch.setattr(agents_module, "coder", lambda **kwargs: "proto")

    result = run("arxiv", "attention", config=hardware_profile, limit=10, retries=1)

    assert result.discovered == 2
    assert result.analyzed == 1
    assert result.failed == 1
    assert [o.status for o in result.candidates] == ["prototyped", "failed"]


def test_run_continues_past_coder_failure(monkeypatch, hardware_profile):
    cand = _FakeCandidate("https://a", "A")

    def fake_scout(**kwargs):
        return [cand]

    def fake_analyze(**kwargs):
        return _analysis_for("https://a")

    def fake_architect(**kwargs):
        return "# ok"

    monkeypatch.setattr(agents_module, "scout", fake_scout)
    monkeypatch.setattr(agents_module, "analyze", fake_analyze)
    monkeypatch.setattr(agents_module, "architect", fake_architect)
    monkeypatch.setattr(agents_module, "coder", lambda **kwargs: None)

    result = run("arxiv", "attention", config=hardware_profile, limit=10, retries=1)

    assert result.prototyped == 0
    assert result.failed == 1
    assert result.candidates[0].status == "prototype_failed"


def test_run_resume_processes_unfinished_candidates(db_session, monkeypatch, hardware_profile):
    """--resume reprocesses status `new`/`failed` candidates and skips finished ones."""
    unfinished = Candidate(
        source="arxiv", url="https://old", title="Old", raw_text="x", status="new"
    )
    failed = Candidate(
        source="arxiv", url="https://failed", title="Failed", raw_text="y", status="failed"
    )
    done = Candidate(
        source="arxiv", url="https://done", title="Done", raw_text="z", status="prototyped"
    )
    db_session.add_all([unfinished, failed, done])
    db_session.commit()
    for cand in (unfinished, failed, done):
        db_session.refresh(cand)

    processed = []

    def fake_scout(**kwargs):
        return []

    def fake_analyze(**kwargs):
        processed.append(kwargs["candidate"].url)
        return _analysis_for("https://a")

    monkeypatch.setattr(agents_module, "scout", fake_scout)
    monkeypatch.setattr(agents_module, "analyze", fake_analyze)
    monkeypatch.setattr(agents_module, "architect", lambda **kwargs: "# ok")
    monkeypatch.setattr(agents_module, "coder", lambda **kwargs: "proto")

    result = run(
        "arxiv", "attention", config=hardware_profile, limit=10, retries=1, resume=True
    )

    assert result.resumed == 2
    assert result.discovered == 0  # scout found nothing new
    assert set(processed) == {"https://old", "https://failed"}
    assert "https://done" not in processed
    assert result.analyzed == 2
    assert result.prototyped == 2


def test_run_resume_combines_with_new_scouting(db_session, monkeypatch, hardware_profile):
    """Resumed and freshly scouted candidates are processed in one batch."""
    old = Candidate(
        source="arxiv", url="https://old", title="Old", raw_text="x", status="new"
    )
    db_session.add(old)
    db_session.commit()
    db_session.refresh(old)

    fresh = _FakeCandidate("https://fresh", "Fresh")
    processed = []

    def fake_scout(**kwargs):
        return [fresh]

    def fake_analyze(**kwargs):
        processed.append(kwargs["candidate"].url)
        return _analysis_for("https://a")

    monkeypatch.setattr(agents_module, "scout", fake_scout)
    monkeypatch.setattr(agents_module, "analyze", fake_analyze)
    monkeypatch.setattr(agents_module, "architect", lambda **kwargs: "# ok")
    monkeypatch.setattr(agents_module, "coder", lambda **kwargs: "proto")

    result = run(
        "arxiv", "attention", config=hardware_profile, limit=10, retries=1, resume=True
    )

    assert result.resumed == 1
    assert result.discovered == 1
    assert set(processed) == {"https://old", "https://fresh"}
    assert result.analyzed == 2


def test_run_resume_failed_again_stays_failed(db_session, monkeypatch, hardware_profile):
    """A resumed candidate that fails again is re-marked failed, not dropped."""
    cand = Candidate(
        source="arxiv", url="https://boom", title="Boom", raw_text="x", status="failed"
    )
    db_session.add(cand)
    db_session.commit()
    db_session.refresh(cand)

    def fake_scout(**kwargs):
        return []

    def fake_analyze(**kwargs):
        raise RuntimeError("analyst boom again")

    monkeypatch.setattr(agents_module, "scout", fake_scout)
    monkeypatch.setattr(agents_module, "analyze", fake_analyze)
    monkeypatch.setattr(agents_module, "architect", lambda **kwargs: "# ok")
    monkeypatch.setattr(agents_module, "coder", lambda **kwargs: "proto")

    result = run(
        "arxiv", "attention", config=hardware_profile, limit=10, retries=1, resume=True
    )

    assert result.resumed == 1
    assert result.failed == 1
    db_session.expire_all()
    assert db_session.get(Candidate, cand.id).status == "failed"


def test_run_resume_survives_scout_failure(db_session, monkeypatch, hardware_profile):
    """With resume, a scout failure still processes the unfinished candidates."""
    cand = Candidate(
        source="arxiv", url="https://old", title="Old", raw_text="x", status="new"
    )
    db_session.add(cand)
    db_session.commit()
    db_session.refresh(cand)

    def fake_scout(**kwargs):
        raise RuntimeError("network down")

    def fake_analyze(**kwargs):
        return _analysis_for("https://a")

    monkeypatch.setattr(agents_module, "scout", fake_scout)
    monkeypatch.setattr(agents_module, "analyze", fake_analyze)
    monkeypatch.setattr(agents_module, "architect", lambda **kwargs: "# ok")
    monkeypatch.setattr(agents_module, "coder", lambda **kwargs: "proto")

    result = run(
        "arxiv", "attention", config=hardware_profile, limit=10, retries=1, resume=True
    )

    assert result.resumed == 1
    assert result.analyzed == 1
    assert result.prototyped == 1


def test_run_without_resume_ignores_existing_candidates(db_session, monkeypatch, hardware_profile):
    """Default run leaves pre-existing candidates untouched."""
    cand = Candidate(
        source="arxiv", url="https://old", title="Old", raw_text="x", status="new"
    )
    db_session.add(cand)
    db_session.commit()

    def fake_scout(**kwargs):
        return []

    seen = {}

    def fake_analyze(**kwargs):
        seen["called"] = True
        return _analysis_for("https://a")

    monkeypatch.setattr(agents_module, "scout", fake_scout)
    monkeypatch.setattr(agents_module, "analyze", fake_analyze)
    monkeypatch.setattr(agents_module, "architect", lambda **kwargs: "# ok")
    monkeypatch.setattr(agents_module, "coder", lambda **kwargs: "proto")

    result = run("arxiv", "attention", config=hardware_profile, limit=10, retries=1)

    assert result.discovered == 0
    assert result.resumed == 0
    assert "called" not in seen


def test_run_reports_usage_delta(db_session, monkeypatch, hardware_profile):
    """Usage rows written during a run show up in the result and summary."""
    cand = _FakeCandidate("https://a", "A")

    def fake_scout(**kwargs):
        return [cand]

    def fake_analyze(**kwargs):
        db_session.add(
            LLMUsage(
                model="m",
                stage="analyst",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
                cost_usd=0.001,
                latency_ms=5,
            )
        )
        db_session.commit()
        return _analysis_for("https://a")

    monkeypatch.setattr(agents_module, "scout", fake_scout)
    monkeypatch.setattr(agents_module, "analyze", fake_analyze)
    monkeypatch.setattr(agents_module, "architect", lambda **kwargs: "# ok")
    monkeypatch.setattr(agents_module, "coder", lambda **kwargs: "proto")

    result = run("arxiv", "attention", config=hardware_profile, limit=10, retries=1)

    assert result.usage_calls == 1
    assert result.usage_total_tokens == 2
    assert result.usage_cost_usd == pytest.approx(0.001)
    text = format_summary(result)
    assert "LLM spend: $0.0010 across 1 calls (2 tokens)" in text


def test_format_summary():
    result = PipelineResult(
        source="arxiv",
        topic="attention",
        discovered=1,
        analyzed=1,
        blueprinted=1,
        prototyped=1,
        candidates=[
            CandidateOutcome(
                title="A",
                url="https://a",
                status="prototyped",
                prototype_path="/tmp/proto-a",
            )
        ],
    )
    text = format_summary(result)
    assert "Summary for topic='attention' source=arxiv" in text
    assert "prototyped: 1" in text
    assert "A [prototyped] (/tmp/proto-a)" in text


def test_run_integration_db_state(db_session, monkeypatch, hardware_profile, tmp_path):
    """Full pipeline with mocked agents; assert DB state for a mixed batch."""
    proto_path = tmp_path / "proto-a"

    def make_candidate(url, title):
        cand = Candidate(source="arxiv", url=url, title=title, raw_text="body", status="new")
        db_session.add(cand)
        db_session.commit()
        db_session.refresh(cand)
        return cand

    a = make_candidate("https://ok", "Ok one")
    b = make_candidate("https://no", "Rejected one")
    c = make_candidate("https://boom", "Analyze crash")
    d = make_candidate("https://aboom", "Architect crash")
    candidates = [a, b, c, d]

    def fake_scout(**kwargs):
        return candidates

    def fake_analyze(**kwargs):
        cand = kwargs["candidate"]
        if cand.url == "https://boom":
            raise RuntimeError("analyst boom")
        result = AnalysisResult(
            technique_summary="t",
            feasibility_score=6 if cand.url == "https://ok" else 2,
            feasibility_reasoning="r",
            rejected=cand.url == "https://no",
        )
        stored = db_session.get(Candidate, cand.id)
        stored.status = "rejected" if result.rejected else "analyzed"
        db_session.commit()
        return result

    def fake_architect(**kwargs):
        cand = kwargs["candidate"]
        if cand.url == "https://aboom":
            raise RuntimeError("architect boom")
        bp = Blueprint(candidate_id=cand.id, feasibility_score=6.0, blueprint_md="# Plan")
        db_session.add(bp)
        stored = db_session.get(Candidate, cand.id)
        stored.status = "blueprinted"
        db_session.commit()
        db_session.refresh(bp)
        return bp

    def fake_coder(**kwargs):
        bp = kwargs["blueprint"]
        stored = db_session.get(Blueprint, bp.id)
        stored.prototype_path = str(proto_path)
        stored_cand = db_session.get(Candidate, bp.candidate_id)
        stored_cand.status = "prototyped"
        db_session.commit()
        return proto_path

    monkeypatch.setattr(agents_module, "scout", fake_scout)
    monkeypatch.setattr(agents_module, "analyze", fake_analyze)
    monkeypatch.setattr(agents_module, "architect", fake_architect)
    monkeypatch.setattr(agents_module, "coder", fake_coder)

    result = run("arxiv", "attention", config=hardware_profile, limit=10, retries=1)

    assert result.discovered == 4
    assert result.analyzed == 2
    assert result.rejected == 1
    assert result.blueprinted == 1
    assert result.prototyped == 1
    assert result.failed == 2

    db_session.expire_all()
    assert db_session.get(Candidate, a.id).status == "prototyped"
    assert db_session.get(Candidate, b.id).status == "rejected"
    assert db_session.get(Candidate, c.id).status == "failed"
    assert db_session.get(Candidate, d.id).status == "failed"

    bp_a = db_session.scalars(select(Blueprint).where(Blueprint.candidate_id == a.id)).first()
    assert bp_a is not None
    assert bp_a.prototype_path == str(proto_path)
    assert bp_a.blueprint_md == "# Plan"
