"""Pipeline tests: full Scout -> Analyst -> Architect -> Coder flow with mocked agents."""

from __future__ import annotations

import praxis.agents as agents_module
from praxis.agents.analyst import AnalysisResult
from praxis.pipeline import run_pipeline


class _FakeCandidate:
    def __init__(self, url):
        self.url = url


def _analysis_for(url):
    return AnalysisResult(
        technique_summary="t",
        feasibility_score=5,
        feasibility_reasoning="r",
        rejected=url.endswith("-no"),
    )


def test_run_pipeline_flows_through_all_agents(monkeypatch, hardware_profile):
    candidates = [_FakeCandidate("https://a"), _FakeCandidate("https://b-no")]
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

    out = run_pipeline("arxiv", "attention", config=hardware_profile, limit=10, retries=1)

    assert out == ["# Blueprint for https://a"]


def test_run_pipeline_passes_analysis_to_architect(monkeypatch, hardware_profile):
    cand = _FakeCandidate("https://a")
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

    run_pipeline("arxiv", "attention", config=hardware_profile, limit=10, retries=1)

    assert seen["analysis"] is analysis


def test_run_pipeline_all_rejected_gives_empty_blueprints(monkeypatch, hardware_profile):
    cand = _FakeCandidate("https://a-no")

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

    out = run_pipeline("arxiv", "attention", config=hardware_profile, limit=10, retries=1)

    assert out == []
    assert coder_calls["blueprint"] == []
