"""Tests for the discover flow: scout+analyze table, picker, CLI wiring."""

from __future__ import annotations

import pytest

from praxis.cli import main
from praxis.config import HardwareProfile
from praxis.discover import (
    DiscoverResult,
    DiscoverRow,
    analyze_candidates,
    discover,
    focus_note,
    format_table,
    prompt_pick,
)


class _FakeCandidate:
    """Duck-typed Candidate for analyze flow tests (agents are mocked)."""

    def __init__(self, cid, url, title):
        self.id = cid
        self.url = url
        self.title = title
        self.source = "arxiv"
        self.raw_text = "raw"
        self.status = "new"
        self.feasibility_score = None
        self.technique_summary = None
        self.feasibility_reasoning = None


@pytest.fixture
def discover_db(db_engine, monkeypatch):
    """get_session bound to a fresh engine in every module discover touches.

    Import every module BEFORE patching: a module first imported while
    ``praxis.db.get_session`` is already patched would capture the test
    session factory permanently via its ``from praxis.db import get_session``
    binding (monkeypatch could never restore it).
    """
    import importlib

    from sqlalchemy.orm import Session

    session = Session(bind=db_engine)

    def fresh_session():
        return Session(bind=db_engine, expire_on_commit=False)

    modules = [
        importlib.import_module(name)
        for name in (
            "praxis.db",
            "praxis.pipeline",
            "praxis.discover",
            "praxis.design_io",
            "praxis.design",
        )
    ]
    for module in modules:
        monkeypatch.setattr(module, "get_session", fresh_session)
    yield session
    session.close()


def _row(index, candidate, verdict):
    return DiscoverRow(
        index=index,
        candidate=candidate,
        score=7,
        technique="LoRA fine-tuning on CPU",
        fit_reason="fits in 8 GB RAM",
        verdict=verdict,
    )


def test_discover_runs_scout_and_analyst_only(discover_db, monkeypatch):
    """discover() calls Scout + Analyst and never the Architect/Coder."""
    import praxis.agents as agents_module
    import praxis.discover as discover_module

    called = {"scout": False, "analyze": False, "architect": False, "coder": False}

    def fake_scout(**kwargs):
        called["scout"] = True
        return [_FakeCandidate(1, "https://a", "A"), _FakeCandidate(2, "https://b", "B")]

    def fake_analyze(candidate, profile, **kwargs):
        called["analyze"] = True
        candidate.feasibility_score = 8
        candidate.technique_summary = "build a CPU trainer"
        candidate.feasibility_reasoning = "runs on CPU"
        return candidate

    monkeypatch.setattr(agents_module, "scout", fake_scout)
    monkeypatch.setattr(agents_module, "analyze", fake_analyze)
    monkeypatch.setattr(
        agents_module, "architect", lambda **k: called.__setitem__("architect", True)
    )
    monkeypatch.setattr(agents_module, "coder", lambda **k: called.__setitem__("coder", True))

    # discover imports scout/analyze lazily from praxis.agents; ensure the
    # patched attributes are the ones it sees.
    assert discover_module.__name__ == "praxis.discover"

    result = discover("arxiv", "cpu training", limit=2, profile=HardwareProfile())
    assert called["scout"] is True
    assert called["analyze"] is True
    assert called["architect"] is False
    assert called["coder"] is False
    assert result.discovered == 2
    assert result.analyzed == 2
    assert len(result.rows) == 2
    assert all(r.verdict == "accept" for r in result.rows)


def test_format_table_shows_scores_and_verdicts():
    result = DiscoverResult(topic="cpu", source="arxiv", discovered=3, analyzed=2, rejected=1)
    result.rows = [
        _row(1, _FakeCandidate(1, "https://a", "Alpha"), "accept"),
        _row(2, _FakeCandidate(2, "https://b", "Beta"), "borderline"),
    ]
    text = format_table(result)
    assert "Discovered 3 candidate(s) for 'cpu'" in text
    assert "  1" in text and "accept" in text
    assert "borderline" in text
    assert "build: LoRA fine-tuning on CPU" in text
    assert "fit:   fits in 8 GB RAM" in text


def test_prompt_pick_interactive_number():
    result = DiscoverResult(topic="t", source="arxiv")
    result.rows = [_row(1, _FakeCandidate(1, "https://a", "A"), "accept")]
    picked = prompt_pick(result, input_fn=lambda _prompt: "1")
    assert picked is not None
    assert picked.index == 1


def test_prompt_pick_quit_and_invalid_then_valid():
    result = DiscoverResult(topic="t", source="arxiv")
    result.rows = [_row(1, _FakeCandidate(1, "https://a", "A"), "accept")]
    assert prompt_pick(result, input_fn=lambda _p: "q") is None
    assert prompt_pick(result, input_fn=lambda _p: "") is None

    answers = iter(["99", "1"])

    def fake_input(_p):
        return next(answers)

    picked = prompt_pick(result, input_fn=fake_input)
    assert picked is not None
    assert picked.index == 1


def test_prompt_pick_eof_returns_none():
    result = DiscoverResult(topic="t", source="arxiv")
    result.rows = [_row(1, _FakeCandidate(1, "https://a", "A"), "accept")]

    def eof_input(_p):
        raise EOFError

    assert prompt_pick(result, input_fn=eof_input) is None


def test_prompt_pick_no_pickable_rows():
    result = DiscoverResult(topic="t", source="arxiv")
    assert prompt_pick(result, input_fn=lambda _p: "1") is None


def test_focus_note_returns_text_or_none():
    row = _row(1, _FakeCandidate(1, "https://a", "A"), "accept")
    assert focus_note(row, input_fn=lambda _p: "  keep it tiny  ") == "keep it tiny"
    assert focus_note(row, input_fn=lambda _p: "") is None


def test_analyze_candidates_skips_failures(discover_db, monkeypatch):
    import praxis.agents as agents_module

    def failing_analyze(candidate, profile, **kwargs):
        raise RuntimeError("analyst boom")

    monkeypatch.setattr(agents_module, "analyze", failing_analyze)
    rows = analyze_candidates(
        [_FakeCandidate(1, "https://a", "A")], HardwareProfile()
    )
    assert rows == []


# ---------------------------------------------------------------------------
# CLI: praxis discover (non-interactive --pick) and praxis design
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_discover_env(tmp_path, monkeypatch):
    """Temp DB + offline grounding + temp cwd for design output files."""
    from sqlalchemy.orm import Session

    from praxis.db import Base, Candidate, get_engine

    db_path = tmp_path / "cli.db"
    monkeypatch.setenv("PRAXIS_DB_URL", f"sqlite:///{db_path}")
    engine = get_engine()
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        candidate = Candidate(
            source="arxiv",
            url="https://arxiv.org/abs/2401.12345",
            title="CPU Fine-Tune",
            raw_text="raw",
            status="analyzed",
        )
        session.add(candidate)
        session.commit()
        cid = candidate.id

    import praxis.design as design_module
    import praxis.grounding as grounding_module

    def fake_ground(candidate):
        return grounding_module.Grounding(chunks=[], provenance=[], source_kind="none")

    monkeypatch.setattr(grounding_module, "ground_candidate", fake_ground)
    monkeypatch.setattr(design_module, "ground_candidate", fake_ground)
    monkeypatch.chdir(tmp_path)
    return cid


def test_cli_discover_pick_runs_design(cli_discover_env, monkeypatch, capsys):
    """--pick N runs the full discover -> record_pick -> design -> files path."""
    from pathlib import Path as _Path

    import praxis.design as design_module
    from praxis import agents as agents_module
    from praxis.db import Candidate, get_session

    cid = cli_discover_env

    def fake_scout(**kwargs):
        session = get_session()
        try:
            c = session.get(Candidate, cid)
            c.feasibility_score = 8
            c.technique_summary = "build it"
            c.feasibility_reasoning = "runs on CPU"
            session.commit()
            return [c]
        finally:
            session.close()

    def fake_analyze(candidate, profile, **kwargs):
        return candidate

    monkeypatch.setattr(agents_module, "scout", fake_scout)
    monkeypatch.setattr(agents_module, "analyze", fake_analyze)

    from tests.test_design import GOOD_PASSES

    def fake_call_llm(prompt, system=None, model=None, **kwargs):
        if "Review it against the defect classes" in prompt:
            return '{"defects": []}'
        for content in GOOD_PASSES.values():
            title = content.split("\n", 1)[0].lstrip("# ").strip()
            if f"start with its '## {title}'" in prompt:
                return content
        raise AssertionError(f"unexpected prompt: {prompt[:120]!r}")

    monkeypatch.setattr(design_module, "call_llm", fake_call_llm)

    code = main(["discover", "--topic", "cpu", "--pick", "1", "--focus", "tiny scope"])
    assert code == 0
    out = capsys.readouterr().out
    assert "picked #1" in out.lower()
    assert any("DESIGN.md" in line for line in out.splitlines())
    # Design files actually written under the temp cwd.
    assert list(_Path("designs").glob("*/DESIGN.md"))
    # The pick was recorded in build_memory with the focus note.
    from praxis.db import BuildMemory

    session = get_session()
    try:
        memories = session.query(BuildMemory).all()
    finally:
        session.close()
    assert len(memories) == 1
    assert memories[0].outcome == "picked_for_design"
    assert "focus: tiny scope" in memories[0].technique


def test_cli_discover_quit(cli_discover_env, monkeypatch, capsys):
    """Answering 'q' at the picker exits 0 with no design work."""
    from praxis import agents as agents_module
    from praxis.db import Candidate, get_session

    cid = cli_discover_env

    def fake_scout(**kwargs):
        session = get_session()
        try:
            c = session.get(Candidate, cid)
            c.feasibility_score = 8
            c.technique_summary = "build it"
            c.feasibility_reasoning = "runs on CPU"
            session.commit()
            return [c]
        finally:
            session.close()

    monkeypatch.setattr(agents_module, "scout", fake_scout)
    monkeypatch.setattr(agents_module, "analyze", lambda candidate, profile, **k: candidate)

    import builtins

    answers = iter(["q"])

    def fake_input(_p=""):
        return next(answers)

    monkeypatch.setattr(builtins, "input", fake_input)

    code = main(["discover", "--topic", "cpu"])
    assert code == 0
    assert "no pick made" in capsys.readouterr().out


def test_cli_design_missing_candidate(tmp_path, monkeypatch):
    monkeypatch.setenv("PRAXIS_DB_URL", f"sqlite:///{tmp_path / 'none.db'}")
    code = main(["design", "999"])
    assert code == 1


# ---------------------------------------------------------------------------
# praxis design flags: --pass N, --resume, usage footer
# ---------------------------------------------------------------------------


def test_cli_design_resume_continues_partial(cli_discover_env, monkeypatch, capsys):
    """--resume continues an in_progress design instead of discarding it."""
    import json

    import praxis.design as design_module
    from praxis.db import Design, get_session, latest_design, save_design_pass
    from tests.test_design import GOOD_PASSES

    cid = cli_discover_env

    # Simulate a design interrupted after pass 1.
    session = get_session()
    design = Design(candidate_id=cid, status="in_progress")
    session.add(design)
    session.commit()
    design_id = design.id
    session.close()
    save_design_pass(design_id, "technique", GOOD_PASSES["technique"])

    calls = []

    def fake_call_llm(prompt, system=None, model=None, **kwargs):
        calls.append(prompt)
        for content in GOOD_PASSES.values():
            title = content.split("\n", 1)[0].lstrip("# ").strip()
            if f"start with its '## {title}'" in prompt:
                return content
        if "Review it against the defect classes" in prompt:
            return json.dumps({"defects": []})
        raise AssertionError(f"unexpected prompt: {prompt[:120]!r}")

    monkeypatch.setattr(design_module, "call_llm", fake_call_llm)

    code = main(["design", str(cid), "--resume"])
    assert code == 0
    out = capsys.readouterr().out
    assert "DESIGN.md" in out
    assert "LLM usage:" in out
    # The stored technique pass was reused, not regenerated: only the other 4
    # content passes + critic were called.
    assert len(calls) == 5
    stored = latest_design(cid)
    assert stored.status == "complete"
    assert json.loads(stored.passes_json)["technique"] == GOOD_PASSES["technique"]


def test_cli_design_without_resume_discards_partial(cli_discover_env, monkeypatch, capsys):
    """A fresh run (no --resume) abandons the stale partial design."""
    import json

    import praxis.design as design_module
    from praxis.db import Design, get_session, latest_design, save_design_pass
    from tests.test_design import GOOD_PASSES

    cid = cli_discover_env

    session = get_session()
    design = Design(candidate_id=cid, status="in_progress")
    session.add(design)
    session.commit()
    design_id = design.id
    session.close()
    save_design_pass(design_id, "technique", "old partial technique")

    regenerated = []

    def fake_call_llm(prompt, system=None, model=None, **kwargs):
        for content in GOOD_PASSES.values():
            title = content.split("\n", 1)[0].lstrip("# ").strip()
            if f"start with its '## {title}'" in prompt:
                regenerated.append(title)
                return content
        if "Review it against the defect classes" in prompt:
            return json.dumps({"defects": []})
        raise AssertionError(f"unexpected prompt: {prompt[:120]!r}")

    monkeypatch.setattr(design_module, "call_llm", fake_call_llm)

    code = main(["design", str(cid)])
    assert code == 0
    assert "Technique" in regenerated  # pass 1 re-ran
    stored = latest_design(cid)
    assert stored.status == "complete"
    assert "old partial technique" not in stored.passes_json


def test_cli_design_rerun_single_pass(cli_discover_env, monkeypatch, capsys):
    """--pass N regenerates only pass N, keeping the other stored passes."""
    import json

    import praxis.design as design_module
    from praxis.db import Design, get_session, latest_design, save_design_pass
    from tests.test_design import GOOD_PASSES

    cid = cli_discover_env

    session = get_session()
    design = Design(candidate_id=cid, status="complete")
    session.add(design)
    session.commit()
    design_id = design.id
    session.close()
    for pass_id in ("technique", "architecture", "data_contracts", "plan", "hardware_fit"):
        save_design_pass(design_id, pass_id, GOOD_PASSES[pass_id])
    # A completed design is reused as-is by --resume; to force a --pass rerun we
    # mark it in_progress like the generator would leave it mid-run.
    session = get_session()
    row = session.get(Design, design_id)
    row.status = "in_progress"
    session.commit()
    session.close()

    called_titles = []

    def fake_call_llm(prompt, system=None, model=None, **kwargs):
        for content in GOOD_PASSES.values():
            title = content.split("\n", 1)[0].lstrip("# ").strip()
            if f"start with its '## {title}'" in prompt:
                called_titles.append(title)
                return content.replace("top_k = 20", "top_k = 25")
        if "Review it against the defect classes" in prompt:
            return json.dumps({"defects": []})
        raise AssertionError(f"unexpected prompt: {prompt[:120]!r}")

    monkeypatch.setattr(design_module, "call_llm", fake_call_llm)

    code = main(["design", str(cid), "--pass", "3"])
    assert code == 0
    assert "re-running pass 3 (data_contracts)" in capsys.readouterr().out
    # Only the data_contracts pass and the critic ran.
    assert called_titles == ["Data Model & Contracts"]
    stored = latest_design(cid)
    passes = json.loads(stored.passes_json)
    assert "top_k = 25" in passes["data_contracts"]  # pass 3 regenerated
    assert passes["data_contracts"] != GOOD_PASSES["data_contracts"]
    assert passes["technique"] == GOOD_PASSES["technique"]  # other passes untouched


def test_cli_design_usage_footer_prints_totals(cli_discover_env, monkeypatch, capsys):
    """The design command prints token/cost totals from the usage ledger."""
    import json

    import praxis.design as design_module
    from praxis.db import LLMUsage, get_session
    from tests.test_design import GOOD_PASSES

    cid = cli_discover_env

    def fake_call_llm(prompt, system=None, model=None, **kwargs):
        for content in GOOD_PASSES.values():
            title = content.split("\n", 1)[0].lstrip("# ").strip()
            if f"start with its '## {title}'" in prompt:
                return content
        if "Review it against the defect classes" in prompt:
            return json.dumps({"defects": []})
        raise AssertionError(f"unexpected prompt: {prompt[:120]!r}")

    monkeypatch.setattr(design_module, "call_llm", fake_call_llm)

    # The fake replaced call_llm itself (no usage rows), so seed the ledger to
    # verify the aggregation printed in the footer.
    session = get_session()
    session.add(
        LLMUsage(
            model="groq/openai/gpt-oss-20b",
            stage="design",
            candidate_id=cid,
            total_tokens=1234,
            cost_usd=0.0025,
        )
    )
    session.commit()
    session.close()

    code = main(["design", str(cid), "--resume"])
    out = capsys.readouterr().out
    assert code == 0
    assert "LLM usage:" in out
    assert "1 calls, 1,234 tokens, $0.0025" in out
