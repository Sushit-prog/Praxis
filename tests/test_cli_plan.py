"""CLI tests for `praxis plan` (the docs-pack command)."""

from __future__ import annotations

import json

import pytest

from praxis.cli import build_parser, main
from praxis.db import LLMUsage, get_session
from praxis.planning import PACK_PASS_IDS
from tests.test_design import GOOD_PASSES
from tests.test_docs_pack import ALL_FILES
from tests.test_planning import PACK_CONTENT, _pack_completion, _seed_design

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_plan_parser_defaults():
    args = build_parser().parse_args(["plan", "1"])

    assert args.command == "plan"
    assert args.candidate_id == 1
    assert args.rerun_pass is None
    assert args.resume is False
    assert args.run_critic is False


def test_plan_parser_critic_and_resume_flags():
    base = ["plan", "1"]

    assert build_parser().parse_args([*base, "--critic"]).run_critic is True
    assert build_parser().parse_args([*base, "--no-critic"]).run_critic is False
    assert build_parser().parse_args([*base, "--resume"]).resume is True


def test_plan_parser_pass_choices():
    assert build_parser().parse_args(["plan", "1", "--pass", "3"]).rerun_pass == 3
    with pytest.raises(SystemExit):
        build_parser().parse_args(["plan", "1", "--pass", "7"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["plan", "1", "--pass", "0"])


# ---------------------------------------------------------------------------
# Fixture + helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def plan_env(tmp_path, monkeypatch):
    """Temp DB with designed/partial/undesigned candidates; offline, unpaced."""
    from sqlalchemy.orm import Session

    import praxis.design as design_module
    import praxis.grounding as grounding_module
    import praxis.planning as planning_module
    from praxis.db import Base, Candidate, get_engine

    monkeypatch.setenv("PRAXIS_DB_URL", f"sqlite:///{tmp_path / 'plan.db'}")
    monkeypatch.chdir(tmp_path)  # designs/ lands in tmp; repo .env stays hidden
    monkeypatch.setenv("PRAXIS_DESIGN_MODEL", "groq/openai/gpt-oss-120b")
    monkeypatch.setenv("PRAXIS_DETECT_HW", "0")

    engine = get_engine()
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        full = Candidate(
            source="arxiv",
            url="https://arxiv.org/abs/2401.12345",
            title="CPU Fine-Tune",
            raw_text="Fine-tune a small transformer on CPU.",
            status="analyzed",
        )
        none_ = Candidate(
            source="arxiv",
            url="https://example.com/a",
            title="No Design Yet",
            raw_text="r",
            status="analyzed",
        )
        part = Candidate(
            source="arxiv",
            url="https://example.com/b",
            title="Partial Design",
            raw_text="r",
            status="analyzed",
        )
        session.add_all([full, none_, part])
        session.commit()
        ids = {"full": full.id, "no_design": none_.id, "partial": part.id}

    _seed_design(ids["full"], GOOD_PASSES, status="complete")
    _seed_design(ids["partial"], {"technique": GOOD_PASSES["technique"]})

    def fake_ground(candidate):
        return grounding_module.Grounding(chunks=[], provenance=[], source_kind="none")

    monkeypatch.setattr(grounding_module, "ground_candidate", fake_ground)
    monkeypatch.setattr(design_module, "ground_candidate", fake_ground)
    monkeypatch.setattr(planning_module, "ground_candidate", fake_ground)
    monkeypatch.setattr(planning_module, "_sleep", lambda seconds: None)
    yield ids


def _recording(fake, candidate_id):
    """Fake call_llm that also writes the LLMUsage row the real one would."""

    def wrapper(prompt, system=None, model=None, **kwargs):
        content = fake(prompt, system=system, model=model, **kwargs)
        session = get_session()
        try:
            session.add(
                LLMUsage(
                    model=model or "test/model",
                    stage=kwargs.get("stage"),
                    candidate_id=candidate_id,
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    cost_usd=0.001,
                    latency_ms=1,
                )
            )
            session.commit()
        finally:
            session.close()
        return content

    return wrapper


def _run(candidate_id, monkeypatch, *flags, fake=None, critic_flag_once=False):
    """Run `praxis plan <candidate_id> <flags>` against a fake LLM."""
    import praxis.design as design_module

    fake = fake if fake is not None else _pack_completion(
        critic_flag_once=critic_flag_once
    )
    monkeypatch.setattr(design_module, "call_llm", _recording(fake, candidate_id))
    rc = main(["plan", str(candidate_id), *flags])
    return rc, fake


def _add_pack_passes(candidate_id, *pass_ids):
    from praxis.db import latest_design, save_design_pass

    design = latest_design(candidate_id)
    assert design is not None
    for pass_id in pass_ids:
        save_design_pass(design.id, pass_id, PACK_CONTENT[pass_id])


def _stored_passes(candidate_id):
    from praxis.db import latest_design

    return json.loads(latest_design(candidate_id).passes_json)


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------


def test_plan_missing_candidate(plan_env, capsys):
    assert main(["plan", "999"]) == 1
    assert "no candidate with id 999" in capsys.readouterr().err


def test_plan_requires_design_first(plan_env, capsys):
    candidate_id = plan_env["no_design"]

    assert main(["plan", str(candidate_id)]) == 1
    err = capsys.readouterr().err
    assert "has no design" in err
    assert f"run `praxis design {candidate_id}` first" in err


def test_plan_partial_design_points_at_resume(plan_env, capsys):
    candidate_id = plan_env["partial"]

    assert main(["plan", str(candidate_id)]) == 1
    err = capsys.readouterr().err
    assert (
        "design incomplete (missing: architecture, data_contracts, plan, "
        "hardware_fit)" in err
    )
    assert f"run `praxis design {candidate_id} --resume` first" in err


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


def test_plan_success_writes_docs(plan_env, monkeypatch, capsys, tmp_path):
    rc, fake = _run(plan_env["full"], monkeypatch)
    out = capsys.readouterr().out

    assert rc == 0
    assert f"planning candidate {plan_env['full']}: CPU Fine-Tune" in out
    assert f"plan complete: candidate {plan_env['full']}" in out
    assert "LLM usage: 6 calls, 180 tokens, $0.0060" in out
    assert "pack critic" not in out  # off by default: no note, no defect lines

    stages = [stage for stage, _ in fake.calls]
    assert stages.count("plan") == 6
    assert stages.count("design_critic") == 0

    docs = tmp_path / "designs" / "001-cpu-fine-tune" / "docs"
    assert str(docs) in out
    assert sorted(p.name for p in docs.iterdir()) == ALL_FILES


def test_plan_complete_pack_plain_rerun_keeps_pack(
    plan_env, monkeypatch, capsys, tmp_path
):
    _add_pack_passes(plan_env["full"], *PACK_PASS_IDS)

    rc, fake = _run(plan_env["full"], monkeypatch)
    out = capsys.readouterr().out

    assert rc == 0
    assert fake.calls == []  # complete pack: nothing to restart, nothing missing
    assert "LLM usage: 0 calls" in out
    assert (tmp_path / "designs" / "001-cpu-fine-tune" / "docs" / "PRD.md").exists()


# ---------------------------------------------------------------------------
# --resume / restart semantics
# ---------------------------------------------------------------------------


def test_plan_partial_pack_restarts_without_resume(plan_env, monkeypatch, capsys):
    _add_pack_passes(plan_env["full"], "prd", "rules")

    rc, fake = _run(plan_env["full"], monkeypatch)
    assert rc == 0
    # The stored pack documents were discarded: all six regenerate.
    assert len(fake.calls) == 6
    assert any("start with its '## Product Requirements'" in p for _, p in fake.calls)


def test_plan_partial_pack_resumes(plan_env, monkeypatch, capsys):
    _add_pack_passes(plan_env["full"], "prd", "rules")

    rc, fake = _run(plan_env["full"], monkeypatch, "--resume")
    assert rc == 0
    assert len(fake.calls) == 4
    prompts = [prompt for _, prompt in fake.calls]
    assert not any("start with its '## Product Requirements'" in p for p in prompts)
    assert not any("start with its '## Engineering Rules'" in p for p in prompts)
    stored = _stored_passes(plan_env["full"])
    assert stored["prd"] == PACK_CONTENT["prd"]
    assert stored["rules"] == PACK_CONTENT["rules"]


def test_plan_pass_reruns_single_document(plan_env, monkeypatch, capsys):
    _add_pack_passes(plan_env["full"], *PACK_PASS_IDS)

    rc, fake = _run(plan_env["full"], monkeypatch, "--pass", "3")
    out = capsys.readouterr().out

    assert rc == 0
    assert "re-running pass 3 (test_plan) only" in out
    assert len(fake.calls) == 1
    stage, prompt = fake.calls[0]
    assert stage == "plan"
    assert "start with its '## Test Plan'" in prompt


# ---------------------------------------------------------------------------
# Critic flags
# ---------------------------------------------------------------------------


def test_plan_critic_flag_flags_and_regenerates(plan_env, monkeypatch, capsys):
    rc, fake = _run(plan_env["full"], monkeypatch, "--critic", critic_flag_once=True)
    out = capsys.readouterr().out

    assert rc == 0
    stages = [stage for stage, _ in fake.calls]
    assert stages.count("plan") == 6
    assert stages.count("design_critic") == 7  # 6 documents + 1 re-review
    assert stages.count("design") == 1  # the flagged prd was regenerated
    assert "critic: [UNSOURCED_CLAIM] Product Requirements" in out
    assert f"plan complete: candidate {plan_env['full']}" in out


def test_plan_no_critic_flag_skips_critic(plan_env, monkeypatch, capsys):
    rc, fake = _run(plan_env["full"], monkeypatch, "--no-critic")
    out = capsys.readouterr().out

    assert rc == 0
    stages = [stage for stage, _ in fake.calls]
    assert stages.count("design_critic") == 0
    assert stages.count("plan") == 6
    assert "critic:" not in out


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


def test_plan_failure_prints_resume_hint(plan_env, monkeypatch, capsys):
    import praxis.design as design_module

    base = _pack_completion()

    def flaky(prompt, system=None, model=None, **kwargs):
        if kwargs.get("stage") == "plan" and "Engineering Rules" in prompt:
            raise RuntimeError("provider exploded")
        return base(prompt, system=system, model=model, **kwargs)

    monkeypatch.setattr(
        design_module, "call_llm", _recording(flaky, plan_env["full"])
    )
    rc = main(["plan", str(plan_env["full"])])
    err = capsys.readouterr().err

    assert rc == 1
    assert (
        f"error: plan failed for candidate {plan_env['full']}: provider exploded"
        in err
    )
    assert "completed pack passes: prd" in err
    assert "re-run the same command with --resume to continue" in err
