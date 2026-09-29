"""Tests for the planning-pack pass generation (`praxis plan`)."""

from __future__ import annotations

import json

import pytest

from praxis.config import FactsSheet, HardwareProfile
from praxis.db import Blueprint, Candidate, Design, LLMUsage
from praxis.design import (
    _PASS_SPECS,
    GROUNDING_CHAR_BUDGETS,
    PASS_IDS,
    _pass_instruction,
)
from praxis.planning import (
    PACK_OUTPUT_TOKENS,
    PACK_PASS_IDS,
    RAW_FILE_PASSES,
    PackResult,
    generate_doc_pack,
)
from tests.test_design import GOOD_PASSES, _Candidate

# Canned pack outputs, one per pack pass; every markdown document starts with
# its '## <title>' heading (the prompt trailer demands it) and env_example is
# raw VAR=value content (its trailer demands bare file content).
PACK_CONTENT = {
    "prd": (
        "## Product Requirements\n\n"
        "- FR-001 Fetch 10 chunks from a sample corpus (acceptance: "
        "test_retriever_returns_chunks passes) [source 1].\n"
        "- FR-002 Report peak RSS before and after a run (inference).\n"
        "- Non-goals: GPU training, web UI.\n"
    ),
    "rules": (
        "## Engineering Rules\n\n"
        "- pip/uv only; no server infrastructure.\n"
        "- Persistence stays SQLite or JSON.\n"
        "- Every task ships its test and passes the phase test gate.\n"
    ),
    "test_plan": (
        "## Test Plan\n\n"
        "- Unit: test_retriever_returns_chunks asserts 10 chunks (TASK-001).\n"
        "- Eval: MRR@10 baseline BM25 = 0.30 [source 1].\n"
        "- Hardware: peak RSS below the facts-sheet ceiling.\n"
    ),
    "security": (
        "## Security Notes\n\n"
        "- Papers and READMEs are untrusted read-only input.\n"
        "- No secrets in code; .env stays out of version control.\n"
    ),
    "readme": (
        "## README Skeleton\n\n"
        "TODO: fill in install and usage once Phase 1 lands.\n"
    ),
    "env_example": (
        "# Data paths\n"
        "DATA_DIR=./data\n"
        "DB_PATH=./memguard.db\n"
        "# Budget\n"
        "MONTHLY_BUDGET_USD=5.0\n"
    ),
}

_BLUEPRINT_MD = (
    "# MemGuard - Blueprint\n"
    "## Problem Statement\n"
    "Prompt-injection defenses for local agent loops [source 1].\n\n"
    "## Proposed Architecture\n"
    "Sanitizer component.\n\n"
    "## Phased Build Plan\n"
    "Phase 1 slice marker.\n\n"
    "## Deferred to Later Versions\n"
    "Vector store (deferred).\n\n"
    "## Difficulty & Time Estimate\n"
    "3 days (inference).\n"
)

_PACK_DEFECT = {
    "class": "UNSOURCED_CLAIM",
    "section": "Product Requirements",
    "defect": "FR-001 carries no [source N] reference and no (inference) label.",
    "fix": "Label the origin of FR-001.",
}


# ---------------------------------------------------------------------------
# Fixtures (local copies of the design tests' session-binding pattern)
# ---------------------------------------------------------------------------


@pytest.fixture
def planning_db(db_engine, monkeypatch):
    """Fresh DB tables and a persisted candidate; get_session bound to it."""
    import importlib

    from sqlalchemy.orm import Session

    session = Session(bind=db_engine)
    candidate = Candidate(
        source="arxiv",
        url="https://arxiv.org/abs/2401.12345",
        title="CPU Fine-Tune",
        raw_text="Fine-tune a small transformer on CPU.",
        status="analyzed",
    )
    session.add(candidate)
    session.commit()
    candidate_id = candidate.id

    def fresh_session():
        return Session(bind=db_engine, expire_on_commit=False)

    # Import every module BEFORE patching: a module first imported while
    # praxis.db.get_session is already patched would capture the test session
    # factory permanently via its ``from praxis.db import get_session``
    # binding (monkeypatch could never restore it).
    modules = [
        importlib.import_module(name)
        for name in (
            "praxis.db",
            "praxis.design",
            "praxis.discover",
            "praxis.design_io",
            "praxis.export",
            "praxis.planning",
        )
    ]
    for module in modules:
        monkeypatch.setattr(module, "get_session", fresh_session)
    yield candidate_id
    session.close()


@pytest.fixture
def no_grounding(monkeypatch):
    """Keep the planning tests offline: no arXiv fetch."""
    import praxis.design as design_module
    import praxis.grounding as grounding_module
    import praxis.planning as planning_module

    def fake_ground(candidate):
        return grounding_module.Grounding(chunks=[], provenance=[], source_kind="none")

    monkeypatch.setattr(grounding_module, "ground_candidate", fake_ground)
    monkeypatch.setattr(design_module, "ground_candidate", fake_ground)
    monkeypatch.setattr(planning_module, "ground_candidate", fake_ground)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_design(candidate_id, passes, status="in_progress"):
    """Persist a Design row with passes stored, like a design run leaves it."""
    from praxis.db import get_session, save_design_pass

    session = get_session()
    row = Design(candidate_id=candidate_id, status=status)
    session.add(row)
    session.commit()
    design_id = row.id
    session.close()
    for pass_id, content in passes.items():
        save_design_pass(design_id, pass_id, content)
    return design_id


def _seed_blueprint(candidate_id, blueprint_md):
    from praxis.db import get_session

    session = get_session()
    session.add(
        Blueprint(
            candidate_id=candidate_id,
            feasibility_score=6.0,
            blueprint_md=blueprint_md,
        )
    )
    session.commit()
    session.close()


def _seed_plan_usage(candidate_id):
    """One design-stage row (must not count), one plan row, one cached row."""
    from praxis.db import get_session

    session = get_session()
    session.add(
        LLMUsage(
            model="groq/openai/gpt-oss-120b",
            stage="design",
            candidate_id=candidate_id,
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            cost_usd=0.01,
        )
    )
    session.add(
        LLMUsage(
            model="groq/openai/gpt-oss-120b",
            stage="plan",
            candidate_id=candidate_id,
            prompt_tokens=200,
            completion_tokens=50,
            total_tokens=250,
            cost_usd=0.02,
        )
    )
    session.add(
        LLMUsage(
            model="groq/openai/gpt-oss-120b",
            stage="plan",
            candidate_id=candidate_id,
            prompt_tokens=40,
            completion_tokens=0,
            total_tokens=40,
            cost_usd=0.0,
            cached=True,
        )
    )
    session.commit()
    session.close()


def _pack_completion(critic_flag_once=False):
    """Fake praxis.design.call_llm answering pack, critic, and regen prompts."""
    calls = []

    def fake(prompt, system=None, model=None, **kwargs):
        stage = kwargs.get("stage")
        calls.append((stage, prompt))
        if "Review it against the defect classes" in prompt:
            if (
                critic_flag_once
                and "SECTION UNDER REVIEW: 'Product Requirements'" in prompt
                and not getattr(fake, "flagged", False)
            ):
                fake.flagged = True
                return json.dumps({"defects": [_PACK_DEFECT]})
            return json.dumps({"defects": []})
        if "env.example" in prompt:
            return PACK_CONTENT["env_example"]
        for content in PACK_CONTENT.values():
            title = content.split("\n", 1)[0].lstrip("# ").strip()
            if f"start with its '## {title}'" in prompt:
                return content
        raise AssertionError(f"unexpected prompt: {prompt[:160]!r}")

    fake.calls = calls
    return fake


def _generate(planning_db, monkeypatch, *, fake=None, **kwargs):
    """Run generate_doc_pack against a faked design.call_llm."""
    import praxis.design as design_module

    fake = fake if fake is not None else _pack_completion()
    monkeypatch.setattr(design_module, "call_llm", fake)
    kwargs.setdefault("pace_seconds", 0.0)
    result = generate_doc_pack(
        _Candidate(planning_db),
        HardwareProfile(),
        **kwargs,
    )
    return result, fake


def _prompt_for(calls, title):
    """The single markdown-trailer prompt for one pack document's title."""
    return next(
        prompt for _, prompt in calls if f"start with its '## {title}'" in prompt
    )


# ---------------------------------------------------------------------------
# Spec-table consistency
# ---------------------------------------------------------------------------


def test_pack_specs_extend_the_pass_table():
    assert PASS_IDS + PACK_PASS_IDS == tuple(_PASS_SPECS)
    titles = [_PASS_SPECS[pass_id]["title"] for pass_id in PACK_PASS_IDS]
    assert len(titles) == len(set(titles))
    facts = FactsSheet()
    for pass_id in PACK_PASS_IDS:
        # _pass_instruction .format()s the instruction: a literal brace in a
        # spec would raise KeyError here instead of at generation time.
        assert _pass_instruction(pass_id, facts).strip()
        assert PACK_OUTPUT_TOKENS[pass_id] > 0
    assert RAW_FILE_PASSES == frozenset({"env_example"})


def test_pack_grounding_budgets_declared():
    for pass_id in PACK_PASS_IDS:
        assert pass_id in GROUNDING_CHAR_BUDGETS
    # the design engine's core budgets stay untouched
    assert GROUNDING_CHAR_BUDGETS["hardware_fit"] == 0
    assert GROUNDING_CHAR_BUDGETS["plan"] == 3000
    # blueprint/paper-reading pack documents get bounded context...
    assert GROUNDING_CHAR_BUDGETS["prd"] == 4000
    assert GROUNDING_CHAR_BUDGETS["test_plan"] == 2000
    assert GROUNDING_CHAR_BUDGETS["readme"] == 1000
    # ...the rules/security/env files derive from the design state only
    assert GROUNDING_CHAR_BUDGETS["rules"] == 0
    assert GROUNDING_CHAR_BUDGETS["security"] == 0
    assert GROUNDING_CHAR_BUDGETS["env_example"] == 0


# ---------------------------------------------------------------------------
# generate_doc_pack
# ---------------------------------------------------------------------------


def test_generate_doc_pack_runs_all_pack_passes(planning_db, no_grounding, monkeypatch):
    from praxis.db import get_session, latest_design

    design_id = _seed_design(planning_db, GOOD_PASSES, status="in_progress")
    result, fake = _generate(planning_db, monkeypatch)

    assert isinstance(result, PackResult)
    assert result.status == "complete"
    assert result.error is None
    assert result.completed_passes == list(PACK_PASS_IDS)
    assert result.critic_skip_note  # critic is off by default
    # Six calls on stage "plan", one per document, in PACK_PASS_IDS order.
    assert len(fake.calls) == 6
    for index, pass_id in enumerate(PACK_PASS_IDS):
        stage, prompt = fake.calls[index]
        assert stage == "plan"
        assert f"'{_PASS_SPECS[pass_id]['title']}'" in prompt
    # Same-row persistence: one Design row, status/design_md untouched,
    # core passes preserved alongside the stored pack documents.
    row = latest_design(planning_db)
    assert row is not None and row.id == design_id
    assert row.status == "in_progress"
    assert row.design_md == ""
    passes = json.loads(row.passes_json)
    for pass_id in PACK_PASS_IDS:
        assert passes[pass_id] == PACK_CONTENT[pass_id]
    for pass_id in PASS_IDS:
        assert passes[pass_id] == GOOD_PASSES[pass_id]
    session = get_session()
    rows = session.query(Design).filter(Design.candidate_id == planning_db).all()
    session.close()
    assert len(rows) == 1


def test_generate_doc_pack_requires_completed_design(
    planning_db, no_grounding, monkeypatch
):
    # No design at all: fail fast with the command to run, zero LLM calls.
    result, fake = _generate(planning_db, monkeypatch)
    assert result.status == "failed"
    assert result.design_id is None
    assert f"praxis design {planning_db}" in result.error
    assert fake.calls == []

    # A design with only some core passes stored: same refusal, names
    # --resume and lists exactly the missing passes.
    _seed_design(planning_db, {"technique": GOOD_PASSES["technique"]})
    result, fake = _generate(planning_db, monkeypatch)
    assert result.status == "failed"
    assert "architecture, data_contracts, plan, hardware_fit" in result.error
    assert f"praxis design {planning_db} --resume" in result.error
    assert fake.calls == []


def test_generate_doc_pack_resumes_stored_pack_passes(
    planning_db, no_grounding, monkeypatch
):
    from praxis.db import latest_design

    _seed_design(
        planning_db,
        {**GOOD_PASSES, "prd": PACK_CONTENT["prd"], "rules": PACK_CONTENT["rules"]},
    )
    result, fake = _generate(planning_db, monkeypatch)

    assert result.status == "complete"
    assert len(fake.calls) == 4  # the two stored pack documents are skipped
    expected = [p for p in PACK_PASS_IDS if p not in ("prd", "rules")]
    for index, pass_id in enumerate(expected):
        stage, prompt = fake.calls[index]
        assert stage == "plan"
        assert f"'{_PASS_SPECS[pass_id]['title']}'" in prompt
    passes = json.loads(latest_design(planning_db).passes_json)
    assert passes["prd"] == PACK_CONTENT["prd"]  # reused, not regenerated
    assert passes["rules"] == PACK_CONTENT["rules"]
    assert result.completed_passes == list(PACK_PASS_IDS)


def test_generate_doc_pack_rerun_passes(planning_db, no_grounding, monkeypatch):
    _seed_design(planning_db, {**GOOD_PASSES, **PACK_CONTENT})
    result, fake = _generate(planning_db, monkeypatch, rerun_passes=["rules"])

    assert result.status == "complete"
    assert len(fake.calls) == 1
    assert fake.calls[0][0] == "plan"
    assert "'Engineering Rules'" in fake.calls[0][1]

    # Core design passes and unknown ids are rejected before any LLM call.
    with pytest.raises(ValueError, match="unknown pack pass"):
        generate_doc_pack(
            _Candidate(planning_db),
            HardwareProfile(),
            pace_seconds=0.0,
            rerun_passes=["plan"],
        )
    with pytest.raises(ValueError, match="unknown pack pass"):
        generate_doc_pack(
            _Candidate(planning_db),
            HardwareProfile(),
            pace_seconds=0.0,
            rerun_passes=["nope"],
        )


def test_prd_prompt_carries_blueprint_excerpt(planning_db, no_grounding, monkeypatch):
    _seed_design(planning_db, GOOD_PASSES)
    _seed_blueprint(planning_db, _BLUEPRINT_MD)
    result, fake = _generate(planning_db, monkeypatch)

    assert result.status == "complete"
    prd_prompt = _prompt_for(fake.calls, "Product Requirements")
    assert "EARLIER BLUEPRINT" in prd_prompt
    assert "Prompt-injection defenses for local agent loops" in prd_prompt
    assert "Vector store (deferred)" in prd_prompt
    assert "3 days" in prd_prompt
    # Only the bounded sections ride along; the rest of the blueprint stays out.
    assert "Sanitizer component" not in prd_prompt
    assert "Phase 1 slice marker" not in prd_prompt


def test_prd_prompt_without_blueprint_uses_fallback(
    planning_db, no_grounding, monkeypatch
):
    _seed_design(planning_db, GOOD_PASSES)
    result, fake = _generate(planning_db, monkeypatch)

    assert result.status == "complete"
    prd_prompt = _prompt_for(fake.calls, "Product Requirements")
    assert "NO BLUEPRINT ON FILE" in prd_prompt
    assert "EARLIER BLUEPRINT" not in prd_prompt


def test_test_plan_prompt_includes_plan_details(planning_db, no_grounding, monkeypatch):
    _seed_design(planning_db, GOOD_PASSES)
    result, fake = _generate(planning_db, monkeypatch)

    assert result.status == "complete"
    test_plan_prompt = _prompt_for(fake.calls, "Test Plan")
    assert "PLAN DETAILS" in test_plan_prompt
    assert "**Tests:** pytest for Retriever and Trainer" in test_plan_prompt
    assert "Metric: MRR@10" in test_plan_prompt  # the ### Eval plan rides along
    readme_prompt = _prompt_for(fake.calls, "README Skeleton")
    assert "PLAN DETAILS" not in readme_prompt


def test_pack_trailers_markdown_vs_raw(planning_db, no_grounding, monkeypatch):
    _seed_design(planning_db, GOOD_PASSES)
    result, fake = _generate(planning_db, monkeypatch)

    assert result.status == "complete"
    env_prompt = next(p for _, p in fake.calls if "env.example" in p)
    assert "raw file content only" in env_prompt
    assert "no markdown fences" in env_prompt
    assert "start with its" not in env_prompt  # raw file: no heading demanded
    rules_prompt = _prompt_for(fake.calls, "Engineering Rules")
    assert "start with its '## Engineering Rules'" in rules_prompt


def test_pack_progress_labels_and_pacing(planning_db, no_grounding, monkeypatch):
    import praxis.planning as planning_module

    _seed_design(planning_db, GOOD_PASSES)

    labels: list[str] = []
    sleeps: list[float] = []

    def spy_wait(model, prompt, facts, *, output_tokens=None, progress_label=""):
        labels.append(progress_label)

    monkeypatch.setattr(planning_module, "_wait_for_tpm_budget", spy_wait)
    monkeypatch.setattr(planning_module, "_sleep", sleeps.append)

    result, fake = _generate(planning_db, monkeypatch, pace_seconds=1.0)

    assert result.status == "complete"
    assert len(fake.calls) == 6
    assert labels == [f"pack {i}/6" for i in range(1, 7)]
    # one pacing sleep between documents, never after the last
    assert sleeps == [1.0] * 5


def test_pack_critic_opt_in_targets_pack_sections(
    planning_db, no_grounding, monkeypatch
):
    _seed_design(planning_db, {**GOOD_PASSES, **PACK_CONTENT})
    fake = _pack_completion(critic_flag_once=True)
    result, _ = _generate(planning_db, monkeypatch, fake=fake, run_critic=True)

    assert result.status == "complete"
    assert result.critic_skip_note is None
    assert len(result.defects) == 1
    assert result.defects[0]["section"] == "Product Requirements"

    stages = [stage for stage, _ in fake.calls]
    # six critic calls (one per pack document) + one re-review of the prd
    assert stages.count("design_critic") == 7
    # the flagged document is regenerated exactly once (stage "design")
    assert stages.count("design") == 1
    regen_prompt = next(p for s, p in fake.calls if s == "design")
    assert "REQUIRED FIX" in regen_prompt
    assert "Label the origin of FR-001." in regen_prompt
    # core design passes are never reviewed by the pack critic
    critic_prompts = [p for s, p in fake.calls if s == "design_critic"]
    for pass_id in PASS_IDS:
        title = _PASS_SPECS[pass_id]["title"]
        assert not any(f"SECTION UNDER REVIEW: '{title}'" in p for p in critic_prompts)


def test_generate_doc_pack_reports_plan_stage_usage(
    planning_db, no_grounding, monkeypatch
):
    _seed_design(planning_db, GOOD_PASSES)
    _seed_plan_usage(planning_db)
    result, _ = _generate(planning_db, monkeypatch)

    assert result.status == "complete"
    # Only non-cached plan-stage rows count; design-stage rows never leak in.
    assert result.calls == 1
    assert result.total_tokens == 250
    assert result.cost_usd == pytest.approx(0.02)
