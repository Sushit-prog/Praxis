"""Tests for the multi-pass design generator and design-doc rubric."""

from __future__ import annotations

import json

import pytest

from praxis.config import HardwareProfile
from praxis.db import Candidate, Design
from praxis.design import (
    PASS_IDS,
    assemble_design_md,
    build_hardware_fit_anchor,
    generate_design,
    render_agent_prompt,
    render_tasks_md,
    resolve_design_model,
)
from praxis.eval import run_design_rubric

GOOD_PASSES = {
    "technique": (
        "## Technique\n\n"
        "### Inputs\n- Documents (str) [source 1].\n\n"
        "### Outputs\n- Chunk vectors (list[float], dim 384) [source 1].\n\n"
        "### Algorithm\n1. Split into 512-token windows [source 1].\n"
        "2. Score each window with the cross-encoder (inference).\n\n"
        "### Evaluation setup\n- MS MARCO, MRR@10 = 0.33 [source 1].\n\n"
        "### Section citations\n- [source 1] = §3 Method.\n"
    ),
    "architecture": (
        "## Architecture\n\n"
        "### Components\n"
        "- Retriever — fetches training chunks [source 1].\n"
        "- Trainer — runs the LoRA loop (inference).\n\n"
        "### Data flow\nRetriever feeds Trainer (inference).\n\n"
        "### Interfaces\n- `train(cfg)` (inference).\n\n"
        "### Decision records\n"
        "- DR-1 retrieval backend — options: BM25, dense; chose: BM25; why: "
        "0 MB RAM; revisit when: corpus > 100k docs.\n\n"
        "```mermaid\ngraph TD\n  A[Retriever] --> B[Trainer]\n```\n"
    ),
    "data_contracts": (
        "## Data Model & Contracts\n\n"
        "### Data model\n"
        "```json\n{\"pairs\": [{\"prompt\": \"str\", \"completion\": \"str\"}]}\n```\n\n"
        "### Contracts\n- `train(cfg: TrainConfig) -> None` (inference).\n\n"
        "### File tree\n```text\npraxis/\n  train.py\n```\n\n"
        "### Core algorithm (pseudocode)\n```text\n"
        "window_size = 512  # tokens, default\n"
        "top_k = 20\n```\n"
    ),
    "plan": (
        "## Phased Implementation Plan\n\n"
        "### Phase 1: Vertical slice\n"
        "- [ ] TASK-001 Build Retriever fetch (acceptance: returns 10 chunks for a sample corpus; "
        "test: test_retriever_returns_chunks)\n"
        "- [ ] TASK-002 Build Trainer loop for Retriever output "
        "(acceptance: loss decreases on a toy run; test: test_trainer_learns)\n"
        "**Tests:** pytest for Retriever and Trainer; eval on a 20-prompt set.\n\n"
        "### Phase 2: Polish\n"
        "- [ ] TASK-003 Add CLI entry point (acceptance: `train --help` exits 0; "
        "test: test_cli_help)\n"
        "**Tests:** smoke test of the CLI; retention eval.\n\n"
        "### Eval plan\n- Metric: MRR@10; baseline: BM25 = 0.30 [source 1]; "
        "target: >= 0.33 (inference).\n"
    ),
    "hardware_fit": (
        "## Hardware & Budget Fit\n\n"
        "### Per-component RAM/CPU/$ table\n\n"
        "| Component | RAM (GB) | CPU (threads) | $/month |\n"
        "|---|---|---|---|\n"
        "| Retriever | 0.2 | 1 | $0 |\n"
        "| Trainer | 3.0 | 4 | $0 |\n"
        "| **Total** | **3.2** | | **$0** |\n\n"
        "### Rejected because it does not fit\n"
        "- Dense retriever with FAISS — rejected: not used because no GPU available "
        "and CUDA-only libraries are on the avoid list.\n\n"
        "### Degradation plan\n- Shrink window to 256 tokens first.\n\n"
        "### Risks\n- Slow epochs; mitigation: small corpora [source 1].\n\n"
        "### Cuts\n- Drop Trainer batch size first (inference).\n"
    ),
}


class _Candidate:
    """Minimal stand-in for a persisted Candidate row."""

    def __init__(self, cid=1):
        self.id = cid
        self.source = "arxiv"
        self.url = "https://arxiv.org/abs/2401.12345"
        self.title = "CPU Fine-Tune"
        self.raw_text = "Fine-tune a small transformer on CPU."
        self.technique_summary = "LoRA fine-tuning on CPU"
        self.feasibility_score = 8


# ---------------------------------------------------------------------------
# Hardware-fit section and model resolution
# ---------------------------------------------------------------------------


def test_hardware_fit_anchor_contains_profile_values(hardware_profile):
    from praxis.config import load_facts

    md = build_hardware_fit_anchor(load_facts(), hardware_profile)
    assert "### Hard constraints (from hardware_profile.yaml)" in md
    assert f"{hardware_profile.ram_gb} GB" in md
    assert f"${hardware_profile.monthly_budget_usd:.2f}" in md
    assert "Windows" in md
    assert "Rate limits" in md
    assert "Total rule" in md


def test_resolve_design_model_precedence(monkeypatch):
    monkeypatch.delenv("PRAXIS_DESIGN_MODEL", raising=False)
    assert resolve_design_model() == "groq/openai/gpt-oss-120b"
    monkeypatch.setenv("PRAXIS_DESIGN_MODEL", "  groq/test-model  ")
    assert resolve_design_model() == "groq/test-model"
    assert resolve_design_model("groq/explicit") == "groq/explicit"


# ---------------------------------------------------------------------------
# generate_design with a mocked LLM
# ---------------------------------------------------------------------------


@pytest.fixture
def design_db(db_engine, monkeypatch):
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
        )
    ]
    for module in modules:
        monkeypatch.setattr(module, "get_session", fresh_session)
    yield candidate_id
    session.close()


@pytest.fixture
def no_grounding(monkeypatch):
    """Keep the design tests offline: no arXiv fetch."""
    import praxis.design as design_module
    import praxis.grounding as grounding_module

    def fake_ground(candidate):
        return grounding_module.Grounding(chunks=[], provenance=[], source_kind="none")

    monkeypatch.setattr(grounding_module, "ground_candidate", fake_ground)
    monkeypatch.setattr(design_module, "ground_candidate", fake_ground)


def _completion_returning_passes(passes, critic_defects=None):
    """A fake call_llm that answers each design pass with its canned text.

    Returns the extracted response text (design.py consumes call_llm output
    directly), matching each pass by the section title in its prompt.
    """
    calls = []

    def fake_call_llm(prompt, system=None, model=None, **kwargs):
        calls.append(prompt)
        for content in passes.values():
            title = content.split("\n", 1)[0].lstrip("# ").strip()
            if f"start with its '## {title}'" in prompt:
                return content
        if "Review it against the defect classes" in prompt:
            return json.dumps({"defects": critic_defects or []})
        raise AssertionError(f"unexpected prompt: {prompt[:200]!r}")

    fake_call_llm.calls = calls
    return fake_call_llm


def test_generate_design_runs_all_passes_and_critic(design_db, no_grounding, monkeypatch):
    import praxis.design as design_module

    seen_models = []
    seen_stages = []

    def fake_call_llm(prompt, system=None, model=None, **kwargs):
        seen_models.append(model)
        seen_stages.append(kwargs.get("stage"))
        if "Review it against the defect classes" in prompt:
            return json.dumps({"defects": []})
        for content in GOOD_PASSES.values():
            title = content.split("\n", 1)[0].lstrip("# ").strip()
            if f"start with its '## {title}'" in prompt:
                return content
        raise AssertionError(f"unexpected prompt: {prompt[:120]!r}")

    monkeypatch.setattr(design_module, "call_llm", fake_call_llm)
    monkeypatch.setattr(design_module, "PASS_PACING_S", 0.0)

    result = generate_design(
        _Candidate(design_db), HardwareProfile(), pace_seconds=0.0
    )

    assert result.status == "complete"
    assert set(result.completed_passes) == set(PASS_IDS)
    assert result.design_md.startswith("# Design: CPU Fine-Tune")
    assert "## Hardware & Budget Fit" in result.design_md
    assert "### Hard constraints (from hardware_profile.yaml)" in result.design_md
    assert "Critic pass completed with no defects." in result.design_md
    # 5 content passes + 1 critic, all on the resolved default design model
    assert len(seen_models) == 6
    assert set(seen_models) == {"groq/openai/gpt-oss-120b"}
    assert seen_stages.count("design") == 5
    assert seen_stages.count("design_critic") == 1


def test_generate_design_regenerates_flagged_sections(design_db, no_grounding, monkeypatch):
    import praxis.design as design_module

    critic_responses = iter(
        [
            json.dumps(
                {
                    "defects": [
                        {
                            "class": "UNCOVERED_COMPONENT",
                            "section": "Architecture",
                            "defect": "Retriever has no task.",
                            "fix": "Add a Retriever task to Phase 1.",
                        }
                    ]
                }
            ),
            json.dumps({"defects": []}),
        ]
    )

    def fake_call_llm(prompt, system=None, model=None, **kwargs):
        if "Review it against the defect classes" in prompt:
            return next(critic_responses)
        for content in GOOD_PASSES.values():
            title = content.split("\n", 1)[0].lstrip("# ").strip()
            if f"start with its '## {title}'" in prompt:
                return content
        raise AssertionError(f"unexpected prompt: {prompt[:120]!r}")

    monkeypatch.setattr(design_module, "call_llm", fake_call_llm)
    monkeypatch.setattr(design_module, "PASS_PACING_S", 0.0)

    result = generate_design(_Candidate(design_db), HardwareProfile(), pace_seconds=0.0)

    assert result.status == "complete"
    assert len(result.defects) == 1
    assert result.defects[0]["class"] == "UNCOVERED_COMPONENT"
    md = result.design_md
    assert "## Critic review" in md
    assert "[UNCOVERED_COMPONENT] Architecture" in md
    # 5 passes + critic + 1 regeneration + second critic
    assert md.count("## Critic review") == 1


def test_generate_design_failure_mid_pass_is_resumable(design_db, no_grounding, monkeypatch):
    import praxis.design as design_module

    state = {"n": 0}

    def flaky_call_llm(prompt, system=None, model=None, **kwargs):
        state["n"] += 1
        if state["n"] == 3:  # fail on the third pass call
            raise RuntimeError("provider 500")
        for content in GOOD_PASSES.values():
            title = content.split("\n", 1)[0].lstrip("# ").strip()
            if f"start with its '## {title}'" in prompt:
                return content
        if "Review it against the defect classes" in prompt:
            return json.dumps({"defects": []})
        raise AssertionError(f"unexpected prompt: {prompt[:120]!r}")

    monkeypatch.setattr(design_module, "call_llm", flaky_call_llm)
    monkeypatch.setattr(design_module, "PASS_PACING_S", 0.0)

    first = generate_design(_Candidate(design_db), HardwareProfile(), pace_seconds=0.0)
    assert first.status == "failed"
    assert first.completed_passes == ["technique", "architecture"]
    assert first.error == "provider 500"

    # Partial progress is persisted; a resumed run redoes only missing passes.

    from praxis.db import latest_design

    stored = latest_design(design_db)
    assert stored is not None
    assert stored.status == "in_progress"
    done = json.loads(stored.passes_json)
    assert set(done) == {"technique", "architecture"}

    monkeypatch.setattr(design_module, "call_llm", _completion_returning_passes(GOOD_PASSES))
    second = generate_design(_Candidate(design_db), HardwareProfile(), pace_seconds=0.0)
    assert second.status == "complete"
    assert set(second.completed_passes) == set(PASS_IDS)
    # The previously completed passes were reused, not regenerated.
    assert json.loads(latest_design(design_db).passes_json)["technique"] == done["technique"]


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------


def test_write_design_files_and_record_pick(design_db, no_grounding, tmp_path, monkeypatch):
    from praxis.db import BuildMemory, latest_design
    from praxis.design import DesignResult
    from praxis.design_io import write_design_files

    result = DesignResult(
        candidate_id=design_db,
        design_id=latest_design(design_db).id if latest_design(design_db) else None,
        status="complete",
        completed_passes=list(PASS_IDS),
        design_md=assemble_design_md(GOOD_PASSES, HardwareProfile(), _Candidate(design_db)),
    )
    # Ensure a Design row exists to load passes from.
    from praxis.db import get_session, save_design_pass

    session = get_session()
    row = Design(candidate_id=design_db, status="complete")
    session.add(row)
    session.commit()
    design_id = row.id
    session.close()
    for pass_id, content in GOOD_PASSES.items():
        save_design_pass(design_id, pass_id, content)

    out_dir = write_design_files(
        result, _Candidate(design_db), HardwareProfile(), passes=GOOD_PASSES, root=tmp_path
    )
    design_md = (out_dir / "DESIGN.md").read_text(encoding="utf-8")
    tasks_md = (out_dir / "TASKS.md").read_text(encoding="utf-8")
    agent_md = (out_dir / "AGENT_PROMPT.md").read_text(encoding="utf-8")

    assert "## Hardware & Budget Fit" in design_md
    # TASKS.md is a checkbox list of TASK-id tasks (not the full plan prose).
    assert "- [ ] TASK-001 Build Retriever fetch" in tasks_md
    assert "Coding agent prompt" in agent_md
    assert "Acceptance checks" in agent_md
    assert "Phase 1 ONLY" in agent_md

    # record_pick lands in build_memory with the focus note.
    from praxis.design_io import record_pick

    record_pick(design_db, "LoRA on CPU", "focus on data loading")
    session = get_session()
    memories = session.query(BuildMemory).all()
    session.close()
    assert len(memories) == 1
    assert "focus on data loading" in memories[0].technique
    assert memories[0].outcome == "picked_for_design"


def test_render_agent_prompt_scopes_to_phase1(hardware_profile):
    md = render_agent_prompt(GOOD_PASSES, hardware_profile, _Candidate(1))
    assert "implement Phase 1 ONLY" in md
    assert "### Phase 2" not in md  # later phases excluded from the paste-ready prompt
    assert f"{hardware_profile.ram_gb} GB" in md


def test_render_tasks_md_missing_plan():
    assert "plan pass missing" in render_tasks_md({})


# ---------------------------------------------------------------------------
# Design-doc rubric
# ---------------------------------------------------------------------------


def test_design_rubric_passes_good_doc(hardware_profile):
    md = assemble_design_md(GOOD_PASSES, hardware_profile, _Candidate(1))
    checks = run_design_rubric(md, hardware_profile)
    failed = [c for c in checks if not c.passed]
    assert failed == [], f"unexpected rubric failures: {[(c.name, c.detail) for c in failed]}"
    names = {c.name for c in checks}
    assert {
        "design_sections",
        "design_hardware_constraints",
        "budget_table_totals",
        "mermaid_parses",
        "claims_labelled",
        "components_map_to_tasks",
        "phase_acceptance",
    } <= names


def test_design_rubric_flags_missing_sections_and_unlabelled_claims(hardware_profile):
    md = (
        "# Design: X\n\n"
        "## Architecture\n\nNo components at all.\n\n"
        "## Phased Implementation Plan\n\nJust prose, no phases.\n"
    )
    checks = run_design_rubric(md, hardware_profile)
    by_name = {c.name: c for c in checks}
    assert by_name["design_sections"].passed is False
    assert by_name["mermaid_parses"].passed is False
    assert by_name["claims_labelled"].passed is False
    assert by_name["phase_acceptance"].passed is False
    assert by_name["design_hardware_constraints"].passed is False


def test_design_rubric_mermaid_declaration_required(hardware_profile):
    passes = dict(GOOD_PASSES)
    passes["architecture"] = passes["architecture"].replace("graph TD", "diagram thing")
    md = assemble_design_md(passes, hardware_profile, _Candidate(1))
    by_name = {c.name: c for c in run_design_rubric(md, hardware_profile)}
    assert by_name["mermaid_parses"].passed is False


# ---------------------------------------------------------------------------
# Item 4: pass structure (technique, decision records, TASK ids, hardware fit)
# ---------------------------------------------------------------------------


def test_technique_pass_requires_paper_vs_inference_separation(hardware_profile):
    """The technique pass instruction keeps 'from the paper' and 'inference' apart."""
    import praxis.design as design_module

    instruction = design_module._PASS_SPECS["technique"]["instruction"]
    assert "[source" in instruction and "inference" in instruction
    assert "Evaluation setup" in instruction.replace("evaluation setup", "Evaluation setup")


def test_pass_prompts_carry_section_citation_requirement(design_db, no_grounding, monkeypatch):
    """Every pass prompt mentions citation and inference labelling rules."""
    import praxis.design as design_module

    prompts = []

    def fake_call_llm(prompt, system=None, model=None, **kwargs):
        prompts.append(prompt)
        for content in GOOD_PASSES.values():
            title = content.split("\n", 1)[0].lstrip("# ").strip()
            if f"start with its '## {title}'" in prompt:
                return content
        if "Review it against the defect classes" in prompt:
            return json.dumps({"defects": []})
        raise AssertionError(f"unexpected prompt: {prompt[:120]!r}")

    monkeypatch.setattr(design_module, "call_llm", fake_call_llm)
    generate_design(_Candidate(design_db), HardwareProfile(), pace_seconds=0.0)

    assert len(prompts) == 6
    for prompt in prompts[:-1]:  # content passes, not the critic
        assert "UNVERIFIED" in prompt  # the labelling rule is in every pass prompt


def test_architecture_pass_demands_decision_records():
    """The architecture instruction requires options/choice/why/revisit records."""
    import praxis.design as design_module

    instruction = design_module._PASS_SPECS["architecture"]["instruction"]
    assert "Decision records" in instruction
    assert "options" in instruction and "revisit" in instruction


def test_plan_pass_demands_vertical_slice_and_task_ids():
    """The plan instruction requires the slice-first structure and TASK ids."""
    import praxis.design as design_module

    instruction = design_module._PASS_SPECS["plan"]["instruction"]
    assert "vertical slice" in instruction
    assert "TASK-001" in instruction
    assert "Eval plan" in instruction


def test_render_tasks_md_normalizes_unnumbered_tasks():
    """TASKS.md keeps TASK ids from the plan and ids any bare checkbox tasks."""
    plan = (
        "## Phased Implementation Plan\n\n"
        "### Phase 1: Slice\n"
        "- [ ] TASK-001 Already numbered (acceptance: done; test: test_a)\n"
        "- [ ] Bare task (acceptance: done; test: test_b)\n"
    )
    md = render_tasks_md({"plan": plan})
    assert "- [ ] TASK-001 Already numbered" in md
    assert "- [ ] TASK-002 Bare task" in md


def test_render_tasks_md_no_tasks():
    assert "no checkbox tasks" in render_tasks_md({"plan": "## Plan\n\nJust prose."})


def test_hardware_fit_pass_demand_quantified_table():
    """The hardware-fit instruction asks for a per-component table vs limits."""
    import praxis.design as design_module

    instruction = design_module._PASS_SPECS["hardware_fit"]["instruction"]
    assert "RAM/CPU/$ table" in instruction
    assert "Rejected because it does not fit" in instruction
    assert "Degradation plan" in instruction


def test_assembled_hardware_fit_keeps_anchor_after_pass_content(hardware_profile):
    """The deterministic anchor is appended inside the hardware_fit section."""
    from praxis.config import load_facts

    md = assemble_design_md(GOOD_PASSES, hardware_profile, _Candidate(1), facts=load_facts())
    fit = md.split("## Hardware & Budget Fit", 1)[1]
    assert "### Per-component RAM/CPU/$ table" in fit  # from the pass
    assert "### Hard constraints (from hardware_profile.yaml)" in fit  # anchor
    assert "### Windows-specific pitfalls" in fit  # anchor
    # The anchor lands before the next pass heading, inside the section.
    assert fit.index("Hard constraints") < fit.index("## Critic review")
