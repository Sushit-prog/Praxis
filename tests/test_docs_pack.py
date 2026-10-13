"""Tests for the deterministic planning docs pack (design_io.write_docs_pack)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from praxis.config import HardwareProfile, load_facts
from praxis.db import BuildMemory, Design
from praxis.design import target_machine_summary
from praxis.design_io import render_decisions_md, write_docs_pack
from tests.test_design import GOOD_PASSES, _Candidate
from tests.test_planning import PACK_CONTENT

PASSES = {**GOOD_PASSES, **PACK_CONTENT}
FROZEN = datetime(2026, 9, 29, 12, 0, 0, tzinfo=UTC)

ALL_FILES = [
    ".env.example",
    "ARCHITECTURE.md",
    "DECISIONS.md",
    "DESIGN.md",
    "MEMORY.md",
    "PRD.md",
    "README.md",
    "RULES.md",
    "SECURITY.md",
    "TASKS.md",
    "TEST_PLAN.md",
]
# The core eleven plus the optional Recommendations document.
ALL_FILES_FULL = sorted([*ALL_FILES, "RECOMMENDATIONS.md"])
CORE_PASSES = {k: v for k, v in PASSES.items() if k != "recommendations"}


@pytest.fixture
def docs_db(db_engine, monkeypatch):
    """Fresh DB tables; get_session bound across the db/design/docs modules.

    Unlike conftest's db_session this also patches praxis.db itself, because
    latest_design (a db-module function) resolves get_session through
    praxis.db globals at call time.
    """
    import importlib

    from sqlalchemy.orm import Session

    def fresh_session():
        return Session(bind=db_engine, expire_on_commit=False)

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
    yield


def _seed_design(*, design_md: str = "# Design: CPU Fine-Tune\n\nbody\n"):
    from praxis.db import get_session

    session = get_session()
    try:
        row = Design(
            candidate_id=1, status="complete", passes_json="{}", design_md=design_md
        )
        session.add(row)
        session.commit()
        return row.id
    finally:
        session.close()


def _add_build_memory():
    from praxis.db import get_session

    session = get_session()
    try:
        session.add(
            BuildMemory(
                candidate_id=1,
                technique="LoRA fine-tuning on CPU | focus: keep it small",
                decision="approved",
                outcome="picked_for_design",
            )
        )
        session.commit()
    finally:
        session.close()


def _write(tmp_path, **kwargs):
    kwargs.setdefault("passes", PASSES)
    kwargs.setdefault("generated_at", FROZEN)
    return write_docs_pack(
        _Candidate(1), HardwareProfile(), root=tmp_path, **kwargs
    )


def test_write_docs_pack_writes_eleven_core_files(docs_db, tmp_path):
    """Without a stored Recommendations pass, exactly the core eleven land."""
    _seed_design()
    out = _write(tmp_path, passes=CORE_PASSES)

    assert out == tmp_path / "designs" / "001-cpu-fine-tune" / "docs"
    assert sorted(p.name for p in out.iterdir()) == ALL_FILES
    assert not (out / "RECOMMENDATIONS.md").exists()
    # The memory seed falls back to a placeholder when no history exists.
    assert "(no build-memory entries for this candidate)" in (
        out / "MEMORY.md"
    ).read_text(encoding="utf-8")


def test_write_docs_pack_writes_recommendations_when_stored(docs_db, tmp_path):
    """A stored Recommendations pass adds the twelfth file, verbatim."""
    _seed_design()
    out = _write(tmp_path)

    assert sorted(p.name for p in out.iterdir()) == ALL_FILES_FULL
    text = (out / "RECOMMENDATIONS.md").read_text(encoding="utf-8")
    assert text == PACK_CONTENT["recommendations"]
    assert text.startswith("# Recommendations\n")
    assert "## Fit summary\n" in text
    assert "no hardware conflicts detected (scan covers RAM" in text


def test_design_md_is_verbatim_with_stamp(docs_db, tmp_path):
    canonical = "# Design: CPU Fine-Tune\n\n## Technique\n\nbody\n"
    _seed_design(design_md=canonical)
    out = _write(tmp_path, generated_at=datetime(2026, 9, 29, 12, 0, 0))

    text = (out / "DESIGN.md").read_text(encoding="utf-8")
    # Naive datetimes are treated as UTC; the stamp is the file's first line.
    assert text == (
        "<!-- generated-from: design passes (assemble_design_md) · "
        "2026-09-29T12:00:00Z -->\n\n" + canonical.strip() + "\n"
    )


def test_design_md_falls_back_to_assembly(docs_db, tmp_path):
    _seed_design(design_md="")
    out = _write(tmp_path)

    text = (out / "DESIGN.md").read_text(encoding="utf-8")
    assert text.startswith("<!-- generated-from: design passes (assemble_design_md)")
    assert "# Design: CPU Fine-Tune" in text
    assert "## Hardware & Budget Fit" in text


def test_tasks_md_is_full_plan_text_with_stamp(docs_db, tmp_path):
    _seed_design()
    out = _write(tmp_path)

    text = (out / "TASKS.md").read_text(encoding="utf-8")
    assert text.startswith(
        "<!-- generated-from: plan pass · 2026-09-29T12:00:00Z -->"
    )
    body = text.split("-->\n\n", 1)[1]
    # Full plan prose: phases, per-phase tests, and the eval plan survive
    # (the top-level render_tasks_md drops all three for its checkbox list).
    assert body == GOOD_PASSES["plan"].strip() + "\n"
    assert "### Phase 2: Polish" in body
    assert "**Tests:** pytest for Retriever" in body


def test_architecture_heading_promotion(docs_db, tmp_path):
    _seed_design()
    out = _write(tmp_path)

    text = (out / "ARCHITECTURE.md").read_text(encoding="utf-8")
    expected = GOOD_PASSES["architecture"].replace(
        "## Architecture", "# Architecture", 1
    )
    assert text == expected
    assert text.startswith("# Architecture\n")
    assert not any(ln.startswith("## ") for ln in text.splitlines())
    assert "### Components" in text
    assert "```mermaid" in text


@pytest.mark.parametrize(
    "emitted", ["# README Skeleton\n\nbody\n", "## README Skeleton\n\nbody\n"]
)
def test_readme_heading_is_code_guaranteed(docs_db, tmp_path, emitted):
    _seed_design()
    out = _write(tmp_path, passes={**PASSES, "readme": emitted})

    text = (out / "README.md").read_text(encoding="utf-8")
    # Whatever heading level the model emitted, the pack ships it at level 1
    # (the same _promote_headings step ARCHITECTURE.md gets).
    assert text.startswith("# README Skeleton\n")
    assert not any(ln.startswith("## ") for ln in text.splitlines())


def test_decisions_md_extraction(docs_db, tmp_path):
    _seed_design()
    out = _write(tmp_path)

    text = (out / "DECISIONS.md").read_text(encoding="utf-8")
    assert text.startswith("# Decision Records")
    assert "- DR-1 retrieval backend" in text
    assert "Dense retriever with FAISS" in text
    assert "## Decision records" in text
    assert "## Rejected because it does not fit" in text
    # The extracted '###' heading lines are stripped, not duplicated.
    assert "### Decision records" not in text


def test_decisions_md_placeholders_when_sections_absent():
    passes = {
        "architecture": "## Architecture\n\n### Components\n- None.\n",
        "hardware_fit": "## Hardware & Budget Fit\n\n### Risks\n- None.\n",
    }
    text = render_decisions_md(passes)
    assert "(none recorded in the architecture pass)" in text
    assert "(none recorded in the hardware & budget fit pass)" in text


def test_memory_md_seed(docs_db, tmp_path):
    _add_build_memory()
    _seed_design()
    out = _write(tmp_path)

    text = (out / "MEMORY.md").read_text(encoding="utf-8")
    assert text.startswith("# Project Memory")
    assert "Maintained by the coding agent" in text
    assert "- Candidate #1 · CPU Fine-Tune" in text
    assert "https://arxiv.org/abs/2401.12345" in text
    facts = load_facts()
    assert target_machine_summary(facts) in text
    assert f"${facts.monthly_budget_usd:.2f}" in text
    assert "### Phase 1: Vertical slice" in text
    assert "- [ ] TASK-001" in text
    assert "DR-1 retrieval backend" in text
    assert "approved · picked_for_design" in text
    assert "focus: keep it small" in text


def test_stamp_only_on_design_and_tasks(docs_db, tmp_path):
    _seed_design()
    out = _write(tmp_path)

    for name in ALL_FILES_FULL:
        if name in ("DESIGN.md", "TASKS.md"):
            continue
        assert "generated-from" not in (out / name).read_text(encoding="utf-8")
    assert "generated-from" in (out / "DESIGN.md").read_text(encoding="utf-8")
    assert "generated-from" in (out / "TASKS.md").read_text(encoding="utf-8")


def test_write_docs_pack_requires_design(docs_db, tmp_path):
    with pytest.raises(ValueError, match="praxis design 1"):
        _write(tmp_path)


def test_write_docs_pack_requires_core_passes(docs_db, tmp_path):
    _seed_design()
    with pytest.raises(ValueError, match="--resume"):
        _write(tmp_path, passes=dict(PACK_CONTENT))


def test_write_docs_pack_requires_pack_passes(docs_db, tmp_path):
    _seed_design()
    partial = {k: v for k, v in PASSES.items() if k != "prd"}
    with pytest.raises(ValueError, match="praxis plan 1"):
        _write(tmp_path, passes=partial)


# ---------------------------------------------------------------------------
# Provenance renders (v0.2): no render-time clocks, MEMORY status fields
# ---------------------------------------------------------------------------


def test_memory_md_has_status_task_and_next_step(docs_db, tmp_path):
    _seed_design()
    out = _write(tmp_path)

    text = (out / "MEMORY.md").read_text(encoding="utf-8")
    assert "## Status\n- Status: planning complete" in text
    assert "## Current task\n- (none yet — pick TASK-001)" in text
    assert "## Next step\n- Run the handoff loop" in text
    # Target repos do not have Praxis: the seed never tells the coding
    # agent to run a praxis command.
    assert "run the task's test" in text
    assert "praxis check" not in text


def test_decisions_md_header_date_from_generated_at():
    passes = {
        "architecture": "## Architecture\n\n### Decision records\n"
        "- DR-1 x — options: a, b; chose: a; why: small; revisit when: big.\n",
        "hardware_fit": "## Hardware & Budget Fit\n\n### Rejected\n- none.\n",
    }
    text = render_decisions_md(passes, generated_at=FROZEN)

    assert text.startswith("# Decision Records\n")
    assert "- Date: 2026-09-29" in text
    assert "- Status: active" in text
    assert "- Source: architecture pass (decision records)" in text
    assert "- DR-1 x" in text


def test_decisions_md_unstamped_without_generated_at():
    passes = {
        "architecture": "## Architecture\n\n### Decision records\n- DR-1 x.\n",
        "hardware_fit": "## Hardware & Budget Fit\n\n### Rejected\n- none.\n",
    }
    text = render_decisions_md(passes)

    assert "- Date: unstamped legacy pack" in text


def test_write_docs_pack_byte_stable_without_generated_at(docs_db, tmp_path):
    """Legacy packs (no _meta): two renders differ in no byte."""
    _seed_design()
    out = _write(tmp_path, generated_at=None)
    first = {name: (out / name).read_bytes() for name in ALL_FILES_FULL}

    out = _write(tmp_path, generated_at=None)
    second = {name: (out / name).read_bytes() for name in ALL_FILES_FULL}

    assert first == second
    # And the legacy stamp literal replaces the old render-time clock.
    assert b"unstamped" in (out / "DESIGN.md").read_bytes()


def test_recommendations_md_byte_stable_empty_state(docs_db, tmp_path):
    """The pinned empty-state Recommendations render is byte-stable."""
    _seed_design()
    out = _write(tmp_path)
    first = (out / "RECOMMENDATIONS.md").read_bytes()

    out = _write(tmp_path)
    second = (out / "RECOMMENDATIONS.md").read_bytes()

    assert first == second
    text = first.decode("utf-8").replace("\r\n", "\n")
    assert (
        "no hardware conflicts detected (scan covers RAM, GPU/CUDA and "
        "remote-host claims); 42 hardware-related statements checked"
    ) in text
    assert "## Risks\nnone" in text
    assert "no substitution found" in text
    assert "scan found no hardware-driven cuts" in text


def test_parse_generated_at_roundtrip():
    from praxis.design_io import parse_generated_at

    parsed = parse_generated_at("2026-10-12T08:30:00Z")
    assert parsed == datetime(2026, 10, 12, 8, 30, 0, tzinfo=UTC)
    assert parse_generated_at(None) is None
    assert parse_generated_at("") is None
    assert parse_generated_at("not-a-timestamp") is None
    assert parse_generated_at(123) is None
