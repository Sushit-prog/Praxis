"""Writers for the design outputs: designs/<slug>/, the docs pack, build memory."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from praxis.config import FactsSheet, HardwareProfile, load_facts
from praxis.db import BuildMemory, Candidate, Design, get_session, latest_design
from praxis.design import (
    PASS_IDS,
    DesignResult,
    _extract_heading_section,
    assemble_design_md,
    render_agent_prompt,
    render_tasks_md,
    target_machine_summary,
)
from praxis.planning import PACK_PASS_IDS

DESIGNS_DIR = Path("designs")


def slugify(text: str) -> str:
    """Filesystem-safe slug for a design directory."""
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug[:48] or "design"


def design_dir(candidate_id: int, title: str, root: Path | None = None) -> Path:
    base = root or Path.cwd()
    return base / DESIGNS_DIR / f"{candidate_id:03d}-{slugify(title)}"


def write_design_files(
    result: DesignResult,
    candidate: Candidate,
    profile: HardwareProfile,
    *,
    passes: dict[str, str] | None = None,
    facts: FactsSheet | None = None,
    root: Path | None = None,
) -> Path:
    """Write DESIGN.md, TASKS.md, AGENT_PROMPT.md; return the design directory.

    AGENT_PROMPT.md and TASKS.md are rendered from the individual pass
    outputs; pass them via ``passes`` when the DesignResult predates the file
    layout (e.g. resumed designs loaded from the DB).
    """
    title = (getattr(candidate, "title", "") or f"candidate-{result.candidate_id}").strip()
    out_dir = design_dir(result.candidate_id, title, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    facts = facts or load_facts()

    (out_dir / "DESIGN.md").write_text(result.design_md or "", encoding="utf-8")
    passes = passes or {}
    (out_dir / "TASKS.md").write_text(render_tasks_md(passes, facts=facts), encoding="utf-8")
    (out_dir / "AGENT_PROMPT.md").write_text(
        render_agent_prompt(passes, profile, candidate, facts=facts), encoding="utf-8"
    )
    return out_dir


def render_partial_md(
    passes: dict[str, str],
    candidate: Candidate,
    *,
    facts: FactsSheet | None = None,
    error: str | None = None,
) -> str:
    """Render DESIGN.partial.md: every completed pass + the missing ones.

    Written when a design run fails or is incomplete so partial progress is
    inspectable on disk (and resumable via ``praxis design <id> --resume``).
    """
    title = (getattr(candidate, "title", "") or "Design").strip()
    facts = facts or FactsSheet()
    completed = [p for p in PASS_IDS if (passes.get(p) or "").strip()]
    missing = [p for p in PASS_IDS if p not in completed]
    lines = [
        f"# Design (partial): {title}",
        "",
        f"_Candidate #{getattr(candidate, 'id', '?')} · "
        f"{getattr(candidate, 'url', '') or 'n/a'}_",
        "",
        f"**Target machine:** {target_machine_summary(facts)}",
        "",
        f"**Incomplete design — {len(completed)} of {len(PASS_IDS)} passes done.**",
    ]
    if error:
        lines += ["", f"Failure: {error}"]
    if missing:
        lines += ["", "Missing passes: " + ", ".join(missing)]
        lines += ["", "Re-run `praxis design <id> --resume` to finish it."]
    lines += [""]
    for pass_id in PASS_IDS:
        content = (passes.get(pass_id) or "").strip()
        if not content:
            continue
        lines += [content, ""]
    return "\n".join(lines).strip() + "\n"


def write_partial_design(
    candidate: Candidate,
    passes: dict[str, str],
    *,
    facts: FactsSheet | None = None,
    error: str | None = None,
    root: Path | None = None,
) -> Path:
    """Write designs/<id>-<slug>/DESIGN.partial.md; return the file path."""
    title = (getattr(candidate, "title", "") or f"candidate-{getattr(candidate, 'id', 0)}").strip()
    out_dir = design_dir(
        getattr(candidate, "id", 0), title, root
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "DESIGN.partial.md"
    path.write_text(
        render_partial_md(passes, candidate, facts=facts, error=error), encoding="utf-8"
    )
    return path


def record_pick(candidate_id: int, technique: str, focus: str | None) -> None:
    """Record the discover pick in build_memory as non-authoritative context.

    Outcome says ``picked_for_design``; the Analyst treats build history as
    data about past human decisions, never as instructions. The focus note is
    appended to the technique text so the memory row is self-describing.
    """
    entry = (technique or "").strip()
    if focus:
        entry = f"{entry} | focus: {focus.strip()}".strip(" |")
    session = get_session()
    try:
        session.add(
            BuildMemory(
                candidate_id=candidate_id,
                technique=entry[:200],
                decision="approved",
                outcome="picked_for_design",
            )
        )
        session.commit()
    finally:
        session.close()


def load_passes(design: Design) -> dict[str, str]:
    """Pass outputs of a stored design (json in passes_json)."""
    import json

    try:
        data = json.loads(design.passes_json or "{}")
        return {k: str(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Planning docs pack (designs/<id>-<slug>/docs/): pure, deterministic writers
# ---------------------------------------------------------------------------


# The six LLM pack documents and their filenames inside docs/.
_PACK_FILENAMES = {
    "prd": "PRD.md",
    "rules": "RULES.md",
    "test_plan": "TEST_PLAN.md",
    "security": "SECURITY.md",
    "readme": "README.md",
    "env_example": ".env.example",
}

# Core passes the deterministic docs derive from (beyond the pack itself).
_DOC_CORE_PASSES = ("plan", "architecture", "hardware_fit")


def _stamp_timestamp(generated_at: datetime | None) -> str:
    """UTC timestamp for the generated-from stamp; naive datetimes are UTC."""
    when = generated_at if generated_at is not None else datetime.now(UTC)
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return when.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_stamped(path: Path, content: str, source: str, stamp_time: str) -> None:
    """Write a duplicated doc with its generated-from stamp as the first line."""
    path.write_text(
        f"<!-- generated-from: {source} · {stamp_time} -->\n\n{content.strip()}\n",
        encoding="utf-8",
    )


def _promote_headings(md: str) -> str:
    """Promote exact level-2 headings to level 1; '### ' subheadings untouched."""
    return re.sub(r"(?m)^## ", "# ", md)


def _strip_first_heading(section: str) -> str:
    lines = section.splitlines()
    if lines and lines[0].strip().startswith("### "):
        lines = lines[1:]
    return "\n".join(lines).strip()


def _plan_outline(plan: str) -> list[str]:
    """Phase headings and TASK checkbox lines from the plan pass (uncapped)."""
    return [
        ln.strip()
        for ln in plan.splitlines()
        if ln.strip().startswith("### Phase")
        or re.match(r"^- \[[ xX]\] TASK-\d+", ln.strip())
    ]


def _decision_sections(passes: dict[str, str]) -> tuple[str, str]:
    """(decision records, rejected alternatives) extracted from their passes."""
    records = _strip_first_heading(
        _extract_heading_section(passes.get("architecture", ""), "Decision records")
    )
    rejected = _strip_first_heading(
        _extract_heading_section(passes.get("hardware_fit", ""), "Rejected")
    )
    return records, rejected


def render_decisions_md(passes: dict[str, str]) -> str:
    """DECISIONS.md: deterministic extraction (architecture + hardware_fit)."""
    records, rejected = _decision_sections(passes)
    return "\n".join(
        [
            "# Decision Records",
            "",
            "_Deterministic extraction from the design's architecture pass "
            "(decision records) and hardware & budget fit pass (rejected "
            "alternatives)._",
            "",
            "## Decision records",
            "",
            records or "(none recorded in the architecture pass)",
            "",
            "## Rejected because it does not fit",
            "",
            rejected or "(none recorded in the hardware & budget fit pass)",
            "",
        ]
    )


def _build_memory_lines(candidate_id: int) -> list[str]:
    from sqlalchemy import select

    session = get_session()
    try:
        rows = session.scalars(
            select(BuildMemory)
            .where(BuildMemory.candidate_id == candidate_id)
            .order_by(BuildMemory.id)
        ).all()
    finally:
        session.close()
    lines = []
    for row in rows:
        when = row.created_at.strftime("%Y-%m-%d") if row.created_at else "?"
        lines.append(f"- {when} · {row.decision} · {row.outcome} · {row.technique}")
    return lines


def render_memory_md(
    candidate: Candidate, passes: dict[str, str], *, facts: FactsSheet | None = None
) -> str:
    """MEMORY.md seed: identity, machine, plan outline, decisions, history.

    The header instructs the coding agent to keep maintaining the file as it
    implements; everything below it is a one-shot deterministic seed.
    """
    facts = facts or load_facts()
    candidate_id = getattr(candidate, "id", None)
    title = (getattr(candidate, "title", "") or "untitled").strip()
    lines = [
        "# Project Memory",
        "",
        "> Maintained by the coding agent: append to this file as you work - "
        "decisions, deviations, bugs, and lessons learned - so later sessions "
        "keep the context. Seeded by Praxis from the design pack; the seed "
        "below is a starting point, not a boundary.",
        "",
        "## Candidate",
        f"- Candidate #{candidate_id} · {title}",
        f"- Source: {getattr(candidate, 'source', 'unknown')} · "
        f"{getattr(candidate, 'url', '') or 'n/a'}",
    ]
    technique = (getattr(candidate, "technique_summary", "") or "").strip()
    if technique:
        lines.append(f"- Technique: {technique}")
    score = getattr(candidate, "feasibility_score", None)
    if score is not None:
        lines.append(f"- Feasibility score: {score}")

    lines += [
        "",
        "## Target machine",
        target_machine_summary(facts),
        f"- Monthly budget: ${facts.monthly_budget_usd:.2f}",
        "",
        "## Phases & tasks",
    ]
    outline = _plan_outline(passes.get("plan", ""))
    lines += outline or ["(no phases or tasks found in the plan pass)"]

    records, _ = _decision_sections(passes)
    lines += ["", "## Decision records"]
    lines += [ln for ln in records.splitlines() if ln.strip()] or ["(none recorded)"]

    lines += ["", "## Build memory (human review history)"]
    memory = _build_memory_lines(candidate_id) if candidate_id is not None else []
    lines += memory or ["(no build-memory entries for this candidate)"]
    return "\n".join(lines) + "\n"


def write_docs_pack(
    candidate: Candidate,
    profile: HardwareProfile,
    *,
    passes: dict[str, str],
    facts: FactsSheet | None = None,
    root: Path | None = None,
    generated_at: datetime | None = None,
) -> Path:
    """Write the coding-agent docs pack into designs/<id>-<slug>/docs/.

        Eleven files: the five bare LLM pack documents verbatim, README.md and
        ARCHITECTURE.md with their title heading promoted to level 1 (so the
        heading level is code-guaranteed, not just prompt-requested), DESIGN.md
        and TASKS.md duplicated from the canonical top-level sources (each
        stamped with its source and generation time so staleness against those
        files is detectable), DECISIONS.md (deterministic extraction) and
        MEMORY.md (seed). Pure and deterministic: no LLM calls.

    DESIGN.md is a verbatim copy of the persisted ``design_md`` (byte-identical
    to the canonical top-level DESIGN.md); ``assemble_design_md`` only runs as
    the fallback when that row is still empty.
    """
    candidate_id = getattr(candidate, "id", None)
    if candidate_id is None:
        raise ValueError("candidate must be persisted (have an id) before writing docs")
    design = latest_design(candidate_id)
    if design is None:
        raise ValueError(
            f"candidate {candidate_id} has no design; run `praxis design {candidate_id}` first"
        )
    missing_core = [p for p in _DOC_CORE_PASSES if not (passes.get(p) or "").strip()]
    if missing_core:
        raise ValueError(
            "design incomplete (missing: "
            + ", ".join(missing_core)
            + f"); run `praxis design {candidate_id} --resume` first"
        )
    missing_pack = [p for p in PACK_PASS_IDS if not (passes.get(p) or "").strip()]
    if missing_pack:
        raise ValueError(
            f"docs pack passes missing ({', '.join(missing_pack)}); "
            f"run `praxis plan {candidate_id}` first"
        )

    facts = facts or load_facts()
    stamp_time = _stamp_timestamp(generated_at)
    title = (getattr(candidate, "title", "") or f"candidate-{candidate_id}").strip()
    out_dir = design_dir(candidate_id, title, root) / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)

    for pass_id, filename in _PACK_FILENAMES.items():
        if pass_id == "readme":
            continue
        (out_dir / filename).write_text(passes[pass_id], encoding="utf-8")

    design_md = (design.design_md or "").strip()
    if not design_md:
        design_md = assemble_design_md(passes, profile, candidate, facts=facts)
    _write_stamped(
        out_dir / "DESIGN.md",
        design_md,
        "design passes (assemble_design_md)",
        stamp_time,
    )
    _write_stamped(out_dir / "TASKS.md", passes["plan"], "plan pass", stamp_time)
    (out_dir / "ARCHITECTURE.md").write_text(
        _promote_headings(passes["architecture"]), encoding="utf-8"
    )
    (out_dir / "README.md").write_text(
        _promote_headings(passes["readme"]), encoding="utf-8"
    )
    (out_dir / "DECISIONS.md").write_text(render_decisions_md(passes), encoding="utf-8")
    (out_dir / "MEMORY.md").write_text(
        render_memory_md(candidate, passes, facts=facts), encoding="utf-8"
    )
    return out_dir
