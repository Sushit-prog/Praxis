"""Writers for the design outputs: designs/<slug>/ + build-memory recording."""

from __future__ import annotations

import re
from pathlib import Path

from praxis.config import HardwareProfile
from praxis.db import BuildMemory, Candidate, Design, get_session
from praxis.design import (
    PASS_IDS,
    DesignResult,
    render_agent_prompt,
    render_tasks_md,
)

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

    (out_dir / "DESIGN.md").write_text(result.design_md or "", encoding="utf-8")
    passes = passes or {}
    (out_dir / "TASKS.md").write_text(render_tasks_md(passes), encoding="utf-8")
    (out_dir / "AGENT_PROMPT.md").write_text(
        render_agent_prompt(passes, profile, candidate), encoding="utf-8"
    )
    return out_dir


def render_partial_md(
    passes: dict[str, str],
    candidate: Candidate,
    *,
    error: str | None = None,
) -> str:
    """Render DESIGN.partial.md: every completed pass + the missing ones.

    Written when a design run fails or is incomplete so partial progress is
    inspectable on disk (and resumable via ``praxis design <id> --resume``).
    """
    title = (getattr(candidate, "title", "") or "Design").strip()
    completed = [p for p in PASS_IDS if (passes.get(p) or "").strip()]
    missing = [p for p in PASS_IDS if p not in completed]
    lines = [
        f"# Design (partial): {title}",
        "",
        f"_Candidate #{getattr(candidate, 'id', '?')} · "
        f"{getattr(candidate, 'url', '') or 'n/a'}_",
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
    path.write_text(render_partial_md(passes, candidate, error=error), encoding="utf-8")
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
