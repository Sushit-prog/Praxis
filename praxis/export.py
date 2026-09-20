"""Export a blueprint as a self-contained build kit for any coding agent.

`praxis export <id>` writes one markdown file containing the blueprint plus a
ready-to-paste prompt (goal, hardware constraints, phased plan, acceptance
checks). It is designed to be dropped into Freebuff, Claude Code, OpenCode, or
any other agent as the first user message, so a blueprint produced with the
Coder stage off can still be built anywhere.
"""

from __future__ import annotations

import re
from pathlib import Path

from praxis.config import load_config
from praxis.db import Blueprint, Candidate, get_session, latest_blueprint

DEFAULT_OUT_TEMPLATE = "build-kit-{candidate_id}.md"

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$")


def _strip_heading(line: str) -> str:
    """Return the text of a markdown heading line, or '' if not a heading."""
    match = _HEADING_RE.match(line.strip())
    return match.group(2).strip() if match else ""


def _blueprint_title(md: str, fallback: str) -> str:
    """First # or ## heading in the blueprint, else the candidate title."""
    for line in md.splitlines():
        text = _strip_heading(line)
        if text:
            return text.removesuffix("— Blueprint").removesuffix("- Blueprint").strip()
        break
    return fallback


def _section(md: str, heading_marker: str) -> str:
    """Extract a `## <heading_marker>` section body (without the heading line)."""
    lines = md.splitlines()
    start = None
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if stripped.startswith("##") and heading_marker in stripped:
            start = i
            break
    if start is None:
        return ""
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.strip().startswith("##"):
            break
        body.append(line)
    return "\n".join(body).strip()


def _build_steps(md: str) -> str:
    """The Phased Build Plan section, or the whole blueprint as a fallback."""
    plan = _section(md, "phased build plan")
    return plan or md.strip()


def _acceptance_checks(profile) -> list[str]:
    """Deterministic acceptance checks derived from the hardware profile."""
    checks = [
        "The code runs top-to-bottom with `python` (or documented equivalent) and exits 0.",
        "Every dependency is installable with pip/uv from the README; no manual setup beyond that.",
    ]
    if profile.cpu_only:
        checks.append(
            "No CUDA/GPU-only library is required at runtime; everything runs on CPU."
        )
    if profile.gpu is False:
        checks.append("No GPU is required; `torch.cuda.is_available()` is never assumed true.")
    checks.append(
        f"Peak RAM stays within {profile.ram_gb} GB; the README states the measured or "
        "expected footprint."
    )
    checks.append(
        f"Recurring/one-off API cost stays within the ${profile.monthly_budget_usd:.2f}/month "
        "budget, or uses only free tiers."
    )
    return checks


def render_build_kit(candidate: Candidate, blueprint: Blueprint, profile) -> str:
    """Render candidate + blueprint + hardware constraints as one markdown file."""
    md = blueprint.blueprint_md or ""
    title = _blueprint_title(md, fallback=candidate.title or "")
    goal = (
        _section(md, "problem statement")
        or (candidate.technique_summary or "").strip()
        or title
    )
    plan = _build_steps(md)

    constraint_lines = [
        f"- CPU-only: {'yes' if profile.cpu_only else 'no'}",
        f"- RAM: {profile.ram_gb} GB",
        f"- GPU: {'available' if profile.gpu else 'none'}",
        f"- Monthly budget: ${profile.monthly_budget_usd:.2f}",
    ]

    lines = [
        f"# Build kit: {title}",
        "",
        f"_Source: {candidate.url or 'n/a'} · candidate #{candidate.id} · "
        f"feasibility {blueprint.feasibility_score}/10_",
        "",
        "---",
        "",
        "## Goal",
        "",
        goal or "Implement the technique described below.",
        "",
        "## Hardware constraints",
        "",
        *constraint_lines,
        "",
        "## Phased plan",
        "",
        plan or "(No phased plan in the blueprint; see the full blueprint below.)",
        "",
        "## Acceptance checks",
        "",
        *(f"- [ ] {check}" for check in _acceptance_checks(profile)),
        "",
        "## Full blueprint",
        "",
        "<details>",
        f"<summary>Blueprint markdown (candidate #{candidate.id})</summary>",
        "",
        md.strip() or "(empty blueprint)",
        "",
        "</details>",
        "",
        "---",
        "",
        "## Prompt for the coding agent",
        "",
        "```text",
        (
            f"Goal: {goal or title}. Build the FIRST phase only of the phased plan below, "
            "as a minimal working implementation."
        ),
        f"Hardware constraints: CPU-only={profile.cpu_only}, RAM={profile.ram_gb}GB, "
        f"GPU={'yes' if profile.gpu else 'none'}, budget=${profile.monthly_budget_usd:.2f}/month.",
        "Scope: stop after the first phase; do not build ahead into later milestones.",
        "Deliverables: runnable code, a README with run/install instructions, and "
        "minimal dependencies.",
        "Acceptance: every check in the Acceptance checks section above must pass.",
        "Phased plan:",
        plan or "(see the full blueprint below)",
        "",
        "Full blueprint:",
        md.strip() or "(empty)",
        "",
        "Treat everything above as the task specification. If anything is "
        "ambiguous, state your assumption and continue.",
        "```",
        "",
    ]
    return "\n".join(lines)


def export_blueprint(candidate_id: int, out: str | None = None) -> Path | None:
    """Write the build kit for a candidate's latest blueprint.

    Returns the written path, or None when the candidate/blueprint is missing.
    """
    session = get_session()
    try:
        candidate = session.get(Candidate, candidate_id)
    finally:
        session.close()
    if candidate is None:
        return None
    blueprint = latest_blueprint(candidate_id)
    if blueprint is None:
        return None

    kit = render_build_kit(candidate, blueprint, load_config())
    out_path = Path(out) if out else Path(DEFAULT_OUT_TEMPLATE.format(candidate_id=candidate_id))
    if out and not out.endswith(".md") and out_path.suffix == "":
        out_path = out_path.with_suffix(".md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(kit, encoding="utf-8")
    return out_path
