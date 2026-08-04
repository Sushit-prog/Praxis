"""Coder: turn a blueprint's first phase into a draft prototype via OpenCode CLI."""

from __future__ import annotations

import logging
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

from praxis.db import Blueprint, Candidate, get_session

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 600
TIMEOUT_ENV = "PRAXIS_CODER_TIMEOUT_S"
SCRATCH_ROOT_ENV = "PRAXIS_SCRATCH_ROOT"
DEFAULT_SCRATCH_ROOT = Path("./scratch")

PHASED_PLAN_HEADING = "phased build plan"
MILESTONE_RE = re.compile(r"^\s*\d+[.)]\s")
SUBITEM_RE = re.compile(r"^\s*[-*]\s")


def _resolve_timeout(timeout: float | None) -> float:
    if timeout is not None:
        return timeout
    raw = os.environ.get(TIMEOUT_ENV)
    if raw is not None:
        try:
            return float(raw)
        except ValueError:
            logger.warning("invalid %s=%r; using default", TIMEOUT_ENV, raw)
    return DEFAULT_TIMEOUT_S


def _resolve_scratch_root(scratch_root: Path | None) -> Path:
    if scratch_root is not None:
        return scratch_root
    raw = os.environ.get(SCRATCH_ROOT_ENV)
    if raw:
        return Path(raw)
    return DEFAULT_SCRATCH_ROOT


def _first_paragraph(lines: list[str]) -> str:
    para: list[str] = []
    for line in lines:
        if not line.strip():
            if para:
                break
            continue
        para.append(line)
    return "\n".join(para).strip()


def _first_milestone(section: list[str]) -> str:
    collected: list[str] = []
    found = False
    for line in section:
        if not found:
            if MILESTONE_RE.match(line) or SUBITEM_RE.match(line):
                found = True
                collected.append(line)
        elif MILESTONE_RE.match(line):
            break
        else:
            collected.append(line)
    if not collected:
        return _first_paragraph(section)
    return "\n".join(collected).strip()


def _extract_first_phase(md: str) -> str:
    """Return the first milestone from a blueprint's Phased Build Plan."""
    if not md.strip():
        return ""
    lines = md.splitlines()
    start = None
    for i, line in enumerate(lines):
        lowered = line.strip().lower()
        if lowered.startswith("##") and PHASED_PLAN_HEADING in lowered:
            start = i
            break
    if start is None:
        return _first_paragraph(lines)
    section: list[str] = []
    for line in lines[start + 1 :]:
        if line.strip().lower().startswith("##"):
            break
        section.append(line)
    return _first_milestone(section)


def _build_coder_prompt(blueprint: Blueprint) -> str:
    first_phase = _extract_first_phase(blueprint.blueprint_md or "")
    return (
        "You are drafting a working prototype inside a fresh, empty directory.\n\n"
        f"Feasibility score of the parent blueprint: {blueprint.feasibility_score}\n\n"
        "First phase to implement (from the blueprint's Phased Build Plan):\n"
        f"{first_phase}\n\n"
        "Build a minimal, self-contained, working implementation of this phase in "
        "this directory. Keep the scope to exactly this phase - do not build ahead "
        "into later milestones. Write real code, add a short README explaining how "
        "to run it, and keep dependencies minimal."
    )


def _invoke_opencode(
    prompt: str,
    cwd: Path,
    timeout: float,
) -> subprocess.CompletedProcess:
    """Run OpenCode CLI in non-interactive mode against a scratch directory."""
    if os.name == "nt":
        cmd = ["cmd", "/c", "opencode", "run", "--auto", prompt]
    else:
        cmd = ["opencode", "run", "--auto", prompt]
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _persist_status(
    blueprint: Blueprint,
    candidate_status: str,
    prototype_path: str | None,
) -> None:
    session = get_session()
    try:
        stored = session.get(Blueprint, blueprint.id) if blueprint.id is not None else None
        if stored is None:
            stored = blueprint
            session.add(stored)
        if prototype_path is not None:
            stored.prototype_path = prototype_path
        if blueprint.candidate_id is not None:
            candidate = session.get(Candidate, blueprint.candidate_id)
            if candidate is not None:
                candidate.status = candidate_status
            else:
                logger.warning(
                    "coder: candidate %s not found; status not updated",
                    blueprint.candidate_id,
                )
        session.commit()
    finally:
        session.close()


def draft_prototype(
    blueprint: Blueprint,
    scratch_root: Path | None = None,
    timeout: float | None = None,
) -> Path | None:
    """Draft a prototype from the blueprint's first phase via OpenCode CLI."""
    timeout = _resolve_timeout(timeout)
    root = _resolve_scratch_root(scratch_root)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    scratch = root / f"proto-{blueprint.candidate_id}-{timestamp}"
    scratch.mkdir(parents=True, exist_ok=True)

    prompt = _build_coder_prompt(blueprint)
    try:
        proc = _invoke_opencode(prompt, scratch, timeout)
    except subprocess.TimeoutExpired as exc:
        logger.warning(
            "coder: opencode timed out after %ss for blueprint %s: %s",
            timeout,
            blueprint.id,
            exc,
        )
        _persist_status(blueprint, "prototype_failed", None)
        return None

    if proc.returncode != 0:
        logger.warning(
            "coder: opencode failed for blueprint %s (rc=%s): %s",
            blueprint.id,
            proc.returncode,
            (proc.stderr or proc.stdout or "")[-2000:],
        )
        _persist_status(blueprint, "prototype_failed", None)
        return None

    _persist_status(blueprint, "prototyped", str(scratch))
    return scratch


coder = draft_prototype
