"""Export a finished design's docs pack into a fresh, agent-ready project dir.

`praxis handoff <id> <target>` is the bridge between a stored design and an
external coding agent: it validates the design (same consistency checks as
``praxis check``), copies the eleven allowlisted pack documents out of
``designs/<NNN>-<slug>/docs/``, runs ``git init``, and writes a plan-mode
HANDOFF_PROMPT.md. Nothing outside the allowlist ever leaves the Praxis tree,
and a post-copy secret scan refuses to leave key-like content behind.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from praxis.config import load_facts
from praxis.consistency import SEVERITY_ERROR, run_checks
from praxis.db import get_design_meta, latest_design
from praxis.design import PROVENANCE_VERSION
from praxis.design_io import design_dir, load_passes
from praxis.planning import PACK_PASS_IDS

# The exact eleven files `praxis plan` writes into docs/. Anything else in
# docs/ (stray .env, .pem, editor droppings) fails the handoff unless --force,
# and even then only allowlisted names are ever copied.
PACK_ALLOWLIST: tuple[str, ...] = (
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
)

# Key-like content that must never reach a fresh repo. Prefixes require real
# key length, and assignments require a non-empty value, so placeholder
# names in .env.example ("NVIDIA_NIM_API_KEY=") and words like "task-" or
# "scikit-learn" can never match.
_KEY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("gsk_", re.compile(r"gsk_[A-Za-z0-9]{8,}")),
    ("nvapi-", re.compile(r"nvapi-[A-Za-z0-9]{8,}")),
    ("sk-", re.compile(r"\bsk-[A-Za-z0-9]{8,}")),
    ("Bearer", re.compile(r"Bearer\s+[A-Za-z0-9\-_.]{8,}")),
    ("api_key=", re.compile(r"(?i)api[_-]?key\s*[=:]\s*[\"']?[A-Za-z0-9\-_]{8,}")),
)

# A matched value carrying one of these markers is a documented placeholder
# (packs ship lines like GROQ_API_KEY=gsk_local_development_key_placeholder),
# not a secret.
_PLACEHOLDER_MARKERS = (
    "placeholder",
    "example",
    "changeme",
    "change_me",
    "your_",
    "your-",
    "dummy",
    "sample",
    "xxxx",
    "<",
)

HANDOFF_PROMPT_TEMPLATE = """\
# Handoff: {title} (candidate {candidate_id})

You are a fresh coding-agent session starting this project in an empty git
repo. The design pack was exported here by `praxis handoff`.

## Workflow — plan mode first, no code yet

1. Read, in order: README.md, PRD.md, ARCHITECTURE.md, DESIGN.md, RULES.md,
   TASKS.md, TEST_PLAN.md, SECURITY.md, DECISIONS.md, MEMORY.md,
   .env.example.
2. Obey RULES.md constraints absolutely.
3. Propose a detailed plan for ONLY milestone/phase 1 of TASKS.md: module
   layout, exact signatures/types, config defaults, error handling, ordered
   file-by-file work, and the full test list. List every ambiguity or
   contradiction you find between the docs with file:line, propose a
   resolution, and mark it "user must confirm".
4. Stop and wait for the user's approval. No code until the plan is approved.

## Safety invariants (every session)

- Never print .env or any key/secret.
- No commits, pushes, or changes outside this project dir without explicit
  per-instance approval.
- No dependencies beyond what the docs justify; respect RULES.md.

## Goal (from the design)

{agent_prompt}
"""


class HandoffError(Exception):
    """A handoff failure whose message is already CLI-friendly."""


def collect_pack_files(docs_dir: Path) -> tuple[list[Path], list[str]]:
    """Split docs_dir entries into (allowlisted files, non-allowlisted names)."""
    files: list[Path] = []
    strays: list[str] = []
    for entry in sorted(docs_dir.iterdir()):
        if entry.is_file() and entry.name in PACK_ALLOWLIST:
            files.append(entry)
        else:
            strays.append(entry.name)
    return files, strays


def _canonical(path: Path) -> Path:
    """Symlink/junction-resolved, case-folded path for containment checks."""
    return Path(os.path.normcase(str(path.resolve())))


def validate_target(target: Path, repo_root: Path, force: bool) -> None:
    """Refuse targets inside the Praxis repo, ancestors of it, or non-empty."""
    t = _canonical(target)
    r = _canonical(repo_root)
    if t == r or t in r.parents or r in t.parents:
        raise HandoffError(
            f"error: target {target} is inside the Praxis repo (or is/contains "
            f"it); choose a directory outside {repo_root}"
        )
    if target.exists() and any(target.iterdir()) and not force:
        raise HandoffError(
            f"error: target {target} is not empty; pass --force to "
            f"overwrite-merge into it (extras are kept, never deleted)"
        )
    if shutil.which("git") is None:
        raise HandoffError("error: git not found on PATH")


def _scan_for_secrets(target: Path) -> list[tuple[str, str]]:
    """Return (relative path, pattern label) hits for key-like content."""
    hits: list[tuple[str, str]] = []
    for path in sorted(target.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            for label, pattern in _KEY_PATTERNS:
                m = pattern.search(line)
                if not m:
                    continue
                matched = m.group(0).lower()
                if any(marker in matched for marker in _PLACEHOLDER_MARKERS):
                    continue
                hits.append((str(path.relative_to(target)), label))
                break
    return hits


def build_handoff_prompt(title: str, candidate_id: int, agent_prompt_md: str) -> str:
    return HANDOFF_PROMPT_TEMPLATE.format(
        title=title, candidate_id=candidate_id, agent_prompt=agent_prompt_md.strip()
    )


def run_handoff(
    candidate_id: int,
    target: str | Path,
    *,
    force: bool = False,
    skip_check: bool = False,
    design_root: Path | None = None,
) -> int:
    """Export candidate's design docs pack into target. Returns a process exit code."""
    from praxis.discover import get_candidate

    def fail(message: str) -> int:
        print(message, file=sys.stderr)
        return 1

    candidate = get_candidate(candidate_id)
    if candidate is None:
        return fail(f"error: no candidate with id {candidate_id}")

    design = latest_design(candidate_id)
    if design is None:
        return fail(f"error: candidate {candidate_id} has no design")

    passes = load_passes(design)
    missing_pack = [p for p in PACK_PASS_IDS if not (passes.get(p) or "").strip()]
    if missing_pack:
        return fail(
            "error: docs pack incomplete (missing: "
            + ", ".join(missing_pack)
            + f"); run 'praxis plan {candidate_id}' first"
        )

    if not skip_check:
        meta = get_design_meta(design.id) if design.id is not None else {}
        findings = run_checks(
            passes,
            load_facts(),
            stamped=meta.get("pack_version") == PROVENANCE_VERSION,
        )
        errors = [f for f in findings if f.severity == SEVERITY_ERROR]
        if errors:
            for f in errors:
                print(f"[ERROR] {f.rule_id} {f.pass_id}: {f.message}")
            return fail(
                f"error: {len(errors)} consistency error(s); "
                f"fix them or pass --skip-check"
            )

    title = (getattr(candidate, "title", "") or f"candidate-{candidate_id}").strip()
    design_path = design_dir(candidate_id, title, root=design_root)
    docs_dir = design_path / "docs"
    if not docs_dir.is_dir():
        return fail(f"error: no docs pack at {docs_dir}; run 'praxis plan {candidate_id}' first")
    agent_prompt_path = design_path / "AGENT_PROMPT.md"
    if not agent_prompt_path.is_file():
        return fail(
            f"error: no AGENT_PROMPT.md at {agent_prompt_path}; "
            f"run 'praxis design {candidate_id}' first"
        )

    pack_files, strays = collect_pack_files(docs_dir)
    if strays and not force:
        return fail(
            f"error: docs/ contains non-pack entries: {', '.join(strays)}; "
            f"remove them or pass --force to copy the {len(PACK_ALLOWLIST)}-file allowlist anyway"
        )

    target_path = Path(target).expanduser()
    repo_root = design_path.parents[1]  # <root>/designs/<NNN>-slug -> <root>
    target_existed = target_path.exists()
    copied: list[Path] = []
    prompt_path = target_path / "HANDOFF_PROMPT.md"
    try:
        validate_target(target_path, repo_root, force)
        target_path.mkdir(parents=True, exist_ok=True)
        for src in pack_files:
            dest = target_path / src.name
            shutil.copy2(src, dest)
            copied.append(dest)
        hits = _scan_for_secrets(target_path)
        if hits:
            where = ", ".join(f"{path} ({label})" for path, label in hits[:5])
            raise HandoffError(
                f"error: key-like content found after copy ({where}); "
                f"refusing the handoff"
            )
        agent_prompt = agent_prompt_path.read_text(encoding="utf-8")
        prompt_path.write_text(
            build_handoff_prompt(title, candidate_id, agent_prompt),
            encoding="utf-8",
        )
        copied.append(prompt_path)
        result = subprocess.run(
            ["git", "init"], cwd=target_path, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise HandoffError(
                f"error: git init failed in {target_path}: {result.stderr.strip()}"
            )
    except HandoffError as exc:
        # Never leave a half-export (or secret-bearing) tree behind.
        for path in copied:
            path.unlink(missing_ok=True)
        if not target_existed and target_path.exists():
            shutil.rmtree(target_path, ignore_errors=True)
        return fail(str(exc))

    if strays:
        print(f"note: skipped non-pack docs/ entries: {', '.join(strays)}")
    print(
        f"handoff ready: {target_path} "
        f"({len(pack_files)} pack files + HANDOFF_PROMPT.md, git initialized)"
    )
    print(f"next: open a coding agent in {target_path} and paste HANDOFF_PROMPT.md")
    return 0
