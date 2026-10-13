"""Tests for `praxis handoff` — design export into an agent-ready project dir."""

from __future__ import annotations

import json
import re
import sys
from types import SimpleNamespace

import pytest

from praxis.cli import main
from praxis.handoff import (
    _PRAXIS_HEADER,
    GITIGNORE_UNIVERSAL,
    PACK_ALLOWLIST,
    collect_pack_files,
    run_handoff,
)

# A subcommand of the CLI may never appear in the agent-facing documents:
# the external agent must not be told to run any praxis command.
_PRAXIS_COMMAND_RE = re.compile(
    r"\bpraxis (check|design|plan|handoff|doctor|specs|discover|run|export|eval)\b",
    re.I,
)

_FAKE_KEY = "gsk_" + "Rt9Kp2Lm" * 3  # real-looking, no placeholder marker


@pytest.fixture
def handoff_env(tmp_path, monkeypatch):
    """A seeded design DB plus an on-disk docs pack under a fake repo root."""
    from sqlalchemy.orm import Session

    from praxis.consistency import KNOWN_PASS_IDS
    from praxis.db import Base, Candidate, Design, get_engine
    from praxis.design_io import design_dir

    db_path = tmp_path / "handoff.db"
    monkeypatch.setenv("PRAXIS_DB_URL", f"sqlite:///{db_path}")
    monkeypatch.setattr("praxis.config.load_facts", lambda *a, **k: None)
    engine = get_engine()
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        candidate = Candidate(
            id=7,
            source="arxiv",
            url="https://a",
            title="Fake Slug",
            raw_text="x",
            status="designed",
        )
        session.add(candidate)
        session.flush()
        session.add(
            Design(
                candidate_id=7,
                status="complete",
                passes_json=json.dumps(
                    {
                        pid: "Everything is internally consistent and within budget."
                        for pid in KNOWN_PASS_IDS
                    }
                ),
            )
        )
        session.commit()

    repo_root = tmp_path / "Repo"
    design_path = design_dir(7, "Fake Slug", root=repo_root)
    docs_dir = design_path / "docs"
    docs_dir.mkdir(parents=True)
    for name in PACK_ALLOWLIST:
        if name == ".env.example":
            # Realistic pack content: a placeholder key value that must NOT
            # trip the post-copy secret scan.
            content = "GROQ_API_KEY=gsk_local_development_key_placeholder\n"
        else:
            content = f"# {name}\n\nPlaceholder content for {name}.\n"
        (docs_dir / name).write_text(content, encoding="utf-8")
    (design_path / "AGENT_PROMPT.md").write_text(
        "# Coding agent prompt\n\n```text\nGoal: build the fake thing.\n```\n",
        encoding="utf-8",
    )
    return SimpleNamespace(
        tmp_path=tmp_path, repo_root=repo_root, design_path=design_path, docs_dir=docs_dir
    )


def _seed_bad_plan(monkeypatch):
    """Make load_passes inject an r1 arithmetic error into the plan pass."""
    import praxis.design_io as design_io

    real_load = design_io.load_passes

    def load_with_error(design):
        passes = real_load(design)
        passes["plan"] = "The fallback is 2k × 3 MB is 6 MB."
        return passes

    monkeypatch.setattr("praxis.handoff.load_passes", load_with_error)


def test_handoff_copies_pack_only(handoff_env, capsys):
    target = handoff_env.tmp_path / "out"
    rc = run_handoff(7, target, design_root=handoff_env.repo_root)
    assert rc == 0
    names = sorted(p.name for p in target.iterdir())
    assert names == sorted(
        [
            *PACK_ALLOWLIST,
            "HANDOFF_PROMPT.md",
            "AGENTS.md",
            ".gitignore",
            ".git",
        ]
    )
    assert (target / "RECOMMENDATIONS.md").exists()
    out = capsys.readouterr().out
    assert "handoff ready" in out and "git initialized" in out
    assert "AGENTS.md" in out and ".gitignore" in out
    assert ".cursor" not in out


def test_handoff_legacy_eleven_file_pack(handoff_env, capsys):
    """A legacy pack without RECOMMENDATIONS.md exports the core eleven."""
    (handoff_env.docs_dir / "RECOMMENDATIONS.md").unlink()
    target = handoff_env.tmp_path / "out"
    rc = run_handoff(7, target, design_root=handoff_env.repo_root)
    assert rc == 0
    names = sorted(p.name for p in target.iterdir())
    expected = sorted(
        [n for n in PACK_ALLOWLIST if n != "RECOMMENDATIONS.md"]
        + ["HANDOFF_PROMPT.md", "AGENTS.md", ".gitignore", ".git"]
    )
    assert names == expected
    assert not (target / "RECOMMENDATIONS.md").exists()
    prompt = (target / "HANDOFF_PROMPT.md").read_text(encoding="utf-8")
    read_line = next(line for line in prompt.splitlines() if "Read, in order:" in line)
    assert "RECOMMENDATIONS.md" not in read_line


def test_handoff_stray_env_fails_without_force(handoff_env, capsys):
    (handoff_env.docs_dir / ".env").write_text("GROQ_API_KEY=\n", encoding="utf-8")
    target = handoff_env.tmp_path / "out"
    rc = run_handoff(7, target, design_root=handoff_env.repo_root)
    assert rc == 1
    assert ".env" in capsys.readouterr().err
    assert not target.exists()


def test_handoff_stray_env_skipped_with_force(handoff_env):
    (handoff_env.docs_dir / ".env").write_text("GROQ_API_KEY=\n", encoding="utf-8")
    target = handoff_env.tmp_path / "out"
    rc = run_handoff(7, target, design_root=handoff_env.repo_root, force=True)
    assert rc == 0
    assert not (target / ".env").exists()
    assert (target / ".env.example").exists()


def test_handoff_stray_pem_fails_unless_forced(handoff_env, capsys):
    pem = handoff_env.docs_dir / "deploy.pem"
    pem.write_text("-----BEGIN FAKE PRIVATE KEY-----\n", encoding="utf-8")
    target = handoff_env.tmp_path / "out"
    assert run_handoff(7, target, design_root=handoff_env.repo_root) == 1
    assert "deploy.pem" in capsys.readouterr().err
    assert run_handoff(7, target, design_root=handoff_env.repo_root, force=True) == 0
    assert not (target / "deploy.pem").exists()


def test_handoff_key_scan_aborts_and_cleans(handoff_env, capsys):
    fake_key = "gsk_" + "Rt9Kp2Lm" * 3  # real-looking, no placeholder marker
    mem = handoff_env.docs_dir / "MEMORY.md"
    mem.write_text(f"# MEMORY\n\ntoken: {fake_key}\n", encoding="utf-8")
    target = handoff_env.tmp_path / "out"
    rc = run_handoff(7, target, design_root=handoff_env.repo_root)
    assert rc == 1
    captured = capsys.readouterr()
    assert "key-like content" in captured.err and "gsk_" in captured.err
    assert fake_key not in captured.err  # the value itself is never printed
    assert not target.exists()  # cleaned up: we created it, so it is gone


def test_handoff_key_scan_placeholder_check_is_token_only(handoff_env, capsys):
    """A real-looking key on a line that merely contains 'example' still aborts.

    The placeholder skip inspects only the matched token, never the whole line.
    """
    fake_key = "gsk_" + "Rt9Kp2Lm" * 3
    mem = handoff_env.docs_dir / "MEMORY.md"
    mem.write_text(
        f"# MEMORY\n\nSee the example config; live token: {fake_key}\n", encoding="utf-8"
    )
    target = handoff_env.tmp_path / "out"
    rc = run_handoff(7, target, design_root=handoff_env.repo_root)
    assert rc == 1
    captured = capsys.readouterr()
    assert "key-like content" in captured.err
    assert fake_key not in captured.err
    assert not target.exists()


def test_handoff_refuses_target_inside_repo(handoff_env):
    target = handoff_env.repo_root / "newproj"
    rc = run_handoff(7, target, design_root=handoff_env.repo_root)
    assert rc == 1
    assert not target.exists()


def test_handoff_refuses_target_parent_of_repo(handoff_env):
    rc = run_handoff(7, handoff_env.tmp_path, design_root=handoff_env.repo_root)
    assert rc == 1


@pytest.mark.skipif(sys.platform != "win32", reason="mixed-case paths only unify on Windows")
def test_handoff_mixed_case_path_still_refused(handoff_env):
    target = handoff_env.tmp_path / "rEpO" / "newproj"  # repo_root is tmp_path/"Repo"
    rc = run_handoff(7, target, design_root=handoff_env.repo_root)
    assert rc == 1
    assert not target.exists()


def test_handoff_refuses_nonempty_target(handoff_env):
    target = handoff_env.tmp_path / "out"
    target.mkdir()
    (target / "keepme.txt").write_text("precious\n", encoding="utf-8")
    rc = run_handoff(7, target, design_root=handoff_env.repo_root)
    assert rc == 1
    assert (target / "keepme.txt").read_text(encoding="utf-8") == "precious\n"
    rc = run_handoff(7, target, design_root=handoff_env.repo_root, force=True)
    assert rc == 0
    assert (target / "keepme.txt").exists()  # extras are kept
    assert (target / "README.md").exists()  # conflicting/missing names merged in


def test_handoff_missing_candidate_or_pack_errors(handoff_env, capsys):
    target = handoff_env.tmp_path / "out"
    assert run_handoff(99, target, design_root=handoff_env.repo_root) == 1
    assert "no candidate" in capsys.readouterr().err
    # Design exists but docs/ was removed -> plan hint.
    import shutil

    shutil.rmtree(handoff_env.docs_dir)
    assert run_handoff(7, target, design_root=handoff_env.repo_root) == 1
    assert "praxis plan 7" in capsys.readouterr().err


def test_handoff_refuses_on_check_errors(handoff_env, capsys, monkeypatch):
    _seed_bad_plan(monkeypatch)
    target = handoff_env.tmp_path / "out"
    assert run_handoff(7, target, design_root=handoff_env.repo_root) == 1
    assert "consistency error" in capsys.readouterr().err
    assert not target.exists()
    assert run_handoff(7, target, design_root=handoff_env.repo_root, skip_check=True) == 0
    assert target.exists()


def test_handoff_prompt_content(handoff_env):
    target = handoff_env.tmp_path / "out"
    assert run_handoff(7, target, design_root=handoff_env.repo_root) == 0
    prompt = (target / "HANDOFF_PROMPT.md").read_text(encoding="utf-8")
    assert "plan mode first" in prompt
    assert "wait for the user's approval" in prompt
    assert "Never print .env" in prompt
    assert "Fake Slug" in prompt and "candidate 7" in prompt
    assert "Goal: build the fake thing." in prompt  # AGENT_PROMPT.md inlined
    assert "exported by the Praxis tool" in prompt  # no command name
    assert "RECOMMENDATIONS.md" in prompt  # derived read order, all 12 copied


def test_handoff_prompt_implement_loop(handoff_env):
    target = handoff_env.tmp_path / "out"
    assert run_handoff(7, target, design_root=handoff_env.repo_root) == 0
    prompt = (target / "HANDOFF_PROMPT.md").read_text(encoding="utf-8")
    assert "Work TASK by TASK within milestone/phase 1 only" in prompt
    assert "the command from TEST_PLAN.md or README.md" in prompt
    assert "do not default to pytest" in prompt
    assert "Tick its checkbox in TASKS.md" in prompt
    assert "MEMORY.md Status / Current task / Next step" in prompt
    assert "Propose a commit message and STOP for approval" in prompt
    assert "Commit only after" in prompt
    assert "Never start milestone 2 in this session" in prompt
    assert "No code until the plan is approved" in prompt  # plan-first steps intact


def test_handoff_prompt_and_agents_name_no_praxis_commands(handoff_env):
    target = handoff_env.tmp_path / "out"
    assert run_handoff(7, target, design_root=handoff_env.repo_root) == 0
    prompt = (target / "HANDOFF_PROMPT.md").read_text(encoding="utf-8")
    agents = (target / "AGENTS.md").read_text(encoding="utf-8")
    assert _PRAXIS_COMMAND_RE.search(prompt) is None
    assert _PRAXIS_COMMAND_RE.search(agents) is None


def test_handoff_agents_md_workflow_and_rules(handoff_env):
    target = handoff_env.tmp_path / "out"
    assert run_handoff(7, target, design_root=handoff_env.repo_root) == 0
    agents = (target / "AGENTS.md").read_text(encoding="utf-8")
    rules = (handoff_env.docs_dir / "RULES.md").read_text(encoding="utf-8")
    assert agents.startswith(_PRAXIS_HEADER)
    assert "## Workflow" in agents
    assert "Read the design docs first: README.md, PRD.md" in agents
    assert "tick it in TASKS.md" in agents
    assert (
        "Work only within the milestone the user approved; never start the "
        "next one without approval." in agents
    )
    assert "update MEMORY.md (Status / Current task / Next step)" in agents
    assert "wait for approval before committing" in agents
    assert "Never print .env or any key/secret" in agents
    assert "Never push" in agents
    assert rules.strip() in agents  # RULES.md verbatim


def test_handoff_cursor_rule_optional(handoff_env):
    target = handoff_env.tmp_path / "out"
    assert run_handoff(7, target, design_root=handoff_env.repo_root) == 0
    assert not (target / ".cursor").exists()

    target2 = handoff_env.tmp_path / "out2"
    assert (
        run_handoff(7, target2, design_root=handoff_env.repo_root, cursor=True) == 0
    )
    mdc = target2 / ".cursor" / "rules" / "praxis.mdc"
    text = mdc.read_text(encoding="utf-8")
    assert text.startswith("---\ndescription: ")
    assert "alwaysApply: true" in text
    assert _PRAXIS_HEADER in text
    rules = (handoff_env.docs_dir / "RULES.md").read_text(encoding="utf-8")
    assert rules.strip() in text
    assert "## Workflow" in text


def test_handoff_keeps_user_agents_md_without_header(handoff_env, capsys):
    target = handoff_env.tmp_path / "out"
    target.mkdir()
    own = "# my own agents file\n"
    (target / "AGENTS.md").write_text(own, encoding="utf-8")
    rc = run_handoff(7, target, design_root=handoff_env.repo_root, force=True)
    assert rc == 0
    assert (target / "AGENTS.md").read_text(encoding="utf-8") == own
    out = capsys.readouterr().out
    assert "note: kept existing AGENTS.md (no Praxis header)" in out


def test_handoff_overwrites_headered_agents_md(handoff_env):
    target = handoff_env.tmp_path / "out"
    target.mkdir()
    (target / "AGENTS.md").write_text(
        f"{_PRAXIS_HEADER}\n\nstale body\n", encoding="utf-8"
    )
    rc = run_handoff(7, target, design_root=handoff_env.repo_root, force=True)
    assert rc == 0
    text = (target / "AGENTS.md").read_text(encoding="utf-8")
    assert "## Workflow" in text
    assert "stale body" not in text


def test_handoff_keeps_user_cursor_rule_without_header(handoff_env, capsys):
    target = handoff_env.tmp_path / "out"
    mdc = target / ".cursor" / "rules" / "praxis.mdc"
    mdc.parent.mkdir(parents=True)
    own = "my own cursor rule\n"
    mdc.write_text(own, encoding="utf-8")
    rc = run_handoff(
        7, target, design_root=handoff_env.repo_root, force=True, cursor=True
    )
    assert rc == 0
    assert mdc.read_text(encoding="utf-8") == own
    assert "note: kept existing .cursor/rules/praxis.mdc (no Praxis header)" in (
        capsys.readouterr().out
    )


def _run_with_arch(handoff_env, arch_text, name="out", **kwargs):
    """Rewrite ARCHITECTURE.md, hand off into <tmp>/<name>, return gitignore."""
    (handoff_env.docs_dir / "ARCHITECTURE.md").write_text(arch_text, encoding="utf-8")
    target = handoff_env.tmp_path / name
    rc = run_handoff(7, target, design_root=handoff_env.repo_root, **kwargs)
    assert rc == 0
    return (target / ".gitignore").read_text(encoding="utf-8")


def test_gitignore_node_next_pack(handoff_env):
    text = _run_with_arch(
        handoff_env,
        arch_text=(
            "# ARCHITECTURE\n\n### Stack\n"
            "- Next.js on TypeScript; package.json manages the deps\n"
            "- Server runs on Node\n"
        ),
    )
    assert "node_modules/" in text
    assert "coverage/" in text
    assert "__pycache__/" not in text
    assert "target/" not in text
    # Next.js block
    assert ".next/" in text
    assert "out/" in text
    assert ".vercel/" in text
    assert "*.tsbuildinfo" in text


def test_gitignore_bare_next_word_fires_nothing(handoff_env):
    """'next step' prose must not add the Next.js block (or anything else)."""
    text = _run_with_arch(
        handoff_env,
        arch_text=(
            "# ARCHITECTURE\n\n### Stack\n"
            "- chore list: next step, follow the next release\n"
        ),
    )
    assert ".next/" not in text
    assert "out/" not in text
    assert ".vercel/" not in text
    assert "*.tsbuildinfo" not in text
    assert "node_modules/" not in text
    assert "__pycache__/" not in text
    for line in GITIGNORE_UNIVERSAL:
        assert line in text


def test_gitignore_python_pack(handoff_env):
    text = _run_with_arch(
        handoff_env,
        arch_text="# ARCHITECTURE\n\n### Stack\n- Python 3.12 via pyproject; pip installs\n",
    )
    assert "__pycache__/" in text
    assert ".venv/" in text
    assert ".ruff_cache/" in text
    assert ".mypy_cache/" in text
    assert "node_modules/" not in text
    assert ".next/" not in text


def test_gitignore_mixed_pack_unions_blocks(handoff_env):
    text = _run_with_arch(
        handoff_env,
        arch_text=(
            "# ARCHITECTURE\n\n### Stack\n"
            "- Python workers via pyproject\n"
            "- Next.js front end (package.json, TypeScript)\n"
        ),
    )
    assert "__pycache__/" in text
    assert "node_modules/" in text
    assert text.count("dist/") == 1  # python and node share it


def test_gitignore_unknown_pack_is_universal_only(handoff_env):
    text = _run_with_arch(
        handoff_env,
        arch_text="# ARCHITECTURE\n\n### Stack\n- Fortran 2008 with a bespoke build\n",
    )
    assert "__pycache__/" not in text
    assert "node_modules/" not in text
    assert "target/" not in text
    for line in GITIGNORE_UNIVERSAL:
        assert line in text


def test_gitignore_word_boundaries_fire_nothing(handoff_env):
    text = _run_with_arch(
        handoff_env,
        arch_text=(
            "# ARCHITECTURE\n\n### Stack\n"
            "- trust the robust pipeline; reactive nodes everywhere\n"
        ),
    )
    assert "__pycache__/" not in text
    assert "node_modules/" not in text
    assert "target/" not in text
    for line in GITIGNORE_UNIVERSAL:
        assert line in text


def test_gitignore_strong_markers_only_without_stack_section(handoff_env):
    text = _run_with_arch(
        handoff_env,
        arch_text="# ARCHITECTURE\n\nWe ship package.json based builds.\n",
        name="out",
    )
    assert "node_modules/" in text  # strong marker, whole-file fallback

    text = _run_with_arch(
        handoff_env,
        arch_text="# ARCHITECTURE\n\nThe reactive pipeline is robust.\n",
        name="out2",
    )
    assert "node_modules/" not in text  # weak words never fire whole-file
    assert "__pycache__/" not in text


def test_gitignore_sqlite_pack_wide(handoff_env):
    text = _run_with_arch(
        handoff_env,
        arch_text="# ARCHITECTURE\n\n### Stack\n- Python via pyproject\n",
    )
    assert "*.db" not in text and "*.sqlite3" not in text

    prd = handoff_env.docs_dir / "PRD.md"
    prd.write_text(
        prd.read_text(encoding="utf-8") + "\nPersistence: SQLite file.\n",
        encoding="utf-8",
    )
    text = _run_with_arch(
        handoff_env,
        arch_text="# ARCHITECTURE\n\n### Stack\n- Python via pyproject\n",
        name="out2",
    )
    assert "*.db" in text and "*.sqlite3" in text


def test_gitignore_merge_append_no_duplicates(handoff_env):
    target = handoff_env.tmp_path / "out"
    target.mkdir()
    (target / ".gitignore").write_text("node_modules/\n", encoding="utf-8")
    rc = run_handoff(7, target, design_root=handoff_env.repo_root, force=True)
    assert rc == 0
    text = (target / ".gitignore").read_text(encoding="utf-8")
    assert text.count("# added by praxis handoff") == 1
    assert text.splitlines()[0] == "node_modules/"
    assert "# added by praxis handoff" in text.splitlines()
    for line in GITIGNORE_UNIVERSAL:
        assert line in text
    first = (target / ".gitignore").read_bytes()

    rc = run_handoff(7, target, design_root=handoff_env.repo_root, force=True)
    assert rc == 0
    assert (target / ".gitignore").read_bytes() == first  # rerun adds nothing


def test_gitignore_merge_preserves_crlf_and_missing_newline(handoff_env):
    target = handoff_env.tmp_path / "out"
    target.mkdir()
    original = b"node_modules/\r\n.custom"  # CRLF endings, no trailing newline
    (target / ".gitignore").write_bytes(original)
    rc = run_handoff(7, target, design_root=handoff_env.repo_root, force=True)
    assert rc == 0
    raw = (target / ".gitignore").read_bytes()
    assert b"\n" not in raw.replace(b"\r\n", b"")  # every ending stayed CRLF
    lines = raw.decode("utf-8").splitlines()
    assert lines[0] == "node_modules/"
    assert lines[1] == ".custom"  # not glued onto the marker line
    assert lines[2] == "# added by praxis handoff"
    assert ".env" in lines
    assert raw.endswith(b"\r\n")


def test_handoff_preexisting_secret_in_force_target_not_scanned(handoff_env, capsys):
    """Only handoff-written files are scanned; user files are never read."""
    target = handoff_env.tmp_path / "out"
    target.mkdir()
    (target / "user_notes.txt").write_text(
        f"my own key {_FAKE_KEY}\n", encoding="utf-8"
    )
    rc = run_handoff(7, target, design_root=handoff_env.repo_root, force=True)
    assert rc == 0
    assert (target / "user_notes.txt").exists()
    out = capsys.readouterr().out
    assert "handoff ready" in out
    assert _FAKE_KEY not in out  # the value is never printed anywhere


def test_handoff_scan_failure_leaves_no_generated_files(handoff_env, capsys):
    """A hit among written files cleans up ours and keeps theirs."""
    target = handoff_env.tmp_path / "out"
    target.mkdir()
    (target / "keepme.txt").write_text("precious\n", encoding="utf-8")
    original_gitignore = b"mine-only\n"
    (target / ".gitignore").write_bytes(original_gitignore)
    mem = handoff_env.docs_dir / "MEMORY.md"
    mem.write_text(f"# MEMORY\n\ntoken: {_FAKE_KEY}\n", encoding="utf-8")

    rc = run_handoff(7, target, design_root=handoff_env.repo_root, force=True)
    assert rc == 1
    assert "key-like content" in capsys.readouterr().err
    assert not (target / "HANDOFF_PROMPT.md").exists()
    assert not (target / "AGENTS.md").exists()
    assert not (target / ".cursor").exists()
    assert not (target / ".git").exists()
    assert (target / ".gitignore").read_bytes() == original_gitignore  # restored
    assert (target / "keepme.txt").read_text(encoding="utf-8") == "precious\n"
    assert not (target / "PRD.md").exists()  # our pack copies are removed too


def test_handoff_cli_wiring(handoff_env, monkeypatch, capsys):
    monkeypatch.chdir(handoff_env.repo_root)
    target = handoff_env.tmp_path / "out"
    rc = main(["handoff", "7", str(target)])
    assert rc == 0
    assert "handoff ready" in capsys.readouterr().out


def test_handoff_cli_cursor_flag(handoff_env, monkeypatch, capsys):
    monkeypatch.chdir(handoff_env.repo_root)
    target = handoff_env.tmp_path / "cursor-out"
    rc = main(["handoff", "7", str(target), "--cursor"])
    assert rc == 0
    assert (target / ".cursor" / "rules" / "praxis.mdc").is_file()
    assert "handoff ready" in capsys.readouterr().out


def test_collect_pack_files_splits_allowlist(tmp_path):
    (tmp_path / "PRD.md").write_text("x", encoding="utf-8")
    (tmp_path / ".env").write_text("x", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    files, strays = collect_pack_files(tmp_path)
    assert [f.name for f in files] == ["PRD.md"]
    assert sorted(strays) == [".env", "notes.txt"]
