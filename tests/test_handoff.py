"""Tests for `praxis handoff` — design export into an agent-ready project dir."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from praxis.cli import main
from praxis.handoff import PACK_ALLOWLIST, collect_pack_files, run_handoff


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
    assert names == sorted([*PACK_ALLOWLIST, "HANDOFF_PROMPT.md", ".git"])
    out = capsys.readouterr().out
    assert "handoff ready" in out and "git initialized" in out


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


def test_handoff_cli_wiring(handoff_env, monkeypatch, capsys):
    monkeypatch.chdir(handoff_env.repo_root)
    target = handoff_env.tmp_path / "out"
    rc = main(["handoff", "7", str(target)])
    assert rc == 0
    assert "handoff ready" in capsys.readouterr().out


def test_collect_pack_files_splits_allowlist(tmp_path):
    (tmp_path / "PRD.md").write_text("x", encoding="utf-8")
    (tmp_path / ".env").write_text("x", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    files, strays = collect_pack_files(tmp_path)
    assert [f.name for f in files] == ["PRD.md"]
    assert sorted(strays) == [".env", "notes.txt"]
