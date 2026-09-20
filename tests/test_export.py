"""Tests for `praxis export`: build-kit rendering and file writing."""

from __future__ import annotations

import pytest

from praxis.cli import main
from praxis.db import Blueprint, Candidate
from praxis.export import render_build_kit


def _blueprint_md() -> str:
    return (
        "# CPU Fine-Tune — Blueprint\n"
        "\n"
        "## Problem Statement\n"
        "Fine-tune a small model on CPU.\n"
        "\n"
        "## Proposed Architecture\n"
        "CPU-only, 8GB RAM; no GPU needed.\n"
        "\n"
        "## Phased Build Plan\n"
        "1. Data loader\n"
        "2. Training loop\n"
        "\n"
        "## Deferred to Later Versions\n"
        "GPU acceleration.\n"
        "\n"
        "## Difficulty & Time Estimate\n"
        "Easy, 1 week.\n"
    )


@pytest.fixture
def seeded_blueprint(tmp_path, monkeypatch):
    """One candidate with a blueprint (id 1) and one without (id 2), on a temp DB."""
    from sqlalchemy.orm import Session

    from praxis.db import Base, get_engine

    db_path = tmp_path / "export.db"
    monkeypatch.setenv("PRAXIS_DB_URL", f"sqlite:///{db_path}")
    engine = get_engine()
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        with_bp = Candidate(
            source="arxiv",
            url="https://arxiv.org/abs/2401.12345",
            title="CPU Fine-Tune",
            raw_text="Fine-tune a small transformer on CPU.",
            technique_summary="LoRA fine-tuning on CPU",
            feasibility_score=8,
            status="blueprinted",
        )
        session.add(with_bp)
        session.flush()
        blueprint_id = with_bp.id
        session.add(
            Blueprint(
                candidate_id=blueprint_id,
                feasibility_score=8.0,
                blueprint_md=_blueprint_md(),
            )
        )
        session.add(
            Candidate(
                source="arxiv",
                url="https://arxiv.org/abs/2401.99999",
                title="No Blueprint Yet",
                raw_text="raw",
                status="new",
            )
        )
        session.commit()
    return blueprint_id


def test_render_build_kit_contains_prompt_and_constraints(hardware_profile):
    from praxis.db import Candidate

    candidate = Candidate(
        source="arxiv",
        url="https://arxiv.org/abs/2401.12345",
        title="CPU Fine-Tune",
        raw_text="x",
        status="blueprinted",
    )

    class _FakeBlueprint:
        blueprint_md = _blueprint_md()
        feasibility_score = 8.0

    kit = render_build_kit(candidate, _FakeBlueprint(), hardware_profile)
    assert "## Goal" in kit
    assert "## Hardware constraints" in kit
    assert f"- RAM: {hardware_profile.ram_gb} GB" in kit
    assert "## Acceptance checks" in kit
    assert "1. Data loader" in kit  # plan content carried through
    assert "CPU-only: yes" in kit


def test_export_writes_file(seeded_blueprint, tmp_path):
    out = tmp_path / "kit.md"
    code = main(["export", str(seeded_blueprint), "--out", str(out)])
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "## Goal" in text
    assert "## Hardware constraints" in text
    assert "1. Data loader" in text
    assert "## Acceptance checks" in text


def test_export_default_path(seeded_blueprint, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code = main(["export", str(seeded_blueprint)])
    assert code == 0
    assert (tmp_path / f"build-kit-{seeded_blueprint}.md").exists()


def test_export_missing_candidate(tmp_path, monkeypatch):
    monkeypatch.setenv("PRAXIS_DB_URL", f"sqlite:///{tmp_path / 'none.db'}")
    code = main(["export", "999"])
    assert code == 1


def test_export_candidate_without_blueprint(seeded_blueprint):
    """Candidate 2 is seeded without a blueprint: export must fail cleanly."""
    code = main(["export", str(seeded_blueprint + 1), "--out", "should-not-exist.md"])
    assert code == 1


def test_record_pick_appends_focus():
    """The focus note is stored in the build-memory entry text."""
    from praxis.db import BuildMemory
    from praxis.design_io import record_pick

    calls = []

    class _Session:
        def add(self, row):
            calls.append(row)

        def commit(self):
            pass

        def close(self):
            pass

    import praxis.design_io as dio

    original = dio.get_session
    dio.get_session = lambda: _Session()
    try:
        record_pick(1, "LoRA on CPU", "keep it small")
    finally:
        dio.get_session = original
    assert len(calls) == 1
    entry = calls[0]
    assert isinstance(entry, BuildMemory)
    assert "LoRA on CPU" in entry.technique
    assert "focus: keep it small" in entry.technique
    assert entry.outcome == "picked_for_design"
