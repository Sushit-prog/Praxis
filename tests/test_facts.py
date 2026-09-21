"""Tests for the facts sheet (hardware_profile.yaml -> hard design constraints)."""

from __future__ import annotations

from praxis.config import FactsSheet, load_facts, render_facts_sheet


def test_load_facts_defaults_when_yaml_missing(tmp_path):
    """A missing/incomplete YAML falls back to safe defaults for every key."""
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")

    facts = load_facts(str(empty))

    assert facts == FactsSheet()  # every field at its documented default
    assert facts.os == "Windows 11"
    assert facts.ram_gb == 8
    assert facts.usable_ram_gb == 4  # ~3-4 GB usable headroom for the app
    assert facts.monthly_budget_usd == 15.0
    assert facts.cpu_only is True and facts.gpu is False
    assert facts.local_only is True  # runs locally; no VPS unless justified
    assert facts.provider_limits == [] and facts.stack_notes == []


def test_load_facts_defaults_scale_headroom_with_ram(tmp_path):
    """Missing usable_ram_gb derives the headroom from ram_gb."""
    yaml_file = tmp_path / "p.yaml"
    yaml_file.write_text("ram_gb: 16\n", encoding="utf-8")
    facts = load_facts(str(yaml_file))
    assert facts.ram_gb == 16
    assert facts.usable_ram_gb == 12


def test_load_facts_reads_full_profile():
    """The repo's hardware_profile.yaml loads with all facts-sheet keys."""
    facts = load_facts()  # default path: ./hardware_profile.yaml

    assert facts.os == "Windows 11"
    assert facts.cpu_only is True
    assert facts.gpu is False
    assert facts.ram_gb == 8
    assert facts.usable_ram_gb == 4
    assert facts.monthly_budget_usd == 15.0
    assert facts.local_only is True
    joined = " | ".join(facts.provider_limits)
    assert "groq gpt-oss-20b" in joined and "8000 TPM" in joined
    notes = " | ".join(facts.stack_notes)
    assert "llama.cpp" in notes and "GGUF" in notes
    assert any("embedding model" in note for note in facts.stack_notes)
    assert any("SQLite" in note for note in facts.stack_notes)
    assert any("CUDA-only" in note for note in facts.stack_notes)
    assert any("Docker" in note or "WSL2" in note for note in facts.stack_notes)


def test_env_overrides_beat_yaml(tmp_path, monkeypatch):
    yaml_file = tmp_path / "p.yaml"
    yaml_file.write_text("ram_gb: 8\nmonthly_budget_usd: 15.0\n", encoding="utf-8")
    monkeypatch.setenv("PRAXIS_RAM_GB", "32")
    monkeypatch.setenv("PRAXIS_MONTHLY_BUDGET_USD", "5")

    facts = load_facts(str(yaml_file))

    assert facts.ram_gb == 32
    assert facts.monthly_budget_usd == 5.0


def test_render_facts_sheet_contains_every_constraint():
    facts = FactsSheet(
        provider_limits=["groq gpt-oss-20b: 8000 TPM"],
        preferred_stack=["Python", "SQLite"],
        stack_notes=["Local models: llama.cpp with GGUF weights"],
        avoid=["CUDA-only libraries"],
    )
    md = render_facts_sheet(facts)

    assert "HARD CONSTRAINTS" in md
    assert "Windows 11" in md
    assert "no GPU (CPU-only)" in md
    assert "8 GB total" in md and "4 GB usable headroom" in md
    assert "$15.00" in md
    assert "no VPS unless justified" in md
    assert "8000 TPM" in md
    assert "Python, SQLite" in md
    assert "llama.cpp with GGUF weights" in md
    assert "CUDA-only libraries" in md
    assert "UNVERIFIED: check before relying" in md


def test_render_facts_sheet_omits_empty_sections():
    md = render_facts_sheet(FactsSheet())
    assert "Provider free-tier limits" not in md
    assert "Stack notes" not in md
    assert "Avoid" not in md
