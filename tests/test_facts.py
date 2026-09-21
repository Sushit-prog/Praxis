"""Tests for the facts sheet (hardware_profile.yaml -> hard design constraints)."""

from __future__ import annotations

from praxis.config import FactsSheet, load_facts, render_facts_sheet, resolve_model_limits


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
    # Per-model limits carry the seeded free-tier observations.
    by_model = {e.model: e for e in facts.model_limits}
    assert by_model["groq/openai/gpt-oss-120b"].tpm == 8000
    assert by_model["groq/openai/gpt-oss-20b"].tpm == 8000
    qwen = by_model["groq/qwen/qwen3.8-27b"]
    assert qwen.itpm == 7000 and qwen.otpm == 1000
    assert all("Sep 2026" in e.note for e in facts.model_limits)
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


# ---------------------------------------------------------------------------
# Per-model rate limits (tpm / itpm / otpm), each optional
# ---------------------------------------------------------------------------


def test_model_limits_default_empty_and_entry_fields_optional():
    """No YAML entry -> no limits; a partial entry throttles only its axes."""
    from praxis.config import ModelLimits

    assert FactsSheet().model_limits == []

    facts = FactsSheet(model_limits=[ModelLimits(model="m/one")])
    limits = resolve_model_limits(facts, "m/one")
    assert limits is not None
    assert limits.tpm is None and limits.itpm is None and limits.otpm is None


def test_resolve_model_limits_exact_and_suffix_match():
    from praxis.config import ModelLimits

    facts = FactsSheet(
        model_limits=[
            ModelLimits(model="groq/openai/gpt-oss-120b", tpm=8000),
            ModelLimits(model="qwen/qwen3.8-27b", itpm=7000, otpm=1000),
        ]
    )
    # Exact full-id match.
    assert resolve_model_limits(facts, "groq/openai/gpt-oss-120b").tpm == 8000
    # Suffix match: YAML entry without the provider prefix.
    limits = resolve_model_limits(facts, "groq/qwen/qwen3.8-27b")
    assert limits.itpm == 7000 and limits.otpm == 1000
    # A model with no entry is not throttled.
    assert resolve_model_limits(facts, "cerebras/qwen-3.8-27b") is None
    assert resolve_model_limits(FactsSheet(), "groq/openai/gpt-oss-120b") is None


def test_model_limits_yaml_missing_or_malformed_is_tolerated(tmp_path):
    """A missing or malformed model_limits block degrades to no limits."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "model_limits:\n  - tpm: 8000\n  - not: a dict\nmodel_limits_str: oops\n",
        encoding="utf-8",
    )
    facts = load_facts(str(bad))
    assert facts.model_limits == []

    wrong_type = tmp_path / "wrong.yaml"
    wrong_type.write_text("model_limits: 42\n", encoding="utf-8")
    assert load_facts(str(wrong_type)).model_limits == []


def test_render_facts_sheet_lists_per_model_limits():
    from praxis.config import ModelLimits

    facts = FactsSheet(
        model_limits=[
            ModelLimits(
                model="groq/openai/gpt-oss-120b",
                tpm=8000,
                note="observed on the free tier, Sep 2026, may change",
            ),
            ModelLimits(model="groq/qwen/qwen3.8-27b", itpm=7000, otpm=1000),
        ]
    )
    md = render_facts_sheet(facts)
    assert "groq/openai/gpt-oss-120b: 8000 TPM total" in md
    assert "groq/qwen/qwen3.8-27b: 7000 ITPM input, 1000 OTPM output" in md
    assert "Sep 2026" in md
