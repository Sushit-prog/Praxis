"""Tests for the facts sheet (hardware_profile.yaml -> hard design constraints)."""

from __future__ import annotations

import praxis.config as config_module
from praxis.config import FactsSheet, load_facts, render_facts_sheet, resolve_model_limits
from praxis.hardware import HostHardware


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

    assert facts.os == "Windows 11 + WSL2"
    assert "HP 15s" in facts.cpu
    assert facts.cpu_cores == 12
    assert facts.cpu_only is True
    assert facts.gpu is False
    assert facts.gpu_name == "Intel Iris Xe Graphics"
    assert "not CUDA-capable" in facts.gpu_note
    assert facts.ram_gb == 8
    assert facts.usable_ram_gb == 4
    assert "7.68 GB reported" in facts.ram_note
    # Snapshot only: a live probe overrides this with the current reading.
    assert facts.storage_free_gb == 46
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
    assert "Ollama" in notes
    assert "Python 3.13" in notes
    assert any("embedding model" in note for note in facts.stack_notes)
    assert any("SQLite" in note for note in facts.stack_notes)
    assert any("CUDA-only" in note for note in facts.stack_notes)
    assert any("Docker" in note or "WSL2" in note for note in facts.stack_notes)
    # Hosted option: optional, rate-limited, never a hard dependency.
    limits = " | ".join(facts.provider_limits)
    assert "nvidia_nim/nvidia/nemotron-3-super-120b-a12b" in limits
    assert "rate-limited" in limits and "must not depend on it" in limits


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
    assert "GPU detail" not in md
    assert "RAM detail" not in md
    assert "Free storage" not in md


# ---------------------------------------------------------------------------
# gpu_note / ram_note / storage_free_gb: fields, rendering, YAML round-trip
# ---------------------------------------------------------------------------


def test_facts_sheet_carries_note_fields():
    facts = FactsSheet(
        gpu_note="Intel Iris Xe, ~4 GB shared, not CUDA-capable",
        ram_note="8 GB total / 7.68 GB reported",
        storage_free_gb=46,
    )
    assert facts.gpu_note.endswith("not CUDA-capable")
    assert facts.ram_note.startswith("8 GB total")
    assert facts.storage_free_gb == 46


def test_render_facts_sheet_includes_notes_and_storage():
    facts = FactsSheet(
        gpu_note="Intel Iris Xe, ~4 GB shared, not CUDA-capable",
        ram_note="8 GB total / 7.68 GB reported",
        storage_free_gb=46,
    )
    md = render_facts_sheet(facts)

    assert "- GPU detail: Intel Iris Xe" in md
    assert "- RAM detail: 8 GB total" in md
    assert "- Free storage: 46 GB" in md


def test_load_facts_reads_note_fields_from_yaml(tmp_path):
    yaml_file = tmp_path / "p.yaml"
    yaml_file.write_text(
        "gpu_note: yaml gpu note\nram_note: yaml ram note\nstorage_free_gb: 46\n",
        encoding="utf-8",
    )

    facts = load_facts(str(yaml_file))

    assert facts.gpu_note == "yaml gpu note"
    assert facts.ram_note == "yaml ram note"
    assert facts.storage_free_gb == 46


# ---------------------------------------------------------------------------
# Detection precedence: YAML record vs live probe (stable vs volatile)
# ---------------------------------------------------------------------------


def _host(**overrides) -> HostHardware:
    """A fake probe result; no test ever touches the real host."""
    fields = dict(
        os="Detected OS",
        cpu="Detected CPU",
        cpu_cores=8,
        gpu_name="Detected GPU",
        gpu=False,
        gpu_note="Detected GPU note",
        ram_gb=16,
        ram_reported_gb=15.9,
        usable_ram_gb=9,
        ram_note="Detected RAM note",
        storage_free_gb=7,
        detection_note="",
    )
    fields.update(overrides)
    return HostHardware(**fields)


def test_yaml_wins_for_stable_facts_but_probe_wins_for_volatile(tmp_path):
    """YAML is the record for stable facts; the live probe beats its snapshot."""
    yaml_file = tmp_path / "p.yaml"
    yaml_file.write_text(
        "os: Windows 11 + WSL2\n"
        "cpu: Intel Core (HP 15s laptop)\n"
        "ram_gb: 8\n"
        "usable_ram_gb: 4\n"
        "storage_free_gb: 46\n"
        "gpu_note: curated gpu note\n",
        encoding="utf-8",
    )

    facts = load_facts(str(yaml_file), detected_host=_host())

    # Stable facts: the curated YAML record wins over the probe.
    assert facts.os == "Windows 11 + WSL2"
    assert facts.cpu == "Intel Core (HP 15s laptop)"
    assert facts.ram_gb == 8
    assert facts.gpu_note == "curated gpu note"
    # Volatile readings: the live probe wins over the YAML snapshot.
    assert facts.storage_free_gb == 7
    assert facts.usable_ram_gb == 9
    # The probe fills what the YAML omits.
    assert facts.cpu_cores == 8
    assert facts.detected is True


def test_praxis_config_env_does_not_disable_host_detection(tmp_path, monkeypatch):
    """PRAXIS_CONFIG picks the facts file; it is not a detection kill-switch."""
    yaml_file = tmp_path / "p.yaml"
    yaml_file.write_text(
        "os: Windows 11 + WSL2\nstorage_free_gb: 46\n", encoding="utf-8"
    )
    monkeypatch.setenv("PRAXIS_CONFIG", str(yaml_file))
    monkeypatch.delenv("PRAXIS_DETECT_HW", raising=False)
    monkeypatch.setattr(config_module, "_detect_host_hardware", _host)

    facts = load_facts()  # no path -> PRAXIS_CONFIG file, detection on

    assert facts.detected is True
    assert facts.storage_free_gb == 7  # volatile probe beats the snapshot
    assert facts.os == "Windows 11 + WSL2"  # stable YAML still wins
    assert "- Host probe:" in render_facts_sheet(facts)


def test_detect_hw_zero_keeps_detection_off(monkeypatch):
    """PRAXIS_DETECT_HW=0 is the opt-out (tests rely on it via conftest)."""
    monkeypatch.setenv("PRAXIS_DETECT_HW", "0")
    monkeypatch.setattr(config_module, "_detect_host_hardware", _host)

    facts = load_facts()

    assert facts.detected is False
    assert facts.storage_free_gb == 46  # repo YAML snapshot, no probe


# ---------------------------------------------------------------------------
# Per-model rate limits (tpm / itpm / otpm per-minute, max_output per-request)
# ---------------------------------------------------------------------------


def test_model_limits_default_empty_and_entry_fields_optional():
    """No YAML entry -> no limits; a partial entry throttles only its axes."""
    from praxis.config import ModelLimits

    assert FactsSheet().model_limits == []

    facts = FactsSheet(model_limits=[ModelLimits(model="m/one")])
    limits = resolve_model_limits(facts, "m/one")
    assert limits is not None
    assert limits.tpm is None and limits.itpm is None and limits.otpm is None
    assert limits.max_output is None


def test_model_limits_yaml_max_output_parses(tmp_path):
    """max_output is a separate per-request key alongside the rate axes."""
    p = tmp_path / "l.yaml"
    p.write_text(
        "model_limits:\n  - model: a/b\n    otpm: 1000\n    max_output: 2500\n",
        encoding="utf-8",
    )
    limits = resolve_model_limits(load_facts(str(p)), "a/b")
    assert limits is not None
    assert limits.otpm == 1000 and limits.max_output == 2500


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
            ModelLimits(model="cerebras/gpt-oss-120b", max_output=3000),
        ]
    )
    md = render_facts_sheet(facts)
    assert "groq/openai/gpt-oss-120b: 8000 TPM total" in md
    assert "groq/qwen/qwen3.8-27b: 7000 ITPM input, 1000 OTPM output" in md
    assert "cerebras/gpt-oss-120b: 3000 max output/request" in md
    assert "Sep 2026" in md
