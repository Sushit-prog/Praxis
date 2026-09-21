"""Hardware profile + budget loading from YAML and environment variables."""

from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = "hardware_profile.yaml"

ENV_PREFIX = "PRAXIS_"


@dataclass
class HardwareProfile:
    """Target machine capabilities that constrain blueprint generation."""

    cpu_only: bool = True
    ram_gb: int = 8
    gpu: bool = False
    monthly_budget_usd: float = 15.0


def _env(name: str) -> str | None:
    return os.environ.get(f"{ENV_PREFIX}{name}")


# Characters that look harmless in a browser but break Windows consoles,
# terminals, and some tooling when model output is stored and re-printed.
_CHAR_REPLACEMENTS = {
    "\u2011": "-",  # non-breaking hyphen
    "\u2010": "-",  # unicode hyphen
    "\u2012": "-",  # figure dash
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash (kept often, but normalizing is safer on Windows)
    "\u2212": "-",  # minus sign
    "\u00a0": " ",  # no-break space
    "\u202f": " ",  # narrow no-break space
    "\u2007": " ",  # figure space
    "\u2009": " ",  # thin space
    "\u200b": "",  # zero-width space
    "\u2018": "'",  # left single smart quote
    "\u2019": "'",  # right single smart quote
    "\u201c": '"',  # left double smart quote
    "\u201d": '"',  # right double smart quote
}


def normalize_model_text(text: str | None) -> str:
    """Normalize model output before storing/printing it.

    Replaces non-breaking hyphens and spaces, smart quotes, and zero-width
    characters with their plain ASCII equivalents so stored artifacts survive
    Windows code pages, consoles, and round-trips through other tools.
    """
    if not text:
        return ""
    for src, dst in _CHAR_REPLACEMENTS.items():
        text = text.replace(src, dst)
    # NFKC folds remaining compatibility characters (full-width forms, etc.).
    return unicodedata.normalize("NFKC", text)


def _load_yaml(path: str | None) -> dict[str, Any]:
    path = path or _env("CONFIG") or DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def load_config(path: str | None = None) -> HardwareProfile:
    """Load a HardwareProfile, preferring env vars over YAML over defaults."""
    data = _load_yaml(path)

    cpu_only_raw = _env("CPU_ONLY") if _env("CPU_ONLY") is not None else data.get("cpu_only", True)
    ram_raw = _env("RAM_GB") if _env("RAM_GB") is not None else data.get("ram_gb", 8)
    gpu_raw = _env("GPU") if _env("GPU") is not None else data.get("gpu", False)
    budget_raw = (
        _env("MONTHLY_BUDGET_USD")
        if _env("MONTHLY_BUDGET_USD") is not None
        else data.get("monthly_budget_usd", 15.0)
    )

    return HardwareProfile(
        cpu_only=_to_bool(cpu_only_raw),
        ram_gb=_to_int(ram_raw, default=8),
        gpu=_to_bool(gpu_raw),
        monthly_budget_usd=_to_float(budget_raw, default=15.0),
    )


@dataclass
class FactsSheet:
    """The full environment facts a design must respect.

    A superset of :class:`HardwareProfile`: the typed profile drives
    feasibility scoring, while the facts sheet (OS, usable RAM headroom,
    provider free-tier limits, stack notes) is rendered into every design pass
    as a set of hard constraints. Missing YAML keys fall back to defaults.
    """

    os: str = "Windows 11"
    cpu: str = "unknown"
    cpu_only: bool = True
    gpu: bool = False
    ram_gb: int = 8
    usable_ram_gb: int = 4
    monthly_budget_usd: float = 15.0
    local_only: bool = True
    provider_limits: list[str] = field(default_factory=list)
    model_limits: list[ModelLimits] = field(default_factory=list)
    preferred_stack: list[str] = field(default_factory=list)
    stack_notes: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)


@dataclass
class ModelLimits:
    """Per-model free-tier rate limits (all optional, all tokens/minute).

    ``tpm`` caps total tokens (input + output), ``itpm`` input tokens only,
    ``otpm`` output tokens only. A model with no entry — or a missing limit
    within its entry — is not throttled on that axis.
    """

    model: str
    tpm: int | None = None
    itpm: int | None = None
    otpm: int | None = None
    note: str = ""


def _load_model_limits(value: Any) -> list[ModelLimits]:
    """Parse the YAML ``model_limits`` list into ModelLimits entries."""
    if not isinstance(value, list):
        return []
    limits: list[ModelLimits] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        model = str(item.get("model") or "").strip()
        if not model:
            continue

        def _opt(name: str, _item: dict[str, Any] = item) -> int | None:
            raw = _item.get(name)
            if raw is None:
                return None
            try:
                return max(1, int(raw))
            except (TypeError, ValueError):
                return None

        limits.append(
            ModelLimits(
                model=model,
                tpm=_opt("tpm"),
                itpm=_opt("itpm"),
                otpm=_opt("otpm"),
                note=str(item.get("note") or ""),
            )
        )
    return limits


def _as_list(value: Any) -> list[str]:
    """Coerce a YAML scalar or sequence into a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def load_facts(path: str | None = None) -> FactsSheet:
    """Load the facts sheet: env vars > YAML > defaults, tolerant of missing keys."""
    data = _load_yaml(path)

    ram_raw = _env("RAM_GB") if _env("RAM_GB") is not None else data.get("ram_gb", 8)
    cpu_only_raw = _env("CPU_ONLY") if _env("CPU_ONLY") is not None else data.get("cpu_only", True)
    gpu_raw = _env("GPU") if _env("GPU") is not None else data.get("gpu", False)
    budget_raw = (
        _env("MONTHLY_BUDGET_USD")
        if _env("MONTHLY_BUDGET_USD") is not None
        else data.get("monthly_budget_usd", 15.0)
    )

    ram_gb = _to_int(ram_raw, default=8)
    default_headroom = max(1, ram_gb - 4)
    return FactsSheet(
        os=str(data.get("os") or "Windows 11"),
        cpu=str(data.get("cpu") or "unknown"),
        cpu_only=_to_bool(cpu_only_raw),
        gpu=_to_bool(gpu_raw),
        ram_gb=ram_gb,
        usable_ram_gb=_to_int(data.get("usable_ram_gb"), default=default_headroom),
        monthly_budget_usd=_to_float(budget_raw, default=15.0),
        local_only=_to_bool(data.get("local_only", True)),
        provider_limits=_as_list(data.get("provider_limits")),
        model_limits=_load_model_limits(data.get("model_limits")),
        preferred_stack=_as_list(data.get("preferred_stack")),
        stack_notes=_as_list(data.get("stack_notes")),
        avoid=_as_list(data.get("avoid")),
    )


def resolve_model_limits(facts: FactsSheet, model: str) -> ModelLimits | None:
    """The limits entry for ``model``, or None when the model is unthrottled.

    Matches the full litellm id first (``groq/openai/gpt-oss-120b``); falls
    back to a suffix match so a YAML entry written as ``openai/gpt-oss-120b``
    still covers the fully qualified model.
    """
    for entry in facts.model_limits:
        if entry.model == model:
            return entry
    for entry in facts.model_limits:
        if model.endswith(entry.model) or entry.model.endswith(model):
            return entry
    return None


def render_facts_sheet(facts: FactsSheet) -> str:
    """Render the facts sheet as a hard-constraints block for every design pass."""
    gpu_line = "GPU available" if facts.gpu else "no GPU (CPU-only)"
    lines = [
        "## Hardware & budget facts (HARD CONSTRAINTS)",
        f"- OS: {facts.os}",
        f"- CPU: {facts.cpu} — {gpu_line}",
        f"- RAM: {facts.ram_gb} GB total, shared with the OS — plan for about "
        f"{facts.usable_ram_gb} GB usable headroom for the app",
        f"- Monthly budget: ${facts.monthly_budget_usd:.2f} (hard limit)",
        f"- Runs locally: {'yes (no VPS unless justified)' if facts.local_only else 'no'}",
    ]
    if facts.provider_limits:
        lines.append("- Provider free-tier limits:")
        lines.extend(f"  - {limit}" for limit in facts.provider_limits)
    if facts.model_limits:
        lines.append("- Per-model rate limits (observed on the free tier, Sep 2026, may change):")
        for entry in facts.model_limits:
            parts = []
            if entry.tpm is not None:
                parts.append(f"{entry.tpm} TPM total")
            if entry.itpm is not None:
                parts.append(f"{entry.itpm} ITPM input")
            if entry.otpm is not None:
                parts.append(f"{entry.otpm} OTPM output")
            note = f" ({entry.note})" if entry.note else ""
            lines.append(f"  - {entry.model}: {', '.join(parts)}{note}")
    if facts.preferred_stack:
        lines.append(f"- Preferred stack: {', '.join(facts.preferred_stack)}")
    if facts.stack_notes:
        lines.append("- Stack notes:")
        lines.extend(f"  - {note}" for note in facts.stack_notes)
    if facts.avoid:
        lines.append(f"- Avoid: {', '.join(facts.avoid)}")
    lines.append(
        "- Rule: never state prices, model sizes, or library capabilities that are "
        "not in this facts sheet or the source paper without labelling them "
        '"UNVERIFIED: check before relying".'
    )
    return "\n".join(lines)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _to_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _to_float(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)
