"""Hardware facts, policy, and budget loading for detected and configured hosts."""

from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import yaml

from praxis.hardware import HostHardware
from praxis.hardware import detect_host_hardware as _detect_host_hardware

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


def load_config(
    path: str | None = None,
    *,
    detect: bool | None = None,
    detected_host: HostHardware | None = None,
) -> HardwareProfile:
    """Load the effective target profile from env, detection, YAML, then defaults."""
    facts = load_facts(path, detect=detect, detected_host=detected_host)
    return HardwareProfile(
        cpu_only=facts.cpu_only,
        ram_gb=facts.ram_gb,
        gpu=facts.gpu,
        monthly_budget_usd=facts.monthly_budget_usd,
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
    cpu_cores: int | None = None
    gpu_name: str = "unknown"
    gpu_note: str = ""
    ram_note: str = ""
    storage_free_gb: int | None = None
    detected: bool = False
    detection_note: str = ""


@dataclass
class ModelLimits:
    """Per-model free-tier rate limits (all optional, all tokens/minute).

    ``tpm`` caps total tokens (input + output), ``itpm`` input tokens only,
    ``otpm`` output tokens only. A model with no entry — or a missing limit
    within its entry — is not throttled on that axis.

    ``max_output`` is a different kind of number: a per-request output
    ceiling, not a per-minute budget, and unverified until set in YAML.
    When absent, ``otpm`` doubles as the ceiling so no single request can
    spend more output than the whole minute's budget.
    """

    model: str
    tpm: int | None = None
    itpm: int | None = None
    otpm: int | None = None
    max_output: int | None = None
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
                max_output=_opt("max_output"),
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


def _detection_enabled(path: str | None, detect: bool | None) -> bool:
    """Should the host be probed? Env flag > explicit arg > default on.

    ``PRAXIS_CONFIG`` only selects the fallback facts file; it does NOT disable
    detection. Probing is off when an explicit ``detect`` is passed (CLI
    ``--hardware``/``--no-detect``), when ``PRAXIS_DETECT_HW`` is falsy, or
    when the caller passed an explicit file path (that file is the record).
    """
    if detect is not None:
        return detect
    configured = _env("DETECT_HW")
    if configured is not None:
        return _to_bool(configured)
    return path is None


def _resolve_host(
    path: str | None,
    detect: bool | None,
    detected_host: HostHardware | None,
) -> HostHardware | None:
    if detected_host is not None:
        return detected_host
    if not _detection_enabled(path, detect):
        return None
    try:
        return _detect_host_hardware()
    except Exception:
        return None


def _effective_str(
    env_name: str,
    preferred: Any,
    fallback: str | None,
    default: str,
) -> str:
    """Precedence: env > preferred (YAML record) > fallback (probe) > default."""
    env_value = _env(env_name)
    if env_value is not None:
        return env_value
    for candidate in (preferred, fallback):
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text and text.lower() != "unknown":
            return text
    return default


def _effective_int(
    env_name: str,
    preferred: Any,
    fallback: int | None,
    default: int,
) -> int:
    """Precedence: env > preferred (YAML record) > fallback (probe) > default."""
    env_value = _env(env_name)
    raw = env_value if env_value is not None else preferred
    if raw is None:
        raw = fallback
    return _to_int(raw, default=default)


def _effective_optional_int(
    env_name: str,
    preferred: Any,
    fallback: Any,
) -> int | None:
    """Precedence: env > preferred (YAML record) > fallback (probe)."""
    env_value = _env(env_name)
    raw = env_value if env_value is not None else preferred
    if raw is None:
        raw = fallback
    return _to_int(raw, default=0) if raw is not None else None


def load_facts(
    path: str | None = None,
    *,
    detect: bool | None = None,
    detected_host: HostHardware | None = None,
) -> FactsSheet:
    """Load env > YAML > detected host > defaults, tolerant of missing keys.

    The YAML file is the curated record, so it wins over the live probe for
    stable facts (OS, CPU, GPU identity, RAM ceiling). Free storage and the
    usable-RAM headroom are volatile: a live probe overrides the YAML snapshot,
    which goes stale. Env vars always win; the probe fills whatever neither
    specifies.
    """
    data = _load_yaml(path)
    host = _resolve_host(path, detect, detected_host)

    ram_gb = _effective_int("RAM_GB", data.get("ram_gb"), host.ram_gb if host else None, 8)
    # Volatile: the live probe beats the YAML snapshot (it goes stale).
    if _env("USABLE_RAM_GB") is not None:
        usable_ram_gb = _effective_optional_int("USABLE_RAM_GB", None, None)
    else:
        usable_ram_gb = _effective_optional_int(
            "USABLE_RAM_GB",
            host.usable_ram_gb if host else None,
            data.get("usable_ram_gb"),
        )
    if usable_ram_gb is None:
        usable_ram_gb = max(1, ram_gb - 4)

    # cpu_only/gpu: an explicit YAML value wins over the probe; a missing or
    # null key defers to detection (falling back to the CPU-only defaults).
    if _env("CPU_ONLY") is not None:
        cpu_only = _to_bool(_env("CPU_ONLY"))
    elif data.get("cpu_only") is not None:
        cpu_only = _to_bool(data["cpu_only"])
    elif host is not None and host.gpu is not None:
        cpu_only = not host.gpu
    else:
        cpu_only = True

    if _env("GPU") is not None:
        gpu = _to_bool(_env("GPU"))
    elif data.get("gpu") is not None:
        gpu = _to_bool(data["gpu"])
    elif host is not None and host.gpu is not None:
        gpu = host.gpu
    else:
        gpu = False

    if host is not None and host.gpu is not None:
        detected_gpu_name = host.gpu_name
        detected_gpu_note = host.gpu_note
    else:
        detected_gpu_name = None
        detected_gpu_note = None

    budget_raw = (
        _env("MONTHLY_BUDGET_USD")
        if _env("MONTHLY_BUDGET_USD") is not None
        else data.get("monthly_budget_usd", 15.0)
    )
    return FactsSheet(
        os=_effective_str("OS", data.get("os"), host.os if host else None, "Windows 11"),
        cpu=_effective_str("CPU", data.get("cpu"), host.cpu if host else None, "unknown"),
        cpu_only=cpu_only,
        gpu=gpu,
        gpu_name=_effective_str("GPU_NAME", data.get("gpu_name"), detected_gpu_name, "unknown"),
        ram_gb=ram_gb,
        usable_ram_gb=usable_ram_gb,
        monthly_budget_usd=_to_float(budget_raw, default=15.0),
        local_only=_to_bool(data.get("local_only", True)),
        provider_limits=_as_list(data.get("provider_limits")),
        model_limits=_load_model_limits(data.get("model_limits")),
        preferred_stack=_as_list(data.get("preferred_stack")),
        stack_notes=_as_list(data.get("stack_notes")),
        avoid=_as_list(data.get("avoid")),
        cpu_cores=_effective_optional_int(
            "CPU_CORES", data.get("cpu_cores"), host.cpu_cores if host else None
        ),
        gpu_note=_effective_str("GPU_NOTE", data.get("gpu_note"), detected_gpu_note, ""),
        ram_note=_effective_str(
            "RAM_NOTE", data.get("ram_note"), host.ram_note if host else None, ""
        ),
        # Volatile: the live probe beats the YAML snapshot.
        storage_free_gb=_effective_optional_int(
            "STORAGE_FREE_GB",
            host.storage_free_gb if host else None,
            data.get("storage_free_gb"),
        ),
        detected=host is not None,
        detection_note=host.detection_note if host else "",
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
    if facts.gpu:
        gpu_line = f"{facts.gpu_name} (GPU available)"
    elif facts.gpu_name.lower() != "unknown":
        gpu_line = f"{facts.gpu_name} (no GPU available; CPU-only)"
    else:
        gpu_line = "no GPU (CPU-only)"
    lines = [
        "## Hardware & budget facts (HARD CONSTRAINTS)",
        f"- OS: {facts.os}",
        f"- CPU: {facts.cpu}"
        + (f" ({facts.cpu_cores} logical cores)" if facts.cpu_cores is not None else ""),
        f"- CPU-only execution: {'yes' if facts.cpu_only else 'no'}",
        f"- GPU: {gpu_line}",
        f"- RAM: {facts.ram_gb} GB total, shared with the OS — plan for about "
        f"{facts.usable_ram_gb} GB usable headroom for the app",
    ]
    if facts.storage_free_gb is not None:
        lines.append(
            f"- Free storage: {facts.storage_free_gb} GB on the current working volume"
        )
    if facts.ram_note:
        lines.append(f"- RAM detail: {facts.ram_note}")
    if facts.gpu_note:
        lines.append(f"- GPU detail: {facts.gpu_note}")
    if facts.detected:
        lines.append(
            "- Host probe: succeeded - live free-storage and usable-RAM readings "
            "override this file's snapshot"
        )
    if facts.detection_note:
        lines.append(f"- Runtime note: {facts.detection_note}")
    lines.extend(
        [
            f"- Monthly budget: ${facts.monthly_budget_usd:.2f} (hard limit)",
            f"- Runs locally: "
            f"{'yes (no VPS unless justified)' if facts.local_only else 'no'}",
        ]
    )
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
            if entry.max_output is not None:
                parts.append(f"{entry.max_output} max output/request")
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
