"""Hardware profile + budget loading from YAML and environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
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

    cpu_only_raw = (
        _env("CPU_ONLY") if _env("CPU_ONLY") is not None else data.get("cpu_only", True)
    )
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
