"""Best-effort host hardware detection using only the Python standard library."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

UNKNOWN = "unknown"
_GIB = 1024**3
_GPU_CLASS_MARKERS = (
    "vga compatible controller",
    "3d controller",
    "display controller",
)
_DISCRETE_GPU_MARKERS = (
    "nvidia",
    "geforce",
    "quadro",
    "tesla",
    "radeon rx",
    "radeon pro",
    "arc gpu",
)
_IGNORED_GPU_NAMES = (
    "microsoft basic display adapter",
    "remote display adapter",
    "parsec virtual display",
)


@dataclass(frozen=True)
class HostHardware:
    """Hardware facts detected on the machine running Praxis."""

    os: str
    cpu: str
    cpu_cores: int | None
    gpu_name: str
    gpu: bool | None
    gpu_note: str
    ram_gb: int | None
    ram_reported_gb: float | None
    usable_ram_gb: int | None
    ram_note: str
    storage_free_gb: int | None
    detection_note: str = ""


def _clean_value(value: str | None) -> str:
    return " ".join((value or "").split()).strip()


def _command_output(command: list[str], timeout: float = 3.0) -> str | None:
    try:
        kwargs = {
            "capture_output": True,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "timeout": timeout,
            "check": False,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        completed = subprocess.run(command, **kwargs)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _windows_registry_value(value_name: str) -> str | None:
    try:
        import winreg
    except ImportError:
        return None
    key_path = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion"
    for access in (winreg.KEY_READ | winreg.KEY_WOW64_64KEY, winreg.KEY_READ):
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, access) as key:
                value, _ = winreg.QueryValueEx(key, value_name)
        except OSError:
            continue
        return _clean_value(str(value))
    return None


def _windows_os_name() -> str:
    product_name = _windows_registry_value("ProductName")
    build = _windows_registry_value("CurrentBuildNumber") or ""
    if product_name:
        if build.isdigit() and int(build) >= 22000:
            product_name = product_name.replace("Windows 10", "Windows 11")
        return product_name
    version = platform.version()
    return f"Windows {version}" if version else "Windows"


def _windows_cpu_name() -> str | None:
    return _windows_registry_value("ProcessorNameString")


def _linux_cpu_name() -> str | None:
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        for line in cpuinfo.splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip().lower() in {"model name", "hardware", "processor"}:
                cleaned = _clean_value(value)
                if cleaned:
                    return cleaned
    except OSError:
        return None
    return None


def _macos_cpu_name() -> str | None:
    return _clean_value(_command_output(["sysctl", "-n", "machdep.cpu.brand_string"]))


def _cpu_name(system: str) -> str:
    if system == "Windows":
        detected = _windows_cpu_name()
    elif system == "Linux":
        detected = _linux_cpu_name()
    elif system == "Darwin":
        detected = _macos_cpu_name()
    else:
        detected = None
    fallback = _clean_value(platform.processor()) or _clean_value(platform.machine())
    return detected or fallback or UNKNOWN


def _windows_ram_bytes() -> int | None:
    try:
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullTotalPhys)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _linux_ram_bytes() -> int | None:
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace")
        for line in meminfo.splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() == "MemTotal":
                parts = value.split()
                if parts and parts[0].isdigit():
                    return int(parts[0]) * 1024
    except (OSError, TypeError, ValueError):
        return None
    return None


def _macos_ram_bytes() -> int | None:
    output = _command_output(["sysctl", "-n", "hw.memsize"])
    if output is None:
        return None
    try:
        return int(output.strip())
    except ValueError:
        return None


def _ram_bytes(system: str) -> int | None:
    if system == "Windows":
        return _windows_ram_bytes()
    if system == "Linux":
        return _linux_ram_bytes()
    if system == "Darwin":
        return _macos_ram_bytes()
    return None


def _unique_names(names: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for raw_name in names:
        name = _clean_value(raw_name)
        if not name or name.lower() in seen:
            continue
        if any(ignored in name.lower() for ignored in _IGNORED_GPU_NAMES):
            continue
        seen.add(name.lower())
        unique.append(name)
    return unique


def _windows_gpu_names() -> tuple[list[str], bool]:
    output = _command_output(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
        ]
    )
    if output is None:
        return [], False
    return _unique_names(output.splitlines()), True


def _linux_gpu_names() -> tuple[list[str], bool]:
    executable = shutil.which("lspci")
    if not executable:
        return [], False
    output = _command_output([executable, "-mm"])
    if output is None:
        return [], False
    names: list[str] = []
    for line in output.splitlines():
        if not any(marker in line.lower() for marker in _GPU_CLASS_MARKERS):
            continue
        parts = line.split('"', 2)
        description = parts[1].strip() if len(parts) >= 2 else line.strip()
        if ":" in description:
            description = description.split(":", 1)[1].strip()
        if description:
            names.append(description)
    return _unique_names(names), True


def _macos_gpu_names() -> tuple[list[str], bool]:
    output = _command_output(["system_profiler", "SPDisplaysDataType", "-json"])
    if output is None:
        return [], False
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return [], False
    names = [
        str(item.get("sppci_model") or "")
        for item in data.get("SPDisplaysDataType", [])
        if isinstance(item, dict)
    ]
    return _unique_names(names), True


def _gpu_inventory(system: str) -> tuple[list[str], bool]:
    if system == "Windows":
        return _windows_gpu_names()
    if system == "Linux":
        return _linux_gpu_names()
    if system == "Darwin":
        return _macos_gpu_names()
    return [], False


def _classify_gpu(names: list[str], probed: bool) -> tuple[str, bool | None, str]:
    if not names:
        if not probed:
            return UNKNOWN, None, "GPU inventory unavailable; no GPU assumed."
        return UNKNOWN, False, "No GPU detected; CPU-only execution."
    name = ", ".join(names)
    usable = any(marker in name.lower() for marker in _DISCRETE_GPU_MARKERS)
    if usable:
        return name, True, "Compute-capable GPU detected; verify runtime and VRAM before use."
    return name, False, "Integrated/shared GPU detected; CPU-only execution."


def _is_wsl() -> bool:
    try:
        version = Path("/proc/version").read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return False
    return "microsoft" in version or "wsl" in version


def _runtime_note(system: str, wsl: bool) -> str:
    if system == "Linux" and wsl:
        return "WSL guest detected; Windows host specifications are not visible from this runtime."
    if system == "Linux" and Path("/.dockerenv").exists():
        return "Container detected; specifications describe the container, not the host machine."
    return ""


def _storage_free_gb() -> int | None:
    try:
        return int(shutil.disk_usage(Path.cwd()).free // _GIB)
    except OSError:
        return None


def _format_reported_gb(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def detect_host_hardware() -> HostHardware:
    """Detect current-host facts without raising when a platform probe fails."""
    system = platform.system()
    wsl = system == "Linux" and _is_wsl()
    if wsl:
        os_name = "WSL2 (Linux guest)"
    elif system == "Windows":
        os_name = _windows_os_name()
    elif system == "Darwin":
        release = platform.mac_ver()[0] or platform.release()
        os_name = f"macOS {release}".strip()
    else:
        release = platform.release()
        os_name = f"{system or 'unknown OS'} {release}".strip()

    ram_bytes = _ram_bytes(system)
    ram_gb = max(1, int(round(ram_bytes / _GIB))) if ram_bytes else None
    ram_reported_gb = ram_bytes / _GIB if ram_bytes else None
    usable_ram_gb = max(1, ram_gb - 4) if ram_gb else None
    ram_note = ""
    if ram_gb is not None and ram_reported_gb is not None:
        ram_note = (
            f"{ram_gb} GB total / {_format_reported_gb(ram_reported_gb)} GB reported / "
            f"about {usable_ram_gb} GB app headroom"
        )

    gpu_names, gpu_probed = _gpu_inventory(system)
    gpu_name, gpu, gpu_note = _classify_gpu(gpu_names, gpu_probed)
    return HostHardware(
        os=os_name,
        cpu=_cpu_name(system),
        cpu_cores=os.cpu_count(),
        gpu_name=gpu_name,
        gpu=gpu,
        gpu_note=gpu_note,
        ram_gb=ram_gb,
        ram_reported_gb=ram_reported_gb,
        usable_ram_gb=usable_ram_gb,
        ram_note=ram_note,
        storage_free_gb=_storage_free_gb(),
        detection_note=_runtime_note(system, wsl),
    )


def format_host_hardware(host: HostHardware) -> str:
    """Render detected host facts as a compact human-readable block."""
    gpu_state = "unknown" if host.gpu is None else ("available" if host.gpu else "CPU-only")
    lines = [
        f"  OS: {host.os}",
        f"  CPU: {host.cpu}"
        + (f" ({host.cpu_cores} logical cores)" if host.cpu_cores is not None else ""),
        f"  GPU: {host.gpu_name} ({gpu_state})",
        f"  RAM: {host.ram_gb if host.ram_gb is not None else 'unknown'} GB total",
        f"  Free storage: "
        f"{host.storage_free_gb if host.storage_free_gb is not None else 'unknown'} GB",
    ]
    if host.ram_note:
        lines.append(f"  RAM detail: {host.ram_note}")
    if host.gpu_note:
        lines.append(f"  GPU detail: {host.gpu_note}")
    if host.detection_note:
        lines.append(f"  Runtime: {host.detection_note}")
    return "\n".join(lines)
