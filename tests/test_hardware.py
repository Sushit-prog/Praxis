"""Tests for stdlib host hardware detection and GPU classification."""

from __future__ import annotations

import pytest

from praxis import hardware


def _gib(value: float) -> int:
    return int(value * hardware._GIB)


def test_detect_windows_host_records_integrated_gpu_as_cpu_only(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hardware, "_windows_os_name", lambda: "Windows 11")
    monkeypatch.setattr(hardware, "_cpu_name", lambda _system: "Intel Core i5-1235U")
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 12)
    monkeypatch.setattr(hardware, "_ram_bytes", lambda _system: _gib(7.68))
    monkeypatch.setattr(
        hardware,
        "_gpu_inventory",
        lambda _system: (["Intel Iris Xe Graphics"], True),
    )
    monkeypatch.setattr(hardware, "_storage_free_gb", lambda: 46)

    host = hardware.detect_host_hardware()

    assert host.os == "Windows 11"
    assert host.cpu == "Intel Core i5-1235U"
    assert host.cpu_cores == 12
    assert host.gpu_name == "Intel Iris Xe Graphics"
    assert host.gpu is False
    assert host.ram_gb == 8
    assert host.ram_reported_gb == pytest.approx(7.68)
    assert host.usable_ram_gb == 4
    assert host.storage_free_gb == 46
    assert "7.68 GB reported" in host.ram_note
    assert "CPU-only" in host.gpu_note


def test_detect_wsl_labels_guest_and_does_not_claim_host_inventory(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hardware, "_is_wsl", lambda: True)
    monkeypatch.setattr(hardware, "_cpu_name", lambda _system: "AMD Ryzen 7")
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(hardware, "_ram_bytes", lambda _system: _gib(8))
    monkeypatch.setattr(
        hardware,
        "_gpu_inventory",
        lambda _system: (["Intel Iris Xe Graphics"], True),
    )
    monkeypatch.setattr(hardware, "_storage_free_gb", lambda: 20)

    host = hardware.detect_host_hardware()

    assert host.os == "WSL2 (Linux guest)"
    assert "Windows host specifications are not visible" in host.detection_note
    assert host.gpu is False


def test_discrete_gpu_is_marked_available():
    name, available, note = hardware._classify_gpu(["NVIDIA GeForce RTX 4070"], True)

    assert name == "NVIDIA GeForce RTX 4070"
    assert available is True
    assert "verify runtime and VRAM" in note


def test_failed_probes_degrade_to_unknown(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hardware, "_is_wsl", lambda: False)
    monkeypatch.setattr(hardware, "_cpu_name", lambda _system: "unknown")
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: None)
    monkeypatch.setattr(hardware, "_ram_bytes", lambda _system: None)
    monkeypatch.setattr(hardware, "_gpu_inventory", lambda _system: ([], False))
    monkeypatch.setattr(hardware, "_storage_free_gb", lambda: None)

    host = hardware.detect_host_hardware()

    assert host.ram_gb is None
    assert host.usable_ram_gb is None
    assert host.gpu is None
    assert "unavailable" in host.gpu_note


def test_platform_gpu_parsers_extract_adapter_names(monkeypatch):
    monkeypatch.setattr(
        hardware,
        "_command_output",
        lambda _command, timeout=3.0: "Intel Iris Xe Graphics\nMicrosoft Basic Display Adapter",
    )
    assert hardware._windows_gpu_names() == (["Intel Iris Xe Graphics"], True)

    monkeypatch.setattr(hardware.shutil, "which", lambda _name: "/usr/bin/lspci")
    monkeypatch.setattr(
        hardware,
        "_command_output",
        lambda _command, timeout=3.0: (
            '00:02.0 "VGA compatible controller: Intel Corporation Device 9a49"\n'
            '00:01.0 "SCSI storage controller: Intel Corporation Device 1234"\n'
        ),
    )
    assert hardware._linux_gpu_names() == (
        ["Intel Corporation Device 9a49"],
        True,
    )

    monkeypatch.setattr(
        hardware,
        "_command_output",
        lambda _command, timeout=3.0: (
            '{"SPDisplaysDataType":[{"sppci_model":"Apple M3 GPU"}]}'
        ),
    )
    assert hardware._macos_gpu_names() == (["Apple M3 GPU"], True)


def test_format_host_hardware_includes_runtime_details(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hardware, "_is_wsl", lambda: True)
    monkeypatch.setattr(hardware, "_cpu_name", lambda _system: "Intel Core i5")
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(hardware, "_ram_bytes", lambda _system: _gib(7.68))
    monkeypatch.setattr(
        hardware,
        "_gpu_inventory",
        lambda _system: (["Intel Iris Xe Graphics"], True),
    )
    monkeypatch.setattr(hardware, "_storage_free_gb", lambda: 46)

    text = hardware.format_host_hardware(hardware.detect_host_hardware())

    assert "OS: WSL2 (Linux guest)" in text
    assert "8 logical cores" in text
    assert "GPU: Intel Iris Xe Graphics (CPU-only)" in text
    assert "Free storage: 46 GB" in text
    assert "Windows host specifications are not visible" in text
