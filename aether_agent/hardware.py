# aether_agent/hardware.py
# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Hardware detection + a curated local-model catalog + a best-fit picker.

``detect_resources`` is fail-soft (any probe error -> a conservative 8 GB / no-GPU
default) and takes injectable probes so tests never touch the real machine.
``best_fit`` never returns an oversized or license-flagged model as the auto-pick.
"""
from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    tag: str
    size_gb: float
    min_ram_gb: float
    gpu_pref: bool
    license_note: str | None
    blurb: str


# Ordered small -> large. license_note != None => never an auto-default.
CATALOG: tuple[ModelInfo, ...] = (
    ModelInfo("qwen2.5-coder:3b", 1.9, 4, False, None, "low-end, permissive (Apache-2.0)"),
    ModelInfo("gemma3n:e4b", 3.3, 6, False, "Gemma3 custom license", "light machines"),
    ModelInfo("qwen2.5-coder:7b", 4.7, 8, False, None, "universal, fits an 8 GB GPU"),
    ModelInfo("qwen3-coder:30b", 24.0, 24, True, None, "depth build, 256K ctx (Apache-2.0)"),
)
DEFAULT_FLOOR = "qwen2.5-coder:3b"


@dataclass(frozen=True)
class Resources:
    ram_gb: float
    vram_gb: float | None
    gpu: str | None


def detect_resources(*, ram_probe=None, vram_probe=None) -> Resources:
    """Detect RAM/VRAM. Fail-soft to (8 GB, no GPU). Probes injectable for tests."""
    try:
        ram = float((ram_probe or _ram_gb)())
    except Exception:  # noqa: BLE001 — detection must never crash startup
        ram = 8.0
    try:
        vram, gpu = (vram_probe or _vram)()
    except Exception:  # noqa: BLE001
        vram, gpu = None, None
    return Resources(ram_gb=ram, vram_gb=vram, gpu=gpu)


def best_fit(res: Resources) -> tuple[str, str]:
    """Largest permissive model that fits. Returns (tag, reason)."""
    if res.vram_gb and res.vram_gb > 0:
        eff, where = res.vram_gb, (res.gpu or "VRAM")
    else:
        eff, where = res.ram_gb * 0.7, "RAM"
    eligible = [m for m in CATALOG if m.license_note is None and m.min_ram_gb <= eff]
    if not eligible:
        return DEFAULT_FLOOR, f"~{eff:.0f} GB usable -> safe default {DEFAULT_FLOOR}"
    best = max(eligible, key=lambda m: m.size_gb)
    return best.tag, f"{where} ~{eff:.0f} GB -> {best.tag}"


# --- real probes (best-effort, platform-specific) --------------------------
def _ram_gb() -> float:
    sysname = platform.system()
    if sysname == "Windows":
        import ctypes

        class _Mem(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        m = _Mem()
        m.dwLength = ctypes.sizeof(_Mem)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullTotalPhys / 1024**3
    if sysname == "Darwin":
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=5)
        return int(out.strip()) / 1024**3
    # Linux / other: /proc/meminfo
    with open("/proc/meminfo", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024**2  # kB -> GB
    raise OSError("cannot read MemTotal")


def _vram() -> tuple[float | None, str | None]:
    if not shutil.which("nvidia-smi"):
        return None, None
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        timeout=5,
    ).decode("utf-8", "replace").strip()
    first = out.splitlines()[0]
    name, mib = (p.strip() for p in first.split(","))
    return float(mib) / 1024, name  # MiB -> GB


__all__ = ["ModelInfo", "CATALOG", "DEFAULT_FLOOR", "Resources", "detect_resources", "best_fit"]
