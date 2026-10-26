# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Cost accounting for the latent-transfer bench: device facts, peak memory, FLOP estimates.

Pure Python at import time; torch and transformers are only imported inside ``device_info``.
"""
from __future__ import annotations

import os
import platform
import sys
from typing import Any


def _version(module: str) -> str | None:
    try:
        return __import__(module).__version__
    except Exception:  # noqa: BLE001 - absent or broken optional dependency
        return None


def device_info() -> dict[str, Any]:
    """Facts about the machine and software stack the numbers were measured on (CPU only)."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "torch": None,
        "torch_threads": None,
        "cuda_used": False,
        "device": "cpu",
        "transformers": _version("transformers"),
        "numpy": _version("numpy"),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_threads"] = int(torch.get_num_threads())
    except Exception:  # noqa: BLE001 - torch is optional for the pure-python helpers
        pass
    return info


def _windows_peak_working_set() -> int | None:
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return None
    return int(counters.PeakWorkingSetSize)


def peak_rss_bytes() -> int | None:
    """Peak resident set size of this process in bytes, or ``None`` if it cannot be measured."""
    try:
        import psutil

        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None) if sys.platform == "win32" else None
        if peak:
            return int(peak)
    except Exception:  # noqa: BLE001 - fall through to the OS-specific probes
        pass
    if sys.platform == "win32":
        try:
            peak = _windows_peak_working_set()
            if peak:
                return peak
        except Exception:  # noqa: BLE001
            pass
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        return int(peak) if peak else None
    except Exception:  # noqa: BLE001
        return None


def forward_flops(blocks: int, head: int, tokens: int, head_tokens: int) -> int:
    """Estimated forward FLOPs: ``2 * blocks * tokens + 2 * head * head_tokens``.

    ``blocks`` is the transformer-block parameter count (input embeddings excluded), ``head`` the
    LM-head matmul parameter count, ``tokens`` the total sequence tokens pushed through the
    blocks and ``head_tokens`` the positions that go through the LM head. Attention-score FLOPs
    are ignored, so this is an estimate (a lower bound for long sequences).
    """
    return 2 * blocks * tokens + 2 * head * head_tokens


def train_flops(blocks: int, head: int, tokens: int, head_tokens: int) -> int:
    """Estimated forward+backward FLOPs, ``2 * forward_flops``.

    This is an estimate: the backward pass computes activation gradients only (the base weights are
    frozen, so the weight-gradient matmuls are skipped), which costs about one more forward pass.
    Attention-score FLOPs and the (small) projector are ignored.
    """
    return 2 * forward_flops(blocks, head, tokens, head_tokens)
