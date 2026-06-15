# aether_agent/progress.py
# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Pure, cp1252-safe render helpers for the Ollama-managed terminal UI.

No IO, no ANSI cursor control here (callers own the terminal) — just strings, so
every function is trivially unit-testable and safe on a Windows cp1252 console.
"""
from __future__ import annotations

_BLOCK = "#"
_EMPTY = "-"
_SPINNER = ("|", "/", "-", "\\")


def progress_bar(frac: float, width: int = 24) -> str:
    """Render ``[####----] 51%``. ``frac`` is clamped to [0, 1]."""
    try:
        f = float(frac)
    except (TypeError, ValueError):
        f = 0.0
    f = 0.0 if f < 0 else 1.0 if f > 1 else f
    filled = round(f * width)
    bar = _BLOCK * filled + _EMPTY * (width - filled)
    return f"[{bar}] {int(round(f * 100))}%"


def fmt_bytes(n: float) -> str:
    """Human bytes: ``4.7 GB``. Whole-byte values render without a decimal."""
    try:
        val = float(n or 0)
    except (TypeError, ValueError):
        val = 0.0
    if val < 1024:
        return f"{int(val)} B"
    for unit in ("KB", "MB", "GB", "TB"):
        val /= 1024
        if val < 1024 or unit == "TB":
            return f"{val:.1f} {unit}"
    return f"{val:.1f} TB"


def spinner_frame(i: int) -> str:
    return _SPINNER[i % len(_SPINNER)]


def step(label: str, ok: bool | None = None) -> str:
    """A step line: ``... label`` (pending) · ``[ok] label`` · ``[x] label``."""
    mark = "..." if ok is None else ("[ok]" if ok else "[x]")
    return f"  {mark} {label}"


def pull_line(tag: str, completed: float, total: float) -> str:
    """Single-line live pull status: ``tag  2.0 GB / 4.0 GB  [####----] 50%``."""
    frac = (float(completed) / float(total)) if total else 0.0
    return f"{tag}   {fmt_bytes(completed)} / {fmt_bytes(total)}   {progress_bar(frac)}"


__all__ = ["progress_bar", "fmt_bytes", "spinner_frame", "step", "pull_line"]
