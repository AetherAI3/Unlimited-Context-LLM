# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Startup splash — the AETHER brand banner above a one-line status column.

Mirror of aether-code ``src/ui/splash.ts`` + ``src/ui/logo.ts``. The brand is a
static "AETHER" wordmark beside a cloud glyph, so the terminal cold-starts
instantly (no figlet font load, no truecolor gradient dependency).

PURE ASCII on the wire (lockstep with the protocol's cp1252 rule): the wordmark
and cloud use only 7-bit characters so the banner can be written straight to a
legacy Windows console (cp1252) without raising ``UnicodeEncodeError``. The TS
host renders its own gradient box-drawing brand; the Python REPL stays
encoding-safe so ``aether`` never crashes on first paint. We never emit color
codes. ``render_splash(version, model, backend)`` returns the full banner string
the REPL prints once at startup.
"""
from __future__ import annotations

# The cloud glyph (ASCII; kept beside the wordmark so brand + splash share one source).
_CLOUD = (
    "   .--.   ",
    " .(    ). ",
    "(___.__)__)",
    "          ",
    "          ",
)

# 5-row block "AETHER" (ASCII figlet-style). Each row is a literal 7-bit string.
_WORDMARK = (
    "  /\\   ___ _____ _  _ ___ ___ ",
    " /  \\ | __|_   _| || | __| _ \\",
    "/ /\\ \\| _|  | | | __ | _||   /",
    "/_/  \\_\\___| |_| |_||_|___|_|_\\",
    "                              ",
)

_GAP = "  "


def _brand_lines() -> list[str]:
    """The cloud glyph beside the AETHER wordmark, fused into one banner block."""
    rows = max(len(_CLOUD), len(_WORDMARK))
    out: list[str] = []
    cloud_w = max(len(c) for c in _CLOUD)
    for i in range(rows):
        cloud = _CLOUD[i] if i < len(_CLOUD) else " " * cloud_w
        word = _WORDMARK[i] if i < len(_WORDMARK) else ""
        out.append(f"{cloud:<{cloud_w}}{_GAP}{word}")
    return out


def status_line(model: str, backend: str) -> str:
    """The single status line under the brand: ``model · backend · /help …``.

    ``model`` is the active model tag (or ``"auto"``); ``backend`` is the brain
    that will serve turns this session (``"cloud"`` / ``"local Ollama"``).
    """
    m = model or "auto"
    b = backend or "auto"
    return f"{m} · {b} · /help for commands"


def render_splash(version: str, model: str, backend: str) -> str:
    """Render the full startup splash: the brand banner, a blank line, a version
    line, then the one-line status column. Returns a single string (no trailing
    newline) the REPL prints once."""
    lines = list(_brand_lines())
    lines.append("")
    lines.append(f"v{version}")
    lines.append(status_line(model, backend))
    return "\n".join(lines)


__all__ = ["render_splash", "status_line"]
