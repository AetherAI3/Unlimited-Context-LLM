# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""The `/agents` column dashboard - a pure, cp1252-safe renderer.

`render_agents_columns(rows, active, width)` lays out agent cards side by side,
N-per-row to the terminal width. No ANSI/color here (keeps width math correct and
the output testable); the active agent is marked with `*`.
"""
from __future__ import annotations

from typing import Any

_CARD_W = 22


def render_agents_columns(rows: list[dict], active: str = "", width: int = 80) -> str:
    if not rows:
        return "(no agents yet - create one with /new-agent <name>)"
    cards = [_card(r, active) for r in rows]
    per_row = max(1, width // (_CARD_W + 2))
    lines: list[str] = []
    for i in range(0, len(cards), per_row):
        group = cards[i:i + per_row]
        height = max(len(c) for c in group)
        for c in group:
            c.extend([""] * (height - len(c)))
        for row_line in range(height):
            lines.append("  ".join(c[row_line].ljust(_CARD_W) for c in group).rstrip())
        lines.append("")  # blank line between card rows
    return "\n".join(lines).rstrip()


def _card(r: dict[str, Any], active: str) -> list[str]:
    name = str(r.get("name", ""))
    mark = "*" if name and name == active else " "
    model = str(r.get("model", ""))
    pool = r.get("pool_gb", 0)
    n = r.get("n_commands", 0)
    return [
        f"{mark} {name}"[:_CARD_W],
        f"  {model}"[:_CARD_W],
        f"  {pool} GB . {n} cmds"[:_CARD_W],
    ]


__all__ = ["render_agents_columns"]
