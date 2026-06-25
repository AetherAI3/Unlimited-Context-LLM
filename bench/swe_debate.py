# bench/swe_debate.py
"""Complexity-gated propose↔critique reasoning escalation for the SWE-bench agent.

classify_complexity: cheap, deterministic gate (keywords + length). debate_patch: a
propose↔critique loop that drafts a fix, has a critic attack it, revises against the
objections (re-grounding each round), and applies the best candidate. Both reusable +
unit-testable with a mock chat; no live API required.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional

# Words that mark a genuinely hard instance — the kind a flat agent drops.
_HEAVY_RE = re.compile(
    r"\b(security|vulnerab|exploit|race condition|deadlock|concurren|"
    r"intermittent|edge case|regression|segfault|corrupt|workaround|"
    r"thread.?safe|memory leak|undefined behaviou?r)\b", re.I)
_HEAVY_LEN = 600   # problem statements this long (chars) are treated as heavy
_LIGHT_LEN = 120   # short + no hard keyword => light

# A "light" task is short AND obviously trivial (cosmetic). Length alone isn't enough —
# a short functional change ("update the parser...") is medium, not light.
_LIGHT_RE = re.compile(
    r"\b(typo|docstring|comment|rename|whitespace|spelling|lint|formatting|import order)\b", re.I)


def classify_complexity(problem: str, files_seen: int = 0) -> str:
    """Return 'light' | 'medium' | 'heavy'. Deterministic: hard keyword or a long/broad
    statement => heavy; short AND trivially-cosmetic => light; otherwise medium. `files_seen`
    (distinct files touched during investigation) nudges toward heavy when many are involved."""
    p = problem or ""
    if _HEAVY_RE.search(p) or len(p) >= _HEAVY_LEN or files_seen >= 4:
        return "heavy"
    if len(p) <= _LIGHT_LEN and _LIGHT_RE.search(p) and files_seen <= 1:
        return "light"
    return "medium"
