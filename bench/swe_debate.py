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
import subprocess
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


from bench.swe_parse import _parse_blocks  # shared parser module (no swe_eval import)

_PROPOSE_SYS = (
    "You are the PROPOSER. Produce a minimal fix as SEARCH/REPLACE edit blocks, EXACTLY:\n"
    "<path>\n<<<<<<< SEARCH\n<exact current lines>\n=======\n<replacement>\n>>>>>>> REPLACE\n"
    "Output ONLY the blocks. Use the grounded context to get SEARCH text right.")
_CRITIQUE_SYS = (
    "You are the CRITIC. Attack the proposed fix: does it address the FAILING behaviour? "
    "Wrong file/location? Missed edge case? Will the repo's tests still fail? "
    "Reply JSON ONLY: {\"accept\": true|false, \"objections\": [\"...\"]}. Accept only if the "
    "fix is correct and complete.")


def _parse_verdict(text: str) -> dict:
    """Tolerant parse of the critic's JSON verdict; unparseable => reject (keep iterating)."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            return {"accept": bool(d.get("accept")), "objections": list(d.get("objections") or [])}
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return {"accept": False, "objections": ["unparseable verdict"]}


def debate_patch(chat: Any, tools: Any, ground_fn: Callable[[str], str], problem: str, *,
                 rounds: int = 2, max_output_tokens: int = 4096,
                 account: Optional[Callable[[dict], bool]] = None) -> dict:
    """Propose↔critique loop. The proposer drafts SEARCH/REPLACE edits (grounded per round on
    the current focus); the critic accepts/rejects with objections; on reject the proposer
    revises against them and re-grounds. Applies the latest candidate via tools.edit_file each
    round (so current_patch() reflects the best attempt). Returns
    {accepted, applied, rounds_used}. account(out) (optional) tallies cost; if it returns
    False (cost spike) the loop stops."""
    focus = problem
    applied_total = 0
    accepted = False
    used = 0
    for used in range(1, rounds + 1):
        ground = ground_fn(focus) or ""
        prop_msgs = [{"role": "system", "content": _PROPOSE_SYS},
                     {"role": "user", "content": f"Problem:\n{problem}\n\nContext:\n{ground}\n\nPropose the fix."}]
        out = chat.chat(prop_msgs, tools=None, max_tokens=max_output_tokens)
        if account is not None and not account(out):
            break
        # Each round is a fresh attempt from the base tree: discard the prior round's edits so
        # this candidate's SEARCH blocks (which always quote the ORIGINAL file) apply cleanly,
        # and current_patch() reflects ONLY the latest candidate (the best attempt so far).
        subprocess.run(["git", "checkout", "-q", "--", "."], cwd=tools.root,
                       capture_output=True, check=True)
        applied_here = 0
        for path, search, replace in _parse_blocks(out.get("content") or ""):
            res = tools.edit_file(path, search, replace)
            if isinstance(res, dict) and res.get("ok"):
                applied_here += 1
                applied_total += 1
        if applied_here == 0:
            focus = problem + "\n\nYour previous output had no applicable edit blocks. " \
                    "Resend valid SEARCH/REPLACE blocks."
            continue
        patch = tools.current_patch()
        crit_msgs = [{"role": "system", "content": _CRITIQUE_SYS},
                     {"role": "user", "content": f"CRITIQUE this fix for:\n{problem}\n\nProposed patch:\n{patch}"}]
        out = chat.chat(crit_msgs, tools=None, max_tokens=max_output_tokens)
        if account is not None and not account(out):
            break
        verdict = _parse_verdict(out.get("content") or "")
        if verdict["accept"] and applied_here:
            accepted = True
            break
        focus = problem + "\n\nFix these objections: " + "; ".join(verdict["objections"])
    return {"accepted": accepted, "applied": applied_total, "rounds_used": used}
