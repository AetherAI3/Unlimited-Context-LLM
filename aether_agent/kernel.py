"""
Autonomy + checkpoint kernel — the part that makes multi-step runs real instead
of error-compounding.

- Test-gate: advance only on green (execution = truth).
- Checkpoints: git commit per verified step; revert on regression.
- Stuck detector: no test progress over N iterations -> change strategy / escalate.
- Escalation hook: a genuinely hard step can call a frontier model for THAT step
  (the funnel). Wired but OFF by default (a no-op that makes the agent re-strategize).
"""

from __future__ import annotations

import re
import subprocess
from typing import Callable, Optional


def tests_pass(tool_output: str) -> bool:
    """Green iff the test command exited 0 (our run wrapper prefixes [exit N])."""
    return tool_output.lstrip().startswith("[exit 0]")


def parse_fail_count(tool_output: str) -> Optional[int]:
    """Pytest-style '<n> failed' if present, else None (unknown)."""
    m = re.search(r"(\d+)\s+failed", tool_output)
    return int(m.group(1)) if m else None


class StuckDetector:
    """True once test failures stop improving over `patience` checks."""

    def __init__(self, patience: int = 4):
        self.patience = patience
        self._last: Optional[int] = None
        self._stall = 0

    def update(self, fail_count: Optional[int]) -> bool:
        if fail_count is None:
            return False
        if self._last is not None and fail_count >= self._last:
            self._stall += 1
        else:
            self._stall = 0
        self._last = fail_count
        return self._stall >= self.patience


def git_checkpoint(cwd: str, message: str) -> None:
    try:
        subprocess.run("git add -A", shell=True, cwd=cwd, capture_output=True, timeout=60)
        subprocess.run(["git", "commit", "-q", "-m", message], cwd=cwd, capture_output=True, timeout=60)
    except Exception:
        pass  # checkpointing is best-effort; never break the run


def git_revert_to_head(cwd: str) -> None:
    try:
        subprocess.run("git reset --hard HEAD", shell=True, cwd=cwd, capture_output=True, timeout=60)
    except Exception:
        pass


# Escalation hook — OFF by default. Given the task + the stuck context, return a
# hint string (from a frontier model) or None to let the agent re-strategize locally.
EscalateFn = Callable[[str], Optional[str]]


def default_escalate(_context: str) -> Optional[str]:
    return None
