# bench/swe_parse.py
"""Shared SEARCH/REPLACE block parsing for the SWE-bench agent.

Both swe_eval (the agent loop) and swe_debate (the propose↔critique escalation) need to
extract Aider-style SEARCH/REPLACE edit blocks from model text. Keeping the regex + parser
here (rather than in swe_eval) avoids the circular-import fragility of swe_debate importing
from swe_eval while swe_eval lazily imports swe_debate.
"""
from __future__ import annotations

import re

# Aider-style SEARCH/REPLACE block. dsv4-pro will not write a line-numbered unified diff from
# memory (it just keeps trying to re-read), but it can reproduce an exact snippet + replacement.
# Applying via edit_file makes the off-vs-codepro gap a function of whether the model REMEMBERS
# the exact code: off (truncated transcript) misremembers -> SEARCH miss; codepro recalls it.
_BLOCK_RE = re.compile(
    r"(?P<path>[^\n<>=]+?)\n<{5,7} SEARCH\n(?P<search>.*?)\n={5,7}\n(?P<replace>.*?)\n>{5,7} REPLACE",
    re.S)


def _parse_blocks(text: str) -> list[tuple[str, str, str]]:
    """Extract (path, search, replace) edit blocks from the model's patch-turn text."""
    if not text:
        return []
    return [(m.group("path").strip().strip("`").strip(),
             m.group("search"), m.group("replace")) for m in _BLOCK_RE.finditer(text)]
