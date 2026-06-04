#!/usr/bin/env python3
"""
Moat-seal guard — the public Unlimited-Context tree must NEVER reference the private
AETHER-ATLAS / qosc core.

The dependency is one-way by design: private ``aether_atlas`` may consume this public
``aether_context`` engine, but this public repo never imports private logic. The engine
here is clean-room — it reimplements behavior, it does not copy the publish-forbidden
qosc core (e.g. retention / latency_budget / intake / exploration). This script turns
"please don't leak the moat" into "the build won't let you": it scans every git-tracked
file and fails if a private namespace appears.

Canonical rule: AETHER-ATLAS ``specs/CONTRACTS.md`` → "Moat seal". Run by
``.github/workflows/moat-seal.yml`` and ``tests/test_moat_seal.py``.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import List

# Private namespaces that must never appear in this public tree.
FORBIDDEN_MODULES = ("aether_atlas", "qosc")
# Private directory components (a vendored private file/dir is an instant leak).
FORBIDDEN_DIR_PARTS = {"aether_atlas", "aether-atlas", "qosc"}

CODE_EXT = {".py", ".pyi"}
DEP_EXT = {".toml", ".cfg", ".ini", ".txt", ".in", ".json"}
# Build artefacts / vendored output are not source — skip to avoid noise.
EXCLUDE_TOP = {"build", "dist", "dist_verify"}
# The guard machinery legitimately names the private modules; exclude from content scan.
ALLOWLIST = {"tools/moat_seal_check.py", "tests/test_moat_seal.py"}

_mod = "|".join(re.escape(m) for m in FORBIDDEN_MODULES)
IMPORT_RE = re.compile(rf"(?:^|[^\w.])(?:from|import)\s+({_mod})\b")
ATTR_RE = re.compile(rf"\b({_mod})\.")


def _tracked_files(root: Path) -> List[Path]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [root / line for line in out.splitlines() if line]


def find_violations(root: Path) -> List[str]:
    """Return a list of human-readable violation strings (empty == moat intact)."""
    violations: List[str] = []
    for f in _tracked_files(root):
        rel = f.relative_to(root).as_posix()
        parts = rel.split("/")

        if parts and parts[0] in EXCLUDE_TOP:
            continue
        if parts[0].endswith(".egg-info"):
            continue

        # 1. a vendored private directory anywhere in the path
        if any(part in FORBIDDEN_DIR_PARTS for part in parts[:-1]):
            violations.append(f"{rel}: tracked file under a private module directory")

        if rel in ALLOWLIST:
            continue

        ext = f.suffix.lower()
        if ext not in CODE_EXT and ext not in DEP_EXT:
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for i, line in enumerate(text.splitlines(), 1):
            if ext in CODE_EXT:
                if IMPORT_RE.search(line) or ATTR_RE.search(line):
                    violations.append(f"{rel}:{i}: private import/use -> {line.strip()[:80]}")
            else:  # dependency / config file
                for m in FORBIDDEN_MODULES:
                    if m in line:
                        violations.append(
                            f"{rel}:{i}: private reference in dependency/config -> {line.strip()[:80]}"
                        )
    return violations


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    violations = find_violations(root)
    if violations:
        print("MOAT SEAL VIOLATION — the public tree references the private AETHER-ATLAS/qosc core:")
        for v in violations:
            print("  " + v)
        print(
            f"\n{len(violations)} violation(s). The public engine must be clean-room: "
            "reimplement behavior, never import private logic."
        )
        return 1
    print("moat seal intact: no aether_atlas/qosc references in the public tree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
