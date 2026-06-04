"""
Moat-seal invariant (local mirror of .github/workflows/moat-seal.yml).

Runs in the normal pytest suite so a leak of the private AETHER-ATLAS/qosc core into
this public tree is caught before push, not only in CI. Loads the checker by file path
so it works regardless of how `tools/` is packaged.
"""
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "moat_seal_check", REPO_ROOT / "tools" / "moat_seal_check.py"
)
_moat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_moat)


def test_no_private_core_references_in_public_tree():
    violations = _moat.find_violations(REPO_ROOT)
    assert violations == [], "Moat seal breached:\n" + "\n".join(violations)
