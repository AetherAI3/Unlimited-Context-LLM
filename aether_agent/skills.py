"""
Skill layer — inject the procedure the small local model lacks.

The third local-hardening layer (memory · procedure · correctness). Retrieval
makes a small model *remember*; it can't make it *know how*. A skill closes that
gap: a reusable procedure packet pinned into the prompt at the right moment, so
the local model behaves like it knows the approach.

A skill is a PRIOR, not truth — it guides; execution decides. If a skill's
approach fails the tests, the test result overrides it: results win over guidance.

MVP matcher = explicit trigger descriptors (keyword / error-pattern substrings),
deterministic and offline. A semantic-recall layer can augment this later
(hybrid match); the packet shape here is what it will surface.
"""

from __future__ import annotations

from dataclasses import dataclass

# A small reserved budget — pin at most this many skills so they never crowd out
# context retrieval (the spec's `k_skill` reserved slots).
SKILL_BUDGET = 2


@dataclass(frozen=True)
class Skill:
    name: str
    triggers: tuple[str, ...]  # lowercased substrings: keywords / error patterns
    procedure: str             # the how-to injected into the prompt
    acceptance: str            # the "done" definition; feeds the grounding gate

    def matches(self, focus: str) -> int:
        """How many trigger descriptors fire against the focus text (0 = miss)."""
        low = focus.lower()
        return sum(1 for t in self.triggers if t in low)

    def as_prompt(self) -> str:
        return (
            f"PROCEDURE (skill: {self.name}) — a prior, not truth; tests override it.\n"
            f"{self.procedure}\n"
            f"DONE WHEN: {self.acceptance}"
        )


# --- starter library (procedure packets) ----------------------------------
LIBRARY: tuple[Skill, ...] = (
    Skill(
        name="fix-failing-tests",
        triggers=("failing test", "fix the test", "tests fail", "pytest", "make tests pass", "failed"),
        procedure=(
            "1. Run the suite first; read the FIRST failure, not the summary.\n"
            "2. After running tests, READ THE SOURCE FILES the failing tests import — the\n"
            "   implementation under the package dir, NOT the test files. The fix lives in\n"
            "   the source, never in the test.\n"
            "3. Edit the source to make the assertions pass; change the smallest thing that\n"
            "   could fix THIS failure; re-run to confirm progress.\n"
            "4. Never edit a test to pass unless the test itself is wrong — fix the code.\n"
            "5. Fix one module at a time; re-run after each; checkpoint after each green step.\n"
            "6. Do not stop until run_tests exits 0."
        ),
        acceptance="the previously-failing tests pass and no previously-passing test regressed",
    ),
    Skill(
        name="safe-db-migration",
        triggers=("migration", "alter table", "schema change", "add column", "drop column", "supabase"),
        procedure=(
            "1. Read the current schema before writing the migration.\n"
            "2. Additive first (add nullable column / new table); backfill; then tighten.\n"
            "3. Never drop/rename in the same step that code still reads the old shape.\n"
            "4. Make it reversible; write the down-path. Test on a copy, not prod."
        ),
        acceptance="the migration applies forward and back cleanly and the suite passes against the new schema",
    ),
    Skill(
        name="debug-async-race",
        triggers=("race condition", "deadlock", "async", "flaky", "intermittent", "hangs", "await"),
        procedure=(
            "1. Make it reproduce deterministically before fixing (seed / loop / inject delay).\n"
            "2. Find the shared state crossed by two flows; that is the race.\n"
            "3. Prefer eliminating the sharing over adding a lock; if locking, narrow the section.\n"
            "4. Re-run the repro many times; one green run does not prove a race fixed."
        ),
        acceptance="the repro passes repeatedly (>=20x) with no intermittent failure",
    ),
    Skill(
        name="trace-the-bug",
        triggers=("bug", "broken", "error", "exception", "traceback", "crash", "not working"),
        procedure=(
            "1. Reproduce it; capture the exact error + stack.\n"
            "2. Walk the stack from the throw site outward; read each frame's real source.\n"
            "3. Form ONE hypothesis; add a check/print that would falsify it; run.\n"
            "4. Fix the cause, not the symptom; add a test that pins the bug closed."
        ),
        acceptance="a new test reproduces the bug, then passes after the fix, suite still green",
    ),
)


def match_skills(focus: str, *, limit: int = SKILL_BUDGET, exclude: frozenset[str] = frozenset()) -> list[Skill]:
    """Top skills whose triggers fire against the focus, best first (>=1 hit).

    `exclude` skips already-loaded skills so a re-match mid-run only surfaces
    something new. Ties break on library order (stable) for determinism.
    """
    scored = [
        (s.matches(focus), -i, s)
        for i, s in enumerate(LIBRARY)
        if s.name not in exclude and s.matches(focus) > 0
    ]
    scored.sort(reverse=True)
    return [s for _, _, s in scored[:limit]]
