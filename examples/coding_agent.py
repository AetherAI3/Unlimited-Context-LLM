# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI - Brandon Barrante
# SPDX-License-Identifier: Apache-2.0
"""Coding agent — the headline use case: a long build that stays coherent.

A coding agent works a multi-stage build (spec -> schema -> auth -> api -> tests -> review). The
load-bearing **constraints** are stated once, in the spec, then must hold across every later
stage — long past any small model's native context window.

With Unlimited Context the spec is **encoded into the local pool** up front and **paged back**
whenever a later stage needs it, so the agent never contradicts its own spec. This script makes
that visible: after a build that overflows the window many times over, we *recall* the original
constraints from the pool and show they are still reachable.

Runs online (real local model) or fully offline (mock fallback) — like the quickstart, it
*always* runs. With a real model you see coherent prose; with mock you still see the mechanism
(the constraints survive the overflow and are recovered).

Run it::

    python examples/coding_agent.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

try:
    from aether_context import Session
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from aether_context import Session

from aether_context.errors import AetherContextError

#: The spec the agent must never contradict — stated once, must hold to the end.
SPEC = (
    "PROJECT SPEC (load-bearing, do not contradict): "
    "the database is postgres; the auth token expiry is exactly 3600 seconds; "
    "the api framework is fastapi; the package manager is uv."
)
#: The build stages a coding agent walks. Each is a real instruction to the model.
STAGES: tuple[tuple[str, str], ...] = (
    ("schema", "Design the database schema and migrations for the user and session models."),
    ("auth", "Implement the auth module: login, refresh tokens, and token expiry handling."),
    ("api", "Build the REST API endpoints for users and sessions."),
    ("tests", "Write the integration test suite for the auth and api modules."),
    ("tooling", "Set up the project tooling, the lockfile, and the CI pipeline."),
    ("review", "Review the whole build against the original spec for any contradictions."),
)
#: A query that should retrieve the spec constraints from the pool at any later stage.
RECOVERY_QUERY = "what are the database, auth token expiry, api framework, and tooling constraints"
#: The constraint tokens we verify survived the overflow.
CONSTRAINTS = ("postgres", "3600", "fastapi", "uv")
#: Preferred real model; mock fallback keeps the example always-runnable offline.
PREFERRED_MODEL = "ollama/qwen2.5"
#: Tiny window + long output for the mock so the build overflows and the engine must page.
_MOCK_WINDOW = 48
_MOCK_OUTPUT = 200


def _open_session(pool_dir: Path) -> tuple[Session, str]:
    """Open a Session on the preferred model; fall back to mock (offline) with a hint."""
    try:
        session = Session(model=PREFERRED_MODEL, pool_gb=5, pool_dir=pool_dir)
        _ = session.context_window  # force the lazy daemon probe
        # a quick generate probe so we fall back *before* the real build if the daemon is down
        next(session.local_llm.generate("ping", max_tokens=1), "")
        return session, PREFERRED_MODEL
    except AetherContextError as exc:
        print(f"[hint] {PREFERRED_MODEL} unavailable ({exc.message.splitlines()[0]}).")
        print("[hint] running the build on the offline 'mock' model instead.")
        print("[hint] for the real thing: `ollama serve` then `ollama pull qwen2.5`.")
        return (
            Session(
                model="mock", pool_gb=5, pool_dir=pool_dir / "mock",
                context_window=_MOCK_WINDOW, output_tokens=_MOCK_OUTPUT,
            ),
            "mock",
        )


def main() -> int:
    """Run a long multi-stage build and prove the spec stayed reachable. Returns 0 on success."""
    with tempfile.TemporaryDirectory(prefix="aether-coding-agent-") as tmp:
        pool_dir = Path(tmp)
        session, label = _open_session(pool_dir)
        try:
            print(f"coding agent on model: {label}\n")
            # 1) state the spec once — it is encoded into the pool and hardened
            session.remember(SPEC, tags={"kind": "spec"})
            print("planted spec into the context pool (encode-on-spill).")

            # 2) walk the build; each stage overflows the window and spills to the pool
            for name, instruction in STAGES:
                result = session.run(instruction)
                flag = " (overflowed window)" if result.overflowed else ""
                print(f"  stage '{name}': {result.spilled} slices spilled{flag}")

            # 3) recover the spec from the pool — still reachable after all that overflow
            print("\nrecalling the original spec after the long build...")
            hits = session.recall(RECOVERY_QUERY, k=8)
            joined = " ".join(h.text.lower() for h in hits)
            reached = [c for c in CONSTRAINTS if c.lower() in joined]
            print(f"  constraints still reachable: {sorted(reached)}")
            print(f"  pager hit rate: {session.pager.hit_rate():.2f}")

            ok = len(reached) == len(CONSTRAINTS)
            verdict = (
                "the spec survived the whole build — no drift."
                if ok
                else "some constraints were not recovered (see hit rate / pool size)."
            )
            print(f"\nverdict: {verdict}")
        finally:
            session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
