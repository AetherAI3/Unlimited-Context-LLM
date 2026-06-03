# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Quickstart — the three lines, made true.

    from aether_context import Session
    s = Session(model="ollama/qwen2.5", pool_gb=5)
    print(s.run("Build me a full-stack weightlifting tracker app.").text)

This script *always* runs, online or off. If Ollama is reachable and the model is pulled it
uses the real local model; if not, it prints a one-line hint and falls back to the built-in
deterministic ``mock`` model so a clean clone runs green with **zero models installed**.

Run it::

    python examples/quickstart.py

No flags, no API key, no account, no network required (the fallback is fully offline).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Make `aether_context` importable when run straight from a source checkout.
try:
    from aether_context import Session
except ImportError:  # pragma: no cover - only when run before an editable install
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from aether_context import Session

from aether_context.errors import AetherContextError

#: The headline prompt — the kind of long build the engine is for.
TASK = "Build me a full-stack weightlifting tracker app."
#: The model we'd prefer (real, local). Falls back to mock if it can't be reached.
PREFERRED_MODEL = "ollama/qwen2.5"


def _open_session(pool_dir: Path) -> tuple[Session, str]:
    """Open a Session on the preferred local model, falling back to mock if unreachable.

    Returns ``(session, model_label)``. The fallback is what makes the quickstart *always*
    run: a missing Ollama daemon / unpulled model is caught (typed error) and we degrade to
    the offline ``mock`` model with a printed hint, never a crash.
    """
    try:
        session = Session(model=PREFERRED_MODEL, pool_gb=5, pool_dir=pool_dir)
        # Touch the window so an unreachable daemon surfaces now, not mid-run. The Ollama
        # adapter probes lazily; reading context_window forces it (fail-soft inside the
        # adapter still returns a fallback, so the real generate check happens at run time).
        _ = session.context_window
        return session, PREFERRED_MODEL
    except AetherContextError as exc:
        print(f"[hint] {PREFERRED_MODEL} unavailable ({exc.message.splitlines()[0]}).")
        print("[hint] falling back to the offline 'mock' model so this still runs.")
        print("[hint] for the real thing: `ollama serve` then `ollama pull qwen2.5`.")
        return Session(model="mock", pool_gb=5, pool_dir=pool_dir), "mock"


def _run_real_or_mock(session: Session, label: str, pool_dir: Path) -> str:
    """Run the task; if a *real* backend fails at generate time, retry once on mock.

    The Ollama adapter only contacts the daemon when ``generate`` actually streams, so a daemon
    that is down can still raise here even though construction succeeded. We catch that and fall
    back to mock so the script's promise ("always runs") holds end to end.
    """
    try:
        return session.run(TASK).text
    except AetherContextError as exc:
        if label == "mock":
            raise  # mock cannot fail this way; surface anything unexpected
        print(f"[hint] {label} could not generate ({exc.message.splitlines()[0]}).")
        print("[hint] falling back to the offline 'mock' model.")
        fallback = Session(model="mock", pool_gb=5, pool_dir=pool_dir / "mock")
        try:
            return fallback.run(TASK).text
        finally:
            fallback.close()


def main() -> int:
    """Run the quickstart end to end and print the result. Always returns 0 on success."""
    with tempfile.TemporaryDirectory(prefix="aether-quickstart-") as tmp:
        pool_dir = Path(tmp)
        session, label = _open_session(pool_dir)
        try:
            print(f"running on model: {label}")
            text = _run_real_or_mock(session, label, pool_dir)
        finally:
            session.close()

        preview = text if len(text) <= 280 else text[:277] + "..."
        print("\n--- model output (preview) ---")
        print(preview)
        print("--- end ---")
        print("\nThat's it: three lines reached a billion-token-capable local pool, offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
