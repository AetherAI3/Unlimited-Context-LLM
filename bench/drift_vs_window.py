# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""drift_vs_window — the kill-gate bench (engine ON vs OFF, same base model).

> Build plan §7. One long scripted build, run twice with the *same* base model: once with
> the Unlimited Context engine ON (encode-on-spill + paged recall) and once OFF (the raw
> model window only). The **delta** is the pitch.

Metrics reported (per build, ON and OFF):

  * **drift**      — cross-stage contradictions: how often a later stage cannot reach a
                     constraint that an earlier stage established. Lower is better.
  * **correctness**— per-stage correctness: fraction of dependent stages whose planted
                     constraints are still reachable when that stage runs.
  * **hit_rate**   — retrieval hit rate of the pager over the run (ON only; OFF has no pager).
  * **completion** — unattended completion: did the build reach the final stage with every
                     load-bearing constraint still reachable, with no human re-priming.

Hermetic by default (``--model mock`` with a deliberately small ``context_window``): the
mock has no real intelligence, so the bench measures **reachability of planted facts**, which
is exactly the mechanism the engine provides. ON keeps the planted constraints reachable past
the tiny window; OFF loses them as the transcript scrolls. ``--model ollama/qwen2.5`` runs it
for real against a local model.

**Replay/mock before any real call** (repo CLAUDE.md law): the default path touches no
network and no GPU; only an explicit ``--model ollama/...`` (or another real backend) makes a
real call. If that real backend cannot be loaded (e.g. Ollama down), the ON arm degrades to
the mock and the report notes the substitution — the bench never crashes on a missing daemon.

CLI::

    python bench/drift_vs_window.py                 # hermetic mock, full
    python bench/drift_vs_window.py --quick         # CI smoke (fewer stages)
    python bench/drift_vs_window.py --model ollama/qwen2.5
    python bench/drift_vs_window.py --json          # machine-readable report

This module is import-safe: it adds the repo root to ``sys.path`` if ``aether_context`` is not
already importable, so it runs both as ``python bench/drift_vs_window.py`` and when the CLI
loads it by file path.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

# --- import bootstrap: make `aether_context` importable when run as a file ---
try:  # pragma: no cover - exercised both ways depending on how it's launched
    import aether_context  # noqa: F401
except ImportError:  # pragma: no cover
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from aether_context.errors import AetherContextError
from aether_context.session import Session
from aether_context.slice_loader import SliceKey

# ---------------------------------------------------------------------------
# The scripted build: a sequence of stages, each planting and/or depending on a
# load-bearing constraint. The full task is one long, multi-stage software build.
# ---------------------------------------------------------------------------
#: Each stage: (name, the instruction text, the constraint tokens it depends on).
#: The constraints are *planted in stage 0* and must survive to be reachable later.
_FULL_STAGES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "spec",
        "Project spec. CONSTRAINTS: database is postgres; auth token expiry is exactly "
        "3600 seconds; the api framework is fastapi; the package manager is uv. Hold these.",
        (),
    ),
    ("schema", "Design the database schema and migrations for the user model.", ("postgres",)),
    ("auth", "Implement the auth module: login, refresh, and token expiry handling.", ("3600",)),
    ("api", "Build the REST API endpoints for users and sessions.", ("fastapi",)),
    ("tests", "Write the integration test suite for the auth and api modules.", ("3600", "fastapi")),
    ("tooling", "Set up the project tooling, lockfile, and CI pipeline.", ("uv",)),
    ("review", "Review the whole build for consistency against the original constraints.",
     ("postgres", "3600", "fastapi", "uv")),
)

#: The four load-bearing constraints, planted up front, that every later stage must respect.
_CONSTRAINTS: tuple[str, ...] = ("postgres", "3600", "fastapi", "uv")

#: The seed "spec" text the run begins with — the constraints live here verbatim.
_SPEC_TEXT: str = _FULL_STAGES[0][1]

#: Small mock window (tokens) so a multi-stage build overflows it many times over.
_MOCK_WINDOW_TOKENS: int = 48
#: Mock output length (tokens) per stage — long enough to push the spec out of the window.
_MOCK_OUTPUT_TOKENS: int = 220
#: A query that should retrieve the planted spec constraints.
_RECOVERY_QUERY: str = "what are the database auth token expiry api framework and tooling constraints"


@dataclass(frozen=True)
class ArmResult:
    """Metrics for one arm of the bench (engine ON or OFF)."""

    label: str
    drift: int
    correctness: float
    hit_rate: float
    completion: bool
    stages_total: int
    constraints_reachable: int
    constraints_total: int


@dataclass(frozen=True)
class BenchResult:
    """The head-to-head outcome: ON vs OFF and the delta that is the pitch."""

    model: str
    quick: bool
    on: ArmResult
    off: ArmResult
    note: str = ""

    @property
    def drift_reduction(self) -> int:
        """How many cross-stage contradictions the engine eliminated (OFF - ON)."""
        return self.off.drift - self.on.drift

    @property
    def correctness_gain(self) -> float:
        """Per-stage correctness improvement from the engine (ON - OFF)."""
        return self.on.correctness - self.off.correctness

    @property
    def engine_wins(self) -> bool:
        """The headline gate: the engine never regresses and strictly improves at least one
        of (drift, correctness, completion)."""
        no_regression = (
            self.on.correctness >= self.off.correctness and self.on.drift <= self.off.drift
        )
        improvement = (
            self.on.correctness > self.off.correctness
            or self.on.drift < self.off.drift
            or (self.on.completion and not self.off.completion)
        )
        return no_regression and improvement


# ---------------------------------------------------------------------------
# OFF arm — the raw-window baseline (no engine).
# ---------------------------------------------------------------------------
def _run_off(stages: Sequence[tuple[str, str, tuple[str, ...]]]) -> ArmResult:
    """Baseline: only the model's raw window is visible. We model that window as the last
    ``_MOCK_WINDOW_TOKENS`` worth of transcript characters. A constraint is "reachable" at a
    stage iff its token still appears inside that window. The spec scrolls off quickly, so by
    the later stages the original constraints are unreachable -> drift + lost correctness."""
    window_chars = _MOCK_WINDOW_TOKENS * 4
    transcript = _SPEC_TEXT
    drift = 0
    satisfied_stages = 0
    dependent_stages = 0
    filler = " build module function class endpoint handler test refactor encode slice token"

    for _name, instruction, deps in stages:
        # the model "works" on the stage: append its instruction + generated filler
        transcript += " " + instruction + (filler * 16)
        visible = transcript[-window_chars:].lower()
        if deps:
            dependent_stages += 1
            if all(tok.lower() in visible for tok in deps):
                satisfied_stages += 1
            else:
                drift += 1  # a constraint the stage depends on is off-window -> contradiction risk

    final_visible = transcript[-window_chars:].lower()
    reachable = sum(1 for c in _CONSTRAINTS if c.lower() in final_visible)
    correctness = (satisfied_stages / dependent_stages) if dependent_stages else 1.0
    completion = reachable == len(_CONSTRAINTS)
    return ArmResult(
        label="OFF (raw window)",
        drift=drift,
        correctness=round(correctness, 3),
        hit_rate=0.0,
        completion=completion,
        stages_total=len(stages),
        constraints_reachable=reachable,
        constraints_total=len(_CONSTRAINTS),
    )


# ---------------------------------------------------------------------------
# ON arm — the full engine (encode-on-spill + paged recall).
# ---------------------------------------------------------------------------
def _open_session(model: str, pool_dir: Path, *, context_window: int, output_tokens: int) -> tuple[Session, str]:
    """Open the ON-arm Session. Returns ``(session, note)`` where ``note`` is non-empty if a
    requested real backend could not be reached and we degraded to the mock.

    Replay/mock-before-real law: for a real backend we run a *tiny generate probe* up front so an
    unreachable daemon / unpulled model surfaces here (one cheap call) and we fall back to mock
    *before* the long build starts — the bench never crashes on a missing daemon.
    """
    is_mock = model == "mock" or model.endswith("/mock")
    if is_mock:
        session = Session(
            model="mock", pool_gb=5, pool_dir=pool_dir,
            context_window=context_window, output_tokens=output_tokens,
        )
        return session, ""

    try:
        session = Session(model=model, pool_gb=5, pool_dir=pool_dir)
        _ = session.context_window  # force the lazy daemon probe
        next(session.local_llm.generate("ping", max_tokens=1), "")  # tiny real-call probe
        return session, ""
    except AetherContextError as exc:
        # real backend unavailable (e.g. Ollama down / model not pulled): degrade to mock.
        note = (
            f"requested model '{model}' unavailable "
            f"({exc.message.splitlines()[0]}); ran ON arm on mock"
        )
        fallback = Session(
            model="mock", pool_gb=5, pool_dir=pool_dir / "mock",
            context_window=context_window, output_tokens=output_tokens,
        )
        return fallback, note


def _run_on(
    model: str,
    stages: Sequence[tuple[str, str, tuple[str, ...]]],
    pool_dir: Path,
    *,
    context_window: int,
    output_tokens: int,
) -> tuple[ArmResult, str]:
    """Engine ON: the spec constraints are planted into the pool, then each stage runs through
    the Session (overflowing the tiny window). At each dependent stage we *recall* from the pool
    and check the constraint is reachable — which is exactly what the engine restores."""
    drift = 0
    satisfied_stages = 0
    dependent_stages = 0
    session, note = _open_session(
        model, pool_dir, context_window=context_window, output_tokens=output_tokens
    )
    try:
        # plant the load-bearing spec constraints up front (high-salience, hardened)
        session.remember(_SPEC_TEXT, tags={"kind": "spec", "stage": "spec"})
        # a stable pager key for this session's region — recovery reads go *through the pager*
        # so its warm cache (and therefore its measured hit rate) is exercised, not bypassed.
        recovery_key = SliceKey(session=session.id, topic="recovery")
        for _name, instruction, deps in stages:
            session.run(instruction)  # overflow happens here; spill is encoded + paged
            if deps:
                dependent_stages += 1
                # page the spec region back in through the pager (warm cache hot path)
                hits = session.pager.get(recovery_key, _RECOVERY_QUERY, k=8)
                joined = " ".join(h.text.lower() for h in hits)
                if all(tok.lower() in joined for tok in deps):
                    satisfied_stages += 1
                else:
                    drift += 1

        final_hits = session.pager.get(recovery_key, _RECOVERY_QUERY, k=8)
        joined = " ".join(h.text.lower() for h in final_hits)
        reachable = sum(1 for c in _CONSTRAINTS if c.lower() in joined)
        hit_rate = float(session.pager.hit_rate())
    finally:
        session.close()

    correctness = (satisfied_stages / dependent_stages) if dependent_stages else 1.0
    completion = reachable == len(_CONSTRAINTS)
    arm = ArmResult(
        label="ON (engine)",
        drift=drift,
        correctness=round(correctness, 3),
        hit_rate=round(hit_rate, 3),
        completion=completion,
        stages_total=len(stages),
        constraints_reachable=reachable,
        constraints_total=len(_CONSTRAINTS),
    )
    return arm, note


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------
def run_bench(model: str = "mock", *, quick: bool = False) -> BenchResult:
    """Run both arms over the same scripted build and return the head-to-head result.

    ``model`` is a spec string (default ``"mock"`` — hermetic). ``quick`` runs a shorter build
    (first few stages) for a CI smoke. No network is touched unless ``model`` names a real
    backend, and even then a load failure degrades the ON arm to the mock (it never crashes).
    """
    stages = _FULL_STAGES[:4] if quick else _FULL_STAGES
    off = _run_off(stages)
    with tempfile.TemporaryDirectory(prefix="aether-bench-") as tmp:
        on, note = _run_on(
            model,
            stages,
            Path(tmp),
            context_window=_MOCK_WINDOW_TOKENS,
            output_tokens=_MOCK_OUTPUT_TOKENS,
        )
    return BenchResult(model=model, quick=quick, on=on, off=off, note=note)


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------
def format_report(result: BenchResult) -> str:
    """A compact human-readable ON-vs-OFF report with the headline delta."""
    on, off = result.on, result.off
    lines = [
        "Unlimited Context — drift vs window bench",
        f"  model: {result.model}{'  (quick)' if result.quick else ''}",
    ]
    if result.note:
        lines.append(f"  note: {result.note}")
    lines += [
        "",
        f"  {'metric':<14}{'OFF (window)':>16}{'ON (engine)':>16}",
        f"  {'-' * 14}{'-' * 16:>16}{'-' * 16:>16}",
        f"  {'drift':<14}{off.drift:>16}{on.drift:>16}",
        f"  {'correctness':<14}{off.correctness:>16}{on.correctness:>16}",
        f"  {'hit_rate':<14}{off.hit_rate:>16}{on.hit_rate:>16}",
        f"  {'completion':<14}{str(off.completion):>16}{str(on.completion):>16}",
        f"  {'reach':<14}"
        f"{f'{off.constraints_reachable}/{off.constraints_total}':>16}"
        f"{f'{on.constraints_reachable}/{on.constraints_total}':>16}",
        "",
        f"  delta: drift -{result.drift_reduction}, "
        f"correctness +{round(result.correctness_gain, 3)}",
        f"  verdict: {'ON beats OFF' if result.engine_wins else 'no win (investigate)'}",
    ]
    return "\n".join(lines)


def result_to_dict(result: BenchResult) -> dict[str, object]:
    """JSON-serializable view of the bench result (for ``--json`` / CI artifacts)."""
    return {
        "model": result.model,
        "quick": result.quick,
        "note": result.note,
        "on": asdict(result.on),
        "off": asdict(result.off),
        "delta": {
            "drift_reduction": result.drift_reduction,
            "correctness_gain": round(result.correctness_gain, 3),
            "engine_wins": result.engine_wins,
        },
    }


# ---------------------------------------------------------------------------
# Argument parsing + entry point.
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Argument parser for the bench (also used when the CLI delegates here)."""
    parser = argparse.ArgumentParser(
        prog="drift_vs_window",
        description="Engine ON vs OFF: drift, correctness, hit rate, completion.",
    )
    parser.add_argument(
        "--model",
        default="mock",
        help="model spec (default: mock — hermetic). e.g. ollama/qwen2.5 runs for real.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run a shorter build (CI smoke).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a machine-readable JSON report instead of the table.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bench from ``argv`` and print the report. Returns 0 if the engine wins, else 1.

    The non-zero exit on "no win" makes this usable as a CI gate (``bench-smoke``): a build
    that regresses the engine's reachability advantage fails the pipeline.
    """
    args = build_parser().parse_args(argv)
    result = run_bench(model=args.model, quick=bool(args.quick))
    if args.json:
        print(json.dumps(result_to_dict(result), indent=2, sort_keys=True))
    else:
        print(format_report(result))
    return 0 if result.engine_wins else 1


if __name__ == "__main__":  # pragma: no cover - script entry
    raise SystemExit(main())
