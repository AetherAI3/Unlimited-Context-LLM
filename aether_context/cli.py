# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""``aether-context`` console script — init / pool-resize / doctor / bench.

Thin by design: every command is a few lines that call into the library. The CLI never
contains engine logic; it wires argparse to :mod:`aether_context.config`, the Ollama probe in
:mod:`aether_context.local_llm`, and the bench script.

Commands
--------
``aether-context init [--pool N] [--dir D]``
    Initialize / re-initialize the pool config. **Non-tty safe** (build plan §12 CRITICAL):
    the interactive slider only runs when stdin is a real tty; otherwise the size comes from
    ``--pool N`` (>=5), then ``$AETHER_POOL_GB``, then the 5 GB default — it never blocks.

``aether-context --pool N [--dir D]``
    Resize the pool (non-destructive re-index): rewrite ``pool_gb`` in the persisted config
    without touching the on-disk payloads. Rejects ``N < 5`` with the floor reason.

``aether-context doctor [--model M] [--dir D]``
    Runs **fully offline**. Checks the three things that ever go wrong (docs/local-models.md):
    Ollama reachable? model pulled? free RAM vs the configured index? — and prints the *exact*
    fix command for each. Never raises on a down daemon; reports it as a fixable condition.

``aether-context bench [--quick] [--model M] [--json]``
    Delegates to ``bench/drift_vs_window.py`` (engine ON vs OFF). Hermetic by default.

No ``print`` discipline note: this is the *one* place user-facing text is intentional, so the
CLI uses ``print`` for its report. The library proper (everything under ``aether_context`` that
is imported by ``Session``) stays silent and uses the logging seam.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

from aether_context import __version__
from aether_context.config import (
    BYTES_PER_GB,
    POOL_GB_FLOOR,
    PoolConfig,
    free_disk_bytes,
    reach_tokens,
)
from aether_context.errors import AetherContextError, PoolBudgetError
from aether_context.session import Session

#: Environment variable read for the pool size when no ``--pool`` flag is given (non-tty).
_ENV_POOL_GB = "AETHER_POOL_GB"
#: Environment variable for the Ollama host the doctor probes (overrides the default).
_ENV_OLLAMA_HOST = "OLLAMA_HOST"
#: Default Ollama host (mirrors local_llm.DEFAULT_OLLAMA_HOST without importing the adapter).
_DEFAULT_OLLAMA_HOST = "http://localhost:11434"
#: In-RAM ANN index size per GB of pool (MB). From the README table: 5 GB -> ~145 MB,
#: 10 GB -> ~291 MB, ... i.e. ~29 MB of resident index per GB of reach (the HNSW graph +
#: compact vector representation, *not* the full float32 vector store which lives on disk).
_INDEX_MB_PER_GB = 29
#: Short timeout (s) for the doctor's reachability probe — fail fast, stay offline-friendly.
_PROBE_TIMEOUT = 2.0


# ---------------------------------------------------------------------------
# Argument parser.
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for the ``aether-context`` console script.

    Exposes the ``init`` / ``doctor`` / ``bench`` subcommands plus a top-level ``--pool N``
    resize shorthand (so ``aether-context --pool 10`` works with no subcommand). Returns the
    parser; callers run ``parser.parse_args(argv)``.
    """
    parser = argparse.ArgumentParser(
        prog="aether-context",
        description=(
            "Unlimited Context — virtual memory for an LLM's attention. "
            "Init/resize the local pool, diagnose your setup, or run the bench."
        ),
    )
    parser.add_argument("--version", action="version", version=f"aether-context {__version__}")
    # top-level resize shorthand: `aether-context --pool 10`
    parser.add_argument(
        "--pool", type=int, default=None, metavar="N",
        help="resize the pool to N GB (>=5), non-destructive; runs with no subcommand.",
    )
    parser.add_argument(
        "--dir", type=str, default=None, metavar="D",
        help="pool directory (default: ~/.aether-context).",
    )

    subparsers = parser.add_subparsers(dest="command")

    p_init = subparsers.add_parser(
        "init", help="initialize the pool (interactive slider on a tty; else --pool/env/5GB)."
    )
    p_init.add_argument("--pool", type=int, default=None, metavar="N", help="pool size in GB (>=5).")
    p_init.add_argument("--dir", type=str, default=None, metavar="D", help="pool directory.")

    p_doctor = subparsers.add_parser(
        "doctor", help="diagnose Ollama reachability, model pull, and RAM-vs-index (offline)."
    )
    p_doctor.add_argument("--model", type=str, default=None, metavar="M", help="model to check.")
    p_doctor.add_argument("--dir", type=str, default=None, metavar="D", help="pool directory.")
    p_doctor.add_argument("--host", type=str, default=None, metavar="URL", help="Ollama host.")

    p_bench = subparsers.add_parser(
        "bench", help="run the engine ON-vs-OFF bench (hermetic mock by default)."
    )
    p_bench.add_argument("--model", type=str, default="mock", metavar="M", help="model spec.")
    p_bench.add_argument("--quick", action="store_true", help="shorter build (CI smoke).")
    p_bench.add_argument("--json", action="store_true", help="machine-readable report.")

    p_run = subparsers.add_parser(
        "run", help="run one task through the engine and print the result + a status line."
    )
    p_run.add_argument("task", type=str, help="the task/prompt to run.")
    _add_session_flags(p_run)

    p_chat = subparsers.add_parser(
        "chat", help="interactive REPL with slash-commands (/clear /new /status /quit ...)."
    )
    _add_session_flags(p_chat)

    p_status = subparsers.add_parser(
        "status", help="print pool GB / slices / reach / hit rate / pool-mode / index."
    )
    _add_session_flags(p_status)

    p_clear = subparsers.add_parser(
        "clear", help="empty the pool (this dir). --all removes the whole pool dir."
    )
    p_clear.add_argument("--dir", type=str, default=None, metavar="D", help="pool directory.")
    p_clear.add_argument(
        "--all", action="store_true",
        help="remove the entire pool dir (always confirms; non-tty needs --yes).",
    )
    p_clear.add_argument(
        "--yes", action="store_true",
        help="proceed without an interactive prompt (required for non-tty destructive clears).",
    )

    return parser


#: Allowed values for the session-config flags (mirror PoolConfig's validators).
_POOL_MODES = ("separate", "shared")
_INDEX_KINDS = ("flat", "hnsw", "tiered")
#: Default model for the run/chat surface — offline-safe so a clean clone just works.
_DEFAULT_MODEL = "mock"


def _add_session_flags(sub: argparse.ArgumentParser) -> None:
    """Attach the shared session-config flags (--model/--pool/--pool-mode/--index/--dir)."""
    sub.add_argument(
        "--model", type=str, default=_DEFAULT_MODEL, metavar="M",
        help="model spec (default: mock — runs fully offline).",
    )
    sub.add_argument(
        "--pool", type=int, default=None, metavar="N", help="pool size in GB (>=5).",
    )
    sub.add_argument(
        "--pool-mode", type=str, default="separate", choices=_POOL_MODES,
        metavar="{separate,shared}", help="pool sharing mode (default: separate).",
    )
    sub.add_argument(
        "--index", type=str, default="flat", choices=_INDEX_KINDS,
        metavar="{flat,hnsw,tiered}",
        help="ANN index kind (default: flat). 'tiered' is reserved and runs flat for now.",
    )
    sub.add_argument(
        "--no-mpo-chain", dest="mpo_chain", action="store_false", default=True,
        help="disable the MPO context chain (retrieval falls back to plain cosine). On by default.",
    )
    sub.add_argument("--dir", type=str, default=None, metavar="D", help="pool directory.")


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and dispatch to the matching command. Returns a process exit code.

    A bare invocation (no command, no ``--pool``) prints help and returns 0. Each command
    returns 0 on success and a non-zero code on a reported failure (so the script is usable in
    CI). Typed library errors are caught and rendered with their ``.hint``; nothing escapes.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "init":
            return _cmd_init(args)
        if args.command == "doctor":
            return _cmd_doctor(args)
        if args.command == "bench":
            return _cmd_bench(args)
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "chat":
            return _cmd_chat(args)
        if args.command == "status":
            return _cmd_status(args)
        if args.command == "clear":
            return _cmd_clear(args)
        # no subcommand: a top-level --pool is a resize; otherwise show help.
        if args.pool is not None:
            return _cmd_resize(args)
        parser.print_help()
        return 0
    except AetherContextError as exc:
        # typed, hinted failure: render it cleanly (never a traceback to the user).
        print(f"error: {exc.message}", file=sys.stderr)
        print(f"  fix: {exc.hint}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# init.
# ---------------------------------------------------------------------------
def _cmd_init(args: argparse.Namespace) -> int:
    """Initialize the pool config. Non-tty safe: prompt only on a real tty.

    Resolution order for the size: explicit ``--pool`` → interactive slider (tty only) →
    ``$AETHER_POOL_GB`` → the 5 GB default. A size below the floor is rejected with the reason.
    """
    pool_dir = _resolve_dir(args.dir)
    pool_gb = _resolve_pool_gb(getattr(args, "pool", None), pool_dir)
    _check_disk_for_pool(pool_gb, pool_dir)  # reject a pool that won't fit on disk
    cfg = _write_config(pool_dir, pool_gb)  # raises PoolBudgetError if < floor
    reach = reach_tokens(cfg.pool_gb)
    print(f"initialized pool at {cfg.dir}")
    print(f"  pool size: {cfg.pool_gb} GB  (reach ~= {reach / 1e9:.2f}B tokens)")
    print(f"  index: {cfg.index}   dim: {cfg.dim}   slice: {cfg.slice_tokens} tok")
    free = free_disk_bytes(pool_dir)
    if free is not None:
        print(f"  disk: {free / BYTES_PER_GB:.1f} GB free at {pool_dir}")
    return 0


def _resolve_pool_gb(flag: int | None, pool_dir: Path) -> int:
    """Resolve the pool size without ever blocking on a non-tty stdin.

    ``flag`` (``--pool``) wins. Else, only if stdin is an interactive tty do we run the slider
    (which shows free disk and rejects a size that won't fit). Else ``$AETHER_POOL_GB`` if set
    and numeric. Else the 5 GB default.
    """
    if flag is not None:
        return int(flag)
    if sys.stdin is not None and sys.stdin.isatty():
        return _prompt_pool_gb(pool_dir)
    env = os.environ.get(_ENV_POOL_GB)
    if env is not None and env.strip().isdigit():
        return int(env.strip())
    return POOL_GB_FLOOR


def _prompt_pool_gb(pool_dir: Path) -> int:
    """Interactive pool-size selector (the README slider). Only called on a real tty.

    The pool is **local disk** the engine reserves for context reach, so the slider shows how
    much disk is free, marks sizes that won't fit, and re-prompts (not just below the floor,
    but also when a pick exceeds free disk). Empty input takes the 5 GB default; EOF falls back
    to the default.
    """
    free = free_disk_bytes(pool_dir)
    free_gb = (free / BYTES_PER_GB) if free is not None else None
    print("Choose a pool size — this is the local DISK reserved for context reach:")
    if free_gb is not None:
        print(f"  ({free_gb:.1f} GB free at {pool_dir})")
    for gb in (5, 10, 15, 20):
        warn = "" if (free_gb is None or gb <= free_gb) else "   (won't fit — not enough free disk)"
        print(f"  {gb:>2} GB  ->  reach ~= {reach_tokens(gb) / 1e9:.2f}B tokens{warn}")
    for _attempt in range(3):
        try:
            raw = input(f"pool GB [default {POOL_GB_FLOOR}]: ").strip()
        except EOFError:
            return POOL_GB_FLOOR
        if not raw:
            return POOL_GB_FLOOR
        if not raw.isdigit():
            print("  enter a whole number of GB (e.g. 5, 10, 20).")
            continue
        value = int(raw)
        if value < POOL_GB_FLOOR:
            print(f"  {value} GB is below the {POOL_GB_FLOOR} GB floor; pick at least {POOL_GB_FLOOR}.")
            continue
        if free_gb is not None and value > free_gb:
            print(f"  {value} GB won't fit — only {free_gb:.1f} GB free at {pool_dir}. Pick a smaller size.")
            continue
        return value
    return POOL_GB_FLOOR


# ---------------------------------------------------------------------------
# resize (top-level --pool).
# ---------------------------------------------------------------------------
def _cmd_resize(args: argparse.Namespace) -> int:
    """Resize the pool to ``--pool N`` GB (non-destructive). Rejects ``N < floor``.

    "Non-destructive" = we only rewrite ``pool_gb`` in the persisted config; the on-disk
    vectors/sidecar payloads are left in place (a later pool open re-indexes around the new
    size). We load any existing config first so other settings (index/dim/...) are preserved.
    """
    pool_dir = _resolve_dir(args.dir)
    existing = PoolConfig.load(pool_dir)  # preserves index/dim/slice; defaults if absent
    _check_disk_for_pool(int(args.pool), pool_dir)  # reject a resize that won't fit on disk
    cfg = _write_config(
        pool_dir, int(args.pool),
        index=existing.index, dim=existing.dim, slice_tokens=existing.slice_tokens,
        mode=existing.mode,
    )
    print(f"resized pool at {cfg.dir} to {cfg.pool_gb} GB "
          f"(reach ~= {reach_tokens(cfg.pool_gb) / 1e9:.2f}B tokens)")
    print("  re-index is non-destructive: your encoded slices are preserved.")
    return 0


def _write_config(pool_dir: Path, pool_gb: int, **fields: object) -> PoolConfig:
    """Build + persist a PoolConfig (validates the floor inside ``__post_init__``)."""
    cfg = PoolConfig(pool_gb=pool_gb, dir=pool_dir, **fields)  # type: ignore[arg-type]
    cfg.save()
    return cfg


def _check_disk_for_pool(pool_gb: int, pool_dir: Path) -> None:
    """Reject a pool that won't fit on local disk (no-op if free space can't be probed).

    The pool reserves ``pool_gb`` of disk for encoded context. If less than that is free on
    the target filesystem, refuse loudly with a typed, hinted error rather than letting the
    pool fill up and fail mid-run.
    """
    free = free_disk_bytes(pool_dir)
    if free is None:
        return  # cannot probe -> do not block
    if free >= pool_gb * BYTES_PER_GB:
        return
    free_gb = free / BYTES_PER_GB
    fits = int(free // BYTES_PER_GB)
    if fits >= POOL_GB_FLOOR:
        hint = f"free up disk, or pick a smaller pool that fits: aether-context --pool {fits}"
    else:
        hint = (
            f"free up disk space — even the {POOL_GB_FLOOR} GB floor needs {POOL_GB_FLOOR} GB "
            f"free (only {free_gb:.1f} GB available at {pool_dir})"
        )
    raise PoolBudgetError(
        f"not enough disk: a {pool_gb} GB pool needs {pool_gb} GB free at {pool_dir}, "
        f"but only {free_gb:.1f} GB is available",
        hint=hint,
    )


def _resolve_dir(flag: str | None) -> Path:
    """Resolve the pool directory: ``--dir`` if given, else ``~/.aether-context``."""
    if flag:
        return Path(flag)
    return PoolConfig().dir


# ---------------------------------------------------------------------------
# run — one task through the engine, then a one-line status.
# ---------------------------------------------------------------------------
def _build_session(args: argparse.Namespace) -> Session:
    """Construct a :class:`Session` from the shared session-config flags (offline-safe).

    ``--pool`` falls back to any persisted config's reach (so ``run`` after ``init`` honors
    the chosen size) then the 5 GB floor. ``fallback_to_mock=True`` keeps a clean clone
    working with no backend installed.
    """
    pool_dir = _resolve_dir(args.dir)
    pool_gb = args.pool if args.pool is not None else PoolConfig.load(pool_dir).pool_gb
    return Session(
        model=args.model,
        pool_gb=int(pool_gb),
        pool_mode=args.pool_mode,
        pool_index=args.index,
        pool_dir=pool_dir,
        mpo_chain=getattr(args, "mpo_chain", True),
        fallback_to_mock=True,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    """Run ``args.task`` through a fresh session, print the text then a one-line status.

    The session is closed in a ``finally`` so the pool is always flushed (the encoded slices
    survive for a later ``status`` / ``chat`` over the same dir). Returns 0 on success.
    """
    session = _build_session(args)
    try:
        result = session.run(args.task)
        print(result.text)
        print(_status_line(session.status_dict()))
        return 0
    finally:
        session.close()


def _status_line(s: dict[str, Any]) -> str:
    """A compact one-line status summary for the ``run`` tail."""
    return (
        f"[pool {s['pool_gb']} GB | slices {s['slices_used']}/{s['capacity']} | "
        f"reach {int(s['reach_tokens']) / 1e9:.2f}B tok | hit {float(s['hit_rate']):.0%} | "
        f"mode {s['pool_mode']} | index {s['index']}]"
    )


# ---------------------------------------------------------------------------
# status — open the pool read-only and print the honest status fields.
# ---------------------------------------------------------------------------
def _cmd_status(args: argparse.Namespace) -> int:
    """Print the status fields for the pool at ``--dir`` (no live session, so hit rate N/A).

    Loads the persisted :class:`PoolConfig`, opens the pool read-only to count its slices,
    and reports reach / capacity / resident-RAM estimate. The hit rate is honestly ``N/A``
    here: there is no running pager to measure, so we do not fabricate one.
    """
    pool_dir = _resolve_dir(args.dir)
    cfg = PoolConfig.load(pool_dir)
    slices, capacity = _pool_counts(cfg)
    reach = reach_tokens(cfg.pool_gb)
    resident_mb = cfg.pool_gb * _INDEX_MB_PER_GB
    print("aether-context status")
    print(f"  pool:        {cfg.pool_gb} GB  (reach ~= {reach / 1e9:.2f}B tokens)")
    print(f"  slices:      {slices} / {capacity}")
    print(f"  reach:       {reach:,} tokens")
    print("  hit rate:    N/A (no live session)")
    print(f"  resident RAM ~= {resident_mb} MB (estimate)")
    print(f"  pool-mode:   {cfg.mode}")
    print(f"  index:       {cfg.index}")
    return 0


def _pool_counts(cfg: PoolConfig) -> tuple[int, int]:
    """``(slices_used, capacity)`` for the pool at ``cfg.dir`` (0/0 if none on disk yet).

    Opens the pool read-only via :class:`Session`'s storage layer; on a fresh/absent pool
    the count is 0. Closed immediately so no mmap handle lingers (Windows-safe).
    """
    from aether_context.context_pool import ContextPool, slice_cost_bytes

    pool = ContextPool(cfg)
    try:
        used = len(pool)
        capacity = pool.ceiling_bytes // slice_cost_bytes(cfg.dim)
        return used, int(capacity)
    finally:
        pool.close()


# ---------------------------------------------------------------------------
# clear — empty the pool (this dir) / remove the whole dir, with confirmation.
# ---------------------------------------------------------------------------
def _cmd_clear(args: argparse.Namespace) -> int:
    """Clear the pool at ``--dir`` (all sessions) or remove the whole dir with ``--all``.

    Confirmation policy (honest + safe): ``--all`` ALWAYS confirms; a non-default ``clear``
    confirms when the pool is ``shared`` or a named/persistent dir. On a tty we ask; off a
    tty we require ``--yes`` and refuse with a message otherwise (never block on input()).
    """
    pool_dir = _resolve_dir(args.dir)
    if args.all:
        return _clear_all(pool_dir, assume_yes=args.yes)
    return _clear_slices(pool_dir, assume_yes=args.yes)


def _clear_all(pool_dir: Path, *, assume_yes: bool) -> int:
    """Remove the entire pool dir (always confirmed)."""
    if not _confirm(f"remove the ENTIRE pool dir {pool_dir}?", assume_yes=assume_yes):
        print("clear --all aborted (no confirmation).")
        return 1
    if pool_dir.exists():
        shutil.rmtree(pool_dir, ignore_errors=True)
    print(f"removed pool dir {pool_dir}")
    return 0


def _clear_slices(pool_dir: Path, *, assume_yes: bool) -> int:
    """Empty the pool's slices (all sessions) at ``pool_dir``, confirming if persistent/shared."""
    cfg = PoolConfig.load(pool_dir)
    needs_confirm = cfg.mode == "shared" or _is_persistent_dir(pool_dir)
    if needs_confirm and not _confirm(
        f"clear all slices in the {cfg.mode} pool at {pool_dir}?", assume_yes=assume_yes
    ):
        print("clear aborted (no confirmation).")
        return 1
    removed = _clear_pool_slices(cfg)
    print(f"cleared {removed} slice(s) from the pool at {pool_dir}")
    return 0


def _clear_pool_slices(cfg: PoolConfig) -> int:
    """Drop every slice in the pool at ``cfg.dir`` (global clear) and flush. Returns the count.

    ``ContextPool.close`` is idempotent, so the ``finally`` flush is safe even though the
    happy path also closes (it must, so the emptied sidecar is on disk before we return and
    a following ``status`` reads zero slices).
    """
    from aether_context.context_pool import ContextPool

    pool = ContextPool(cfg)
    try:
        return pool.clear_session(None)  # None -> clear all sessions' slices
    finally:
        pool.close()  # flush the now-empty sidecar so a later status sees 0 (idempotent)


def _is_persistent_dir(pool_dir: Path) -> bool:
    """A dir is 'persistent' (worth confirming before clearing) iff it is not the default.

    The default ``~/.aether-context`` is the throwaway/ephemeral home; an explicit ``--dir``
    is treated as a named/persistent pool, so clearing it asks first on a tty.
    """
    try:
        return pool_dir.resolve() != PoolConfig().dir.resolve()
    except OSError:
        return True


def _confirm(question: str, *, assume_yes: bool) -> bool:
    """Confirm a destructive action. ``--yes`` / a tty 'y' proceeds; non-tty without --yes refuses.

    Never blocks under a non-tty (CI / pipes): if stdin is not interactive and ``--yes`` was
    not passed we return False with an explanatory message rather than calling ``input()``.
    """
    if assume_yes:
        return True
    if sys.stdin is None or not sys.stdin.isatty():
        print(f"refusing: {question} (non-interactive; pass --yes to proceed)", file=sys.stderr)
        return False
    try:
        answer = input(f"{question} [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


# ---------------------------------------------------------------------------
# chat — interactive REPL with a pure, testable slash-command dispatcher.
# ---------------------------------------------------------------------------
@dataclass
class ReplState:
    """Mutable state the slash dispatcher reads/advises on (the REPL owns the side effects).

    ``dispatch_slash`` is **pure**: it never touches the session or prints — it parses one
    slash line against this state and returns an ``(action, message)`` pair. The REPL loop is
    the only place that actually mutates the session, prints, or exits.
    """

    model: str = _DEFAULT_MODEL
    pool_gb: int = POOL_GB_FLOOR
    pool_mode: str = "separate"
    index: str = "flat"
    extended: bool = False


#: The actions ``dispatch_slash`` can return (the REPL maps each to a real side effect).
SLASH_ACTIONS: tuple[str, ...] = (
    "continue", "quit", "clear", "new", "status", "pool", "model",
    "think", "export", "help", "unknown",
)

#: The help text shown for ``/help`` (also printed at chat start).
_CHAT_HELP = (
    "slash-commands: /clear (alias /cls)  /new  /status  /pool <GB>  /model <name>  "
    "/think  /export [file]  /help  /quit"
)


def dispatch_slash(state: ReplState, line: str) -> tuple[str, str]:
    """Parse one slash ``line`` against ``state`` -> ``(action, message)``. PURE: no side effects.

    Recognized: ``/clear`` (alias ``/cls``), ``/new``, ``/status``, ``/pool <GB>``,
    ``/model <name>``, ``/think``, ``/export [file]``, ``/help``, ``/quit`` (aliases
    ``/exit`` / ``/q``). Anything else yields ``("unknown", ...)``. ``message`` is the
    argument payload for parameterized commands (e.g. the GB for ``/pool``, the path for
    ``/export``) or a human-readable note; the REPL performs the actual effect.

    Robust to a leading UTF-8 BOM that some shells prepend to piped/redirected input — both
    the decoded form (``﻿``) and the raw 3-byte form (``\xef\xbb\xbf``) that appears when
    Windows reads piped stdin under a non-UTF-8 console encoding — so a ``/command`` is still
    recognized when the line is fed in non-interactively.
    """
    text = line.lstrip("﻿\xef\xbb\xbf").strip()
    if not text.startswith("/"):
        return ("continue", text)
    parts = text[1:].split(maxsplit=1)
    cmd = parts[0].lower() if parts else ""
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("quit", "exit", "q"):
        return ("quit", "")
    if cmd in ("clear", "cls"):
        return ("clear", "")
    if cmd == "new":
        return ("new", "")
    if cmd == "status":
        return ("status", "")
    if cmd == "help":
        return ("help", _CHAT_HELP)
    if cmd == "think":
        return ("think", "")
    if cmd == "export":
        return ("export", arg)
    if cmd == "pool":
        return ("pool", arg)
    if cmd == "model":
        return ("model", arg)
    return ("unknown", f"unknown command: /{cmd} ({_CHAT_HELP})")


def _cmd_chat(args: argparse.Namespace) -> int:
    """Interactive REPL. Non-tty: read a single line (or none) and exit cleanly.

    Each input line is routed through :func:`dispatch_slash`; non-slash lines call
    ``session.ask`` and print the reply. A best-effort ``readline`` binding inserts
    ``/clear`` on Ctrl+L. In ``separate`` mode the ephemeral pool is dropped on exit.
    """
    session = _build_session(args)
    state = ReplState(
        model=args.model, pool_gb=session.pool_gb, pool_mode=args.pool_mode,
        index=args.index, extended=session.extended,
    )
    _bind_readline_clear()
    interactive = sys.stdin is not None and sys.stdin.isatty()
    if interactive:
        print(f"aether-context chat — {_CHAT_HELP}")
    try:
        return _chat_loop(args, session, state, interactive=interactive)
    finally:
        _drop_ephemeral(args, session)


def _chat_loop(
    args: argparse.Namespace, session: Session, state: ReplState, *, interactive: bool
) -> int:
    """The read/dispatch/print loop. Returns 0 on a clean exit."""
    while True:
        try:
            line = input("> " if interactive else "")
        except (EOFError, KeyboardInterrupt):
            return 0
        action, message = dispatch_slash(state, line)
        if action == "quit":
            return 0
        cont = _apply_chat_action(args, session, state, action, message)
        if not cont:
            return 0
        if not interactive:
            # Non-tty chat handles exactly one line then exits cleanly (never blocks).
            return 0


def _apply_chat_action(
    args: argparse.Namespace,
    session: Session,
    state: ReplState,
    action: str,
    message: str,
) -> bool:
    """Perform the side effect for one dispatched ``action``. Returns False to end the loop."""
    if action == "continue":
        if message:
            print(session.ask(message))
        return True
    if action == "help":
        print(message)
        return True
    if action == "status":
        for ln in _status_lines(session.status_dict()):
            print(ln)
        return True
    if action == "clear":
        removed = session.clear(scope="session")
        print(f"cleared {removed} slice(s); resident window reset.")
        return True
    if action == "new":
        session.clear(scope="resident")
        print("resident window cleared; reachable pool kept.")
        return True
    if action == "think":
        on = session.toggle_extended()
        state.extended = on
        print(f"extended thinking: {'on' if on else 'off'}")
        return True
    if action == "export":
        path = session.export(message or None)
        print(f"transcript exported to {path}")
        return True
    if action == "pool":
        print(_apply_pool_change(state, message))
        return True
    if action == "model":
        if message:
            state.model = message
        print(f"model set to {state.model} (applies to the next `chat`/`run`).")
        return True
    # unknown
    print(message)
    return True


def _apply_pool_change(state: ReplState, message: str) -> str:
    """Validate + record a ``/pool <GB>`` change on the REPL state (advisory; not live-resized)."""
    if not message.isdigit():
        return f"usage: /pool <GB>  (got {message!r})"
    gb = int(message)
    if gb < POOL_GB_FLOOR:
        return f"{gb} GB is below the {POOL_GB_FLOOR} GB floor; keeping {state.pool_gb} GB."
    state.pool_gb = gb
    return f"pool size set to {gb} GB (applies to the next `chat`/`run`)."


def _status_lines(s: dict[str, Any]) -> list[str]:
    """Multi-line status block for the REPL ``/status`` (mirrors the shell ``status`` fields)."""
    reach = int(s["reach_tokens"])
    return [
        f"  pool:        {s['pool_gb']} GB  (reach ~= {reach / 1e9:.2f}B tokens)",
        f"  slices:      {s['slices_used']} / {s['capacity']}",
        f"  reach:       {reach:,} tokens",
        f"  hit rate:    {float(s['hit_rate']):.0%}",
        f"  resident RAM ~= {s['resident_ram_mb']} MB (estimate)",
        f"  pool-mode:   {s['pool_mode']}",
        f"  index:       {s['index']}",
        f"  model:       {s['model']}    extended: {s['extended']}",
    ]


def _bind_readline_clear() -> None:
    """Best-effort: bind Ctrl+L to insert ``/clear`` via readline (no-op if unavailable)."""
    try:
        import readline  # noqa: PLC0415 - optional, best-effort on this platform
    except ImportError:
        return
    bind = getattr(readline, "parse_and_bind", None)
    if bind is None:
        return
    try:
        bind(r'"\C-l": "/clear\n"')
    except (OSError, ValueError) as exc:
        # readline present but the binding syntax was rejected on this build — non-fatal.
        print(f"  (note: could not bind Ctrl+L: {exc})", file=sys.stderr)


def _drop_ephemeral(args: argparse.Namespace, session: Session) -> None:
    """On exit, close the session; in separate/ephemeral mode also drop its slices.

    Honest cleanup: ``separate`` mode is ephemeral, so the slices this chat encoded are
    dropped on exit (the next chat starts clean). ``shared`` / a persistent dir keeps them.
    """
    if args.pool_mode == "separate":
        try:
            session.clear(scope="session")
        except AetherContextError as exc:
            print(f"  (note: ephemeral clear skipped: {exc})", file=sys.stderr)
    session.close()


# ---------------------------------------------------------------------------
# doctor — runs fully offline, prints exact fixes.
# ---------------------------------------------------------------------------
def _cmd_doctor(args: argparse.Namespace) -> int:
    """Diagnose the three common failure modes, printing the exact fix for each.

    Runs fully offline: the Ollama reachability probe has a short timeout and any network
    failure is rendered as a fixable condition (with ``ollama serve`` / ``ollama pull``), never
    raised. Returns 0 if everything checks out, 1 if any check found a problem.
    """
    host = _resolve_host(args.host)
    model = args.model
    pool_dir = _resolve_dir(args.dir)
    print("aether-context doctor")
    print(f"  ollama host: {host}")

    ok = True
    reachable = _probe_ollama(host)
    ok = _report_ollama(reachable, host) and ok
    ok = _report_model(reachable, host, model) and ok
    ok = _report_disk_vs_pool(pool_dir) and ok
    ok = _report_ram_vs_index(pool_dir) and ok

    print("")
    print("  all good." if ok else "  some checks need attention (see fixes above).")
    return 0 if ok else 1


def _resolve_host(flag: str | None) -> str:
    """Resolve the Ollama host: ``--host`` → ``$OLLAMA_HOST`` → default."""
    if flag:
        return flag.rstrip("/")
    env = os.environ.get(_ENV_OLLAMA_HOST)
    if env:
        return env.rstrip("/")
    return _DEFAULT_OLLAMA_HOST


def _probe_ollama(host: str) -> bool:
    """Best-effort reachability probe of the Ollama daemon. Never raises (offline-safe)."""
    try:
        req = urllib.request.Request(f"{host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=_PROBE_TIMEOUT) as resp:
            return 200 <= resp.status < 500
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _report_ollama(reachable: bool, host: str) -> bool:
    """Print the Ollama reachability check + its fix command. Returns True iff reachable."""
    if reachable:
        print(f"  [ok]   ollama daemon reachable at {host}")
        return True
    print(f"  [fail] ollama daemon not reachable at {host}")
    print("         fix: start it with `ollama serve`")
    return False


def _report_model(reachable: bool, host: str, model: str | None) -> bool:
    """Check whether ``model`` is pulled (only meaningful if the daemon is up).

    Always prints the exact ``ollama pull <model>`` fix command when a model was named, so the
    user sees the remedy even fully offline.
    """
    if model is None:
        print("  [skip] no --model given; pass --model qwen2.5 to check a specific model")
        return True
    if not reachable:
        print(f"  [fail] cannot check model '{model}' (daemon down)")
        print(f"         fix: start ollama then `ollama pull {model}`")
        return False
    pulled = _model_is_pulled(host, model)
    if pulled:
        print(f"  [ok]   model '{model}' is pulled")
        return True
    print(f"  [fail] model '{model}' is not pulled")
    print(f"         fix: `ollama pull {model}`  (or Session(model='ollama/{model}', pull=True))")
    return False


def _model_is_pulled(host: str, model: str) -> bool:
    """Return True iff ``model`` appears in ``/api/tags``. Offline-safe (False on any failure)."""
    try:
        req = urllib.request.Request(f"{host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=_PROBE_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return False
    names = {m.get("name", "") for m in (body.get("models") or [])}
    # ollama tags carry a ':tag' suffix; match the bare name or any tag of it.
    base = model.split(":", 1)[0]
    return any(n == model or n.split(":", 1)[0] == base for n in names)


def _report_disk_vs_pool(pool_dir: Path) -> bool:
    """Check free disk against the configured pool size (the pool reserves that much disk)."""
    cfg = PoolConfig.load(pool_dir)
    free = free_disk_bytes(pool_dir)
    if free is None:
        print(f"  [ok]   {cfg.pool_gb} GB pool — free disk unknown (could not probe)")
        return True
    free_gb = free / BYTES_PER_GB
    if free >= cfg.pool_gb * BYTES_PER_GB:
        print(f"  [ok]   {cfg.pool_gb} GB pool fits ({free_gb:.1f} GB free at {pool_dir})")
        return True
    fits = max(POOL_GB_FLOOR, int(free // BYTES_PER_GB))
    print(f"  [warn] {cfg.pool_gb} GB pool vs only {free_gb:.1f} GB free at {pool_dir}")
    print(f"         fix: free up disk, or `aether-context --pool {fits}`")
    return False


def _report_ram_vs_index(pool_dir: Path) -> bool:
    """Estimate the index RAM for the configured pool and compare to free system RAM.

    Reads the persisted ``PoolConfig`` (or defaults), computes the index RAM from the pool
    reach math (README table: ~145 MB at 5 GB), and warns if it would not comfortably fit in
    free RAM. Free RAM is probed best-effort; if it cannot be read we report the estimate only.
    """
    cfg = PoolConfig.load(pool_dir)
    index_bytes = _estimate_index_bytes(cfg)
    index_mb = index_bytes / (1024 * 1024)
    free_bytes = _free_ram_bytes()
    if free_bytes is None:
        print(f"  [ok]   index RAM estimate ~= {index_mb:.0f} MB "
              f"(free RAM unknown; could not probe)")
        return True
    free_mb = free_bytes / (1024 * 1024)
    # comfortable = index fits in well under half of free RAM.
    if index_bytes * 2 < free_bytes:
        print(f"  [ok]   index RAM ~= {index_mb:.0f} MB fits in {free_mb:.0f} MB free")
        return True
    print(f"  [warn] index RAM ~= {index_mb:.0f} MB vs only {free_mb:.0f} MB free")
    print("         fix: use a smaller --pool (a paged 'tiered' index is not built yet)")
    return False


def _estimate_index_bytes(cfg: PoolConfig) -> int:
    """In-RAM ANN index estimate for ``cfg``, matching the README table (~145 MB at 5 GB).

    The resident index scales with reach (more reach -> more slices -> a bigger graph), so the
    published table is linear in ``pool_gb`` at ~29 MB/GB. The full float32 vector store is far
    larger but lives on disk (mmap), so it is not what bounds RAM — the in-RAM index is.
    """
    return int(cfg.pool_gb * _INDEX_MB_PER_GB * 1024 * 1024)


def _free_ram_bytes() -> int | None:
    """Best-effort free-RAM probe using only the standard library. None if unavailable.

    Tries ``os.sysconf`` (POSIX) then a Windows ctypes call. Never raises — a probe failure
    simply yields ``None`` and the report degrades gracefully.
    """
    posix = _free_ram_posix()
    if posix is not None:
        return posix
    return _free_ram_windows()


def _free_ram_posix() -> int | None:
    """POSIX free-RAM via ``os.sysconf`` (available pages * page size). None off-POSIX."""
    try:
        names = getattr(os, "sysconf_names", {})
        sysconf = getattr(os, "sysconf", None)  # absent on Windows
        if sysconf is not None and "SC_AVPHYS_PAGES" in names and "SC_PAGE_SIZE" in names:
            pages = sysconf("SC_AVPHYS_PAGES")
            page_size = sysconf("SC_PAGE_SIZE")
            if pages > 0 and page_size > 0:
                return int(pages) * int(page_size)
    except (AttributeError, ValueError, OSError):
        return None
    return None


def _free_ram_windows() -> int | None:
    """Windows free-RAM via ``GlobalMemoryStatusEx`` (ctypes). None on non-Windows/failure."""
    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = _MemStatus()
        stat.dwLength = ctypes.sizeof(_MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
            return int(stat.ullAvailPhys)
    except (OSError, AttributeError, ValueError):
        return None
    return None


# ---------------------------------------------------------------------------
# bench — delegate to bench/drift_vs_window.py.
# ---------------------------------------------------------------------------
def _cmd_bench(args: argparse.Namespace) -> int:
    """Delegate to ``bench/drift_vs_window.py``. Returns its exit code.

    The bench lives outside the importable package (it's a script under ``bench/``), so we load
    it by file path. If it cannot be located (e.g. an installed wheel that omits ``bench/``) we
    print an actionable hint rather than crashing.
    """
    module = _load_bench_module()
    if module is None:
        print("bench script not found (bench/drift_vs_window.py).", file=sys.stderr)
        print("  fix: run from a source checkout, or `python bench/drift_vs_window.py`.",
              file=sys.stderr)
        return 1
    bench_argv: list[str] = ["--model", str(args.model)]
    if args.quick:
        bench_argv.append("--quick")
    if getattr(args, "json", False):
        bench_argv.append("--json")
    return int(module.main(bench_argv))


def _load_bench_module() -> ModuleType | None:
    """Locate + import ``bench/drift_vs_window.py`` by file path. None if not found."""
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "bench" / "drift_vs_window.py",  # repo checkout: <root>/bench/
        Path.cwd() / "bench" / "drift_vs_window.py",          # invoked from repo root
    ]
    for path in candidates:
        if path.is_file():
            mod_name = "aether_context_bench"
            spec = importlib.util.spec_from_file_location(mod_name, path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            # Register before exec so the module's own dataclasses can resolve
            # ``cls.__module__`` via ``sys.modules`` during class creation.
            sys.modules[mod_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:  # noqa: BLE001 - a broken bench file shouldn't crash the CLI
                sys.modules.pop(mod_name, None)
                raise
            return module
    return None


__all__ = ["main", "build_parser", "dispatch_slash", "ReplState", "SLASH_ACTIONS"]


if __name__ == "__main__":  # pragma: no cover - exercised via `python -m aether_context.cli`
    sys.exit(main())
