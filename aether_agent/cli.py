"""`aether code "<task>"` — run the Aether coding agent (neo-lite)."""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from aether_agent.adapter import DEFAULT_MODEL
from aether_agent.agent import run_agent


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="aether",
        description="Aether Coding Agent (neo-lite) — autonomous local coding on Unlimited Context.",
    )
    sub = p.add_subparsers(dest="cmd")
    c = sub.add_parser("code", help="Run a coding task autonomously.")
    c.add_argument("task", nargs="+", help='The task, e.g. "fix the failing tests".')
    c.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model tag (default {DEFAULT_MODEL}).")
    c.add_argument("--pool", type=int, default=5, help="Pool size in GB (reach = pool x 233M tokens).")
    c.add_argument("--cwd", default=".", help="Workspace directory.")
    c.add_argument("--max-steps", type=int, default=40)
    c.add_argument("--test-cmd", default="pytest -q", help="Command the test-gate runs.")

    args = p.parse_args(argv)
    if args.cmd != "code":
        p.print_help()
        return 0

    task = " ".join(args.task)
    try:
        res = run_agent(
            task,
            model=args.model,
            pool_gb=args.pool,
            cwd=args.cwd,
            max_steps=args.max_steps,
            test_cmd=args.test_cmd,
        )
    except RuntimeError as e:
        print(f"\n✗ {e}", file=sys.stderr)
        return 1
    mark = "✓" if res.ok else "✗"
    print(f"\n{mark} {res.summary}  ({res.steps} steps)")
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
