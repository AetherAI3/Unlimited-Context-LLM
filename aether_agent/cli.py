"""``aether`` — the Aether coding agent front door.

Dispatch (mirror of aether-code ``src/cli.ts``):

  - bare ``aether``                 -> interactive REPL (``repl.main``)
  - ``aether "<prompt>"``           -> one-shot chat turn via ``select_brain``
  - ``aether code "<task>"``        -> autonomous coding run (``run_agent``)
  - ``aether brain``                -> headless stdio brain (for the TS host)
  - ``aether auth login|status|logout|token``
  - ``aether models``               -> list models (cloud catalog or local hint)
  - ``aether config [show|get k|set k v]``

The ``code`` and ``brain`` subcommands keep their exact prior behavior. Anything
that is not a known subcommand and does not start with ``-`` is treated as a
one-shot prompt (so ``aether "fix the bug"`` works without a subcommand).
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Optional

from aether_agent.adapter import DEFAULT_MODEL
from aether_agent.agent import run_agent
from aether_agent.auth import FileTokenStore, auth_status, login_with_password
from aether_agent.brains import select_brain
from aether_agent.config import load_config, save_config
from aether_agent.transport import ApiClient

#: The recognized subcommands. A first positional matching one of these is routed
#: to its handler; anything else (not starting with ``-``) is a one-shot prompt.
_SUBCOMMANDS = frozenset({"code", "brain", "auth", "models", "config"})


# --- top-level dispatch ----------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    # Bare invocation -> interactive REPL.
    if not args:
        from aether_agent import repl

        return repl.main()

    head = args[0]
    if head == "code":
        return _cmd_code(args[1:])
    if head == "brain":
        from aether_agent.headless import main as brain_main

        return brain_main()
    if head == "auth":
        return _cmd_auth(args[1:])
    if head == "models":
        return _cmd_models(args[1:])
    if head == "config":
        return _cmd_config(args[1:])

    # An explicit flag with no subcommand (e.g. ``aether --help``) -> show help.
    if head.startswith("-"):
        _print_help()
        return 0

    # Otherwise: a one-shot prompt (``aether "fix the bug"``).
    prompt = " ".join(args)
    return _one_shot(prompt)


# --- code (unchanged behavior) ---------------------------------------------
def _cmd_code(rest: list[str]) -> int:
    p = argparse.ArgumentParser(prog="aether code", description="Run a coding task autonomously.")
    p.add_argument("task", nargs="+", help='The task, e.g. "fix the failing tests".')
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model tag (default {DEFAULT_MODEL}).")
    p.add_argument("--pool", type=int, default=5, help="Pool size in GB (reach = pool x 233M tokens).")
    p.add_argument("--cwd", default=".", help="Workspace directory.")
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--test-cmd", default="pytest -q", help="Command the test-gate runs.")
    ns = p.parse_args(rest)

    task = " ".join(ns.task)
    try:
        res = run_agent(
            task,
            model=ns.model,
            pool_gb=ns.pool,
            cwd=ns.cwd,
            max_steps=ns.max_steps,
            test_cmd=ns.test_cmd,
        )
    except RuntimeError as e:
        print(f"\n✗ {e}", file=sys.stderr)
        return 1
    mark = "✓" if res.ok else "✗"
    print(f"\n{mark} {res.summary}  ({res.steps} steps)")
    return 0 if res.ok else 1


# --- one-shot chat turn ----------------------------------------------------
def _one_shot(prompt: str) -> int:
    """Run a single chat turn via the policy-selected brain and print the answer."""
    if not prompt.strip():
        _print_help()
        return 0
    cfg = load_config()
    store = FileTokenStore()
    api = ApiClient(cfg.get("baseUrl", ""), store)
    brain = select_brain(
        authed=store.get() is not None,
        backend=str(cfg.get("backend", "auto")),
        api=api,
        model=str(cfg.get("defaultModel", "") or ""),
    )
    try:
        for ev in brain.run(prompt):
            _render_oneshot(ev)
    except RuntimeError as e:
        print(f"\n✗ {e}", file=sys.stderr)
        return 1
    return 0


def _render_oneshot(ev: dict[str, Any]) -> None:
    etype = ev.get("type")
    if etype in ("monologue", "done"):
        text = str(ev.get("text", ""))
        if text:
            print(text)
    elif etype == "tool_call":
        print(f"  · {ev.get('name', '')}", file=sys.stderr)
    elif etype == "error":
        print(f"\n✗ {ev.get('msg', 'error')}", file=sys.stderr)


# --- auth ------------------------------------------------------------------
def _cmd_auth(rest: list[str]) -> int:
    p = argparse.ArgumentParser(prog="aether auth", description="Manage Aether authentication.")
    sub = p.add_subparsers(dest="action")
    lg = sub.add_parser("login", help="Log in (token or username/password).")
    lg.add_argument("--token", help="Use an API key / session token directly.")
    lg.add_argument("--username", "-u", help="Username for password login.")
    lg.add_argument("--password", "-p", help="Password for password login.")
    lg.add_argument("--with-token", action="store_true", help="Read a token from stdin.")
    sub.add_parser("status", help="Show the current auth state.")
    sub.add_parser("logout", help="Clear the stored token.")
    sub.add_parser("token", help="Print the masked active token.")
    ns = p.parse_args(rest)

    cfg = load_config()
    base_url = cfg.get("baseUrl", "")
    store = FileTokenStore()
    action = ns.action or "status"

    if action == "login":
        return _auth_login(ns, base_url, store)
    if action == "logout":
        store.clear()
        print("logged out.")
        return 0
    if action == "token":
        st = auth_status(base_url, store)
        print(st["masked"] if st["logged_in"] else "(not logged in)")
        return 0
    # status (default)
    st = auth_status(base_url, store)
    if st["logged_in"]:
        print(f"logged in ({st['token_type']}) {st['masked']}  @ {st['base_url']}")
    else:
        print(f"not logged in  @ {st['base_url']}")
    return 0


def _auth_login(ns: argparse.Namespace, base_url: str, store: FileTokenStore) -> int:
    token = ns.token
    if ns.with_token and not token:
        token = sys.stdin.readline().strip()
    if token:
        store.set(token)
        print("token saved.")
        return 0
    if ns.username and ns.password:
        try:
            res = login_with_password(base_url, store, ns.username, ns.password)
        except RuntimeError as e:
            print(f"✗ {e}", file=sys.stderr)
            return 1
        plan = res.get("plan")
        print(f"logged in{f' (plan {plan})' if plan else ''}.")
        return 0
    print("usage: aether auth login [--token T | --username U --password P | --with-token]", file=sys.stderr)
    return 1


# --- models ----------------------------------------------------------------
def _cmd_models(rest: list[str]) -> int:
    cfg = load_config()
    store = FileTokenStore()
    if store.get() is None:
        print("(local Ollama — set a model with: aether config set defaultModel <tag>)")
        return 0
    api = ApiClient(cfg.get("baseUrl", ""), store)
    from aether_agent.transport import MODELS_PATH

    try:
        payload = api.get_json(MODELS_PATH)
    except Exception as e:  # noqa: BLE001 — never crash on a network/parse error
        print(f"✗ could not fetch models: {e}", file=sys.stderr)
        return 1
    models = payload.get("models", []) if isinstance(payload, dict) else []
    tier = payload.get("tier", "") if isinstance(payload, dict) else ""
    if tier:
        print(f"tier: {tier}")
    for i, m in enumerate(models, 1):
        if isinstance(m, dict):
            print(f"{i:>2}. {m.get('id', '')}\t{m.get('label', '')}".rstrip())
    return 0


# --- config ----------------------------------------------------------------
def _cmd_config(rest: list[str]) -> int:
    action = rest[0] if rest else "show"
    cfg = load_config()
    if action == "show":
        for k in sorted(cfg):
            print(f"{k} = {cfg[k]}")
        return 0
    if action == "get":
        if len(rest) < 2:
            print("usage: aether config get <key>", file=sys.stderr)
            return 1
        key = rest[1]
        if key not in cfg:
            print(f"(no such key: {key})", file=sys.stderr)
            return 1
        print(cfg[key])
        return 0
    if action == "set":
        if len(rest) < 3:
            print("usage: aether config set <key> <value>", file=sys.stderr)
            return 1
        key, value = rest[1], " ".join(rest[2:])
        cfg[key] = _coerce(value)
        save_config(cfg)
        print(f"{key} = {cfg[key]}")
        return 0
    print("usage: aether config [show | get <key> | set <key> <value>]", file=sys.stderr)
    return 1


def _coerce(value: str) -> Any:
    """Coerce a config string value to bool when it clearly is one (else keep str)."""
    low = value.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    return value


# --- help ------------------------------------------------------------------
def _print_help() -> None:
    print(
        "\n".join(
            [
                "aether — the Aether coding agent",
                "",
                "  aether                       open the interactive REPL",
                '  aether "<prompt>"            one-shot chat turn',
                '  aether code "<task>"         autonomous coding run',
                "  aether brain                 headless stdio brain (for the TS host)",
                "  aether auth login|status|logout|token",
                "  aether models                list models",
                "  aether config [show|get k|set k v]",
            ]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
