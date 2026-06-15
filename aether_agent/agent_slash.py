# aether_agent/agent_slash.py
# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""The `/agent` command grammar (sub-project B).

Pure handlers: configuration verbs mutate the store and return a ``text`` result;
switching/running return action flags (``switch_agent`` / ``run_agent``) the REPL
acts on (mirrors the existing ``setup``/``restart`` flags in slash.py). No TTY,
no Ollama — fully unit-testable.

    /new-agent <name>
    /agents
    /agent <name>                 -> {"switch_agent": name}
    /agent <name> <task...>       -> {"run_agent": {"name", "task"}}
    /agent <name> set <key> <val>
    /agent <name> show | edit | delete [--purge]
"""
from __future__ import annotations

import os
import subprocess
from typing import Any

from aether_agent import agent_store
from aether_agent.agent_profile import Agent

SlashResult = dict[str, Any]
_VERBS = {"set", "show", "edit", "delete", "cmd", "run"}
_SET_ALIASES = {"memory": "pool_gb"}


def _text(s: str) -> SlashResult:
    return {"text": s}


def dispatch_agent(line: str, *, active: str) -> SlashResult:
    """Route a `/new-agent`, `/agents`, or `/agent ...` line. Never raises."""
    body = (line or "").strip()
    if body.startswith("/"):
        body = body[1:]
    parts = body.split()
    if not parts:
        return _text("usage: /agent <name> [task | set <k> <v> | show | edit | delete]")
    head = parts[0].lower()
    if head == "new-agent":
        return _new_agent(parts[1:])
    if head == "agents":
        return _agents(active)
    if head == "agent":
        return _agent(parts[1:], active)
    return _text(f"(unknown command: /{head})")


def _new_agent(args: list) -> SlashResult:
    if not args:
        return _text("usage: /new-agent <name>")
    name = args[0]
    if agent_store.exists(name):
        return _text(f"agent already exists: {name}")
    try:
        Agent.create(name)
    except (ValueError, OSError) as e:
        return _text(f"could not create {name}: {e}")
    return _text(f"created agent '{name}' (customize: /agent {name} set <key> <val>)")


def _agents(active: str) -> SlashResult:
    from aether_agent import agents_view
    names = agent_store.list_agents()
    if not names:
        return _text("(no agents yet - create one with /new-agent <name>)")
    rows = []
    for n in names:
        try:
            a = agent_store.load(n)
            rows.append({"name": n, "model": a.model, "pool_gb": a.pool_gb,
                         "n_commands": len(a.commands)})
        except (ValueError, OSError):
            rows.append({"name": n, "model": "(corrupt)", "pool_gb": 0, "n_commands": 0})
    return _text(agents_view.render_agents_columns(rows, active=active))


def _agent(args: list, active: str) -> SlashResult:
    if not args:
        return _text("usage: /agent <name> [task | set <k> <v> | show | edit | delete]")
    name = args[0]
    rest = args[1:]
    if not agent_store.exists(name):
        return _text(f"no such agent: {name} (see /agents, or /new-agent {name})")
    if not rest:
        return {"switch_agent": name, "text": f"switched to {name}"}
    verb = rest[0].lower()
    if verb == "set":
        return _set(name, rest[1:])
    if verb == "show":
        return _show(name)
    if verb == "edit":
        return _edit(name)
    if verb == "delete":
        return _delete(name, rest[1:])
    if verb == "cmd":
        return _cmd(name, rest[1:])
    if verb == "run":
        return _run_cmd(name, rest[1:])
    return {"run_agent": {"name": name, "task": " ".join(rest)}, "text": ""}


def _set(name: str, kv: list) -> SlashResult:
    if len(kv) < 2:
        return _text("usage: /agent <name> set <key> <value>")
    key = _SET_ALIASES.get(kv[0].lower(), kv[0].lower())
    value = _coerce(key, " ".join(kv[1:]))
    try:
        updated = agent_store.load(name).set(**{key: value})
    except (ValueError, TypeError) as e:
        return _text(f"invalid {key}: {e}")
    agent_store.save(updated)
    return _text(f"{name}.{key} = {getattr(updated, key, value)}")


def _coerce(key: str, raw: str) -> Any:
    if key in ("pool_gb", "max_steps"):
        try:
            return int(raw)
        except ValueError:
            return raw
    if key == "show_tools":
        return raw.strip().lower() in ("true", "1", "yes", "on")
    if key == "tools":
        return [t.strip() for t in raw.replace(",", " ").split() if t.strip()]
    return raw


def _show(name: str) -> SlashResult:
    a = agent_store.load(name)
    return _text("\n".join(f"{k} = {v}" for k, v in a.to_dict().items() if k != "commands"))


def _edit(name: str) -> SlashResult:
    path = agent_store.agent_path(name)
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
    if not editor:
        return _text(f"set $EDITOR to edit, or edit the file directly: {path}")
    try:
        subprocess.run([editor, str(path)], check=False, timeout=600)
    except Exception as e:  # noqa: BLE001 — editor failures must not crash the REPL
        return _text(f"could not open editor ({e}); file: {path}")
    return _text(f"edited {path}")


def _delete(name: str, flags: list) -> SlashResult:
    purge = "--purge" in flags
    agent_store.delete(name, purge=purge)
    return _text(f"deleted {name}{' (+pool)' if purge else ''}")


def _cmd(name: str, args: list) -> SlashResult:
    from aether_agent import agent_commands
    if not args:
        return _text("usage: /agent <name> cmd add <cmd> = <template> | list | remove <cmd>")
    sub = args[0].lower()
    if sub == "list":
        cmds = agent_commands.list_commands(agent_store.load(name))
        if not cmds:
            return _text("(no commands)")
        return _text("\n".join(f"/{k}\t{v}" for k, v in sorted(cmds.items())))
    if sub == "remove":
        if len(args) < 2:
            return _text("usage: /agent <name> cmd remove <cmd>")
        updated = agent_commands.remove(agent_store.load(name), args[1])
        agent_store.save(updated)
        return _text(f"removed /{args[1]}")
    if sub == "add":
        rest = args[1:]
        if "=" not in rest:
            return _text("usage: /agent <name> cmd add <cmd> = <template>")
        eq = rest.index("=")
        cmd_name = " ".join(rest[:eq]).strip()
        template = " ".join(rest[eq + 1:]).strip()
        try:
            updated = agent_commands.add(agent_store.load(name), cmd_name, template)
        except (ValueError, TypeError) as e:
            return _text(f"could not add command: {e}")
        agent_store.save(updated)
        return _text(f"added /{cmd_name} = {template}")
    return _text("usage: /agent <name> cmd add <cmd> = <template> | list | remove <cmd>")


def _run_cmd(name: str, args: list) -> SlashResult:
    from aether_agent import agent_commands
    if not args:
        return _text("usage: /agent <name> run <cmd> [args]")
    cmd_name = args[0]
    cmds = agent_commands.list_commands(agent_store.load(name))
    if cmd_name not in cmds:
        return _text(f"no such command: /{cmd_name} (see /agent {name} cmd list)")
    task = agent_commands.expand(cmds[cmd_name], args[1:])
    return {"run_agent": {"name": name, "task": task}, "text": ""}


__all__ = ["dispatch_agent", "SlashResult"]
