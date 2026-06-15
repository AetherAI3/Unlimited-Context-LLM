# aether_agent/agent_commands.py
# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Custom commands — per-agent prompt-macros (sub-project C).

A command is a string template stored in ``Agent.commands`` that expands into an
agent turn. Placeholders: ``$1``..``$9`` (positional args), ``$*``/``$@`` (all
args), ``$$`` -> literal ``$``. Pure expansion — no shell, no eval. Authoring
(add/remove) returns a NEW validated ``Agent`` (immutability).
"""
from __future__ import annotations

import re

from aether_agent.agent_profile import Agent, is_valid_command_name

_POS_RE = re.compile(r"\$([1-9])")
_SENTINEL = "\x00"


def valid_name(name: str) -> bool:
    return is_valid_command_name(name)


def expand(template: str, args: list[str]) -> str:
    """Expand a command template against positional args (pure; no shell/eval)."""
    all_args = " ".join(args)
    s = (template or "").replace("$$", _SENTINEL)  # protect literal $ first
    s = s.replace("$*", all_args).replace("$@", all_args)
    s = _POS_RE.sub(lambda m: args[int(m.group(1)) - 1] if int(m.group(1)) - 1 < len(args) else "", s)
    return s.replace(_SENTINEL, "$")


def add(agent: Agent, name: str, template: str) -> Agent:
    """Return a NEW agent with command ``name`` set. Raises ValueError on a bad
    name or empty template (validation also runs inside Agent.set)."""
    if not valid_name(name):
        raise ValueError(f"invalid command name: {name!r} (must be a slug and not a built-in)")
    if not str(template).strip():
        raise ValueError("command template must be a non-empty string")
    cmds = dict(agent.commands)
    cmds[name] = template
    return agent.set(commands=cmds)


def remove(agent: Agent, name: str) -> Agent:
    cmds = dict(agent.commands)
    cmds.pop(name, None)
    return agent.set(commands=cmds)


def list_commands(agent: Agent) -> dict:
    return dict(agent.commands)


__all__ = ["valid_name", "expand", "add", "remove", "list_commands"]
