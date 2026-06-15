# aether_agent/agent_profile.py
# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""The Agent — a named, shareable local agent profile + the library API.

An Agent is an immutable dataclass persisted as one JSON file per agent (see
``agent_store``). It carries the agent's model, its own memory-pool size, a
persona, a visual identity (accent / prompt / banner), and tool + permission
guardrails. No ``effort`` field — effort tiers are an AetherCloud-only concept.

Library use::

    a = Agent.create("jane", model="qwen2.5-coder:7b", pool_gb=10)
    for ev in a.run("summarize the repo"): ...
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterator

from aether_agent.adapter import DEFAULT_MODEL
from aether_agent.protocol import TOOLS

#: accent name -> ANSI SGR color code (cp1252-safe; used by validate + the REPL).
ACCENTS: dict[str, str] = {
    "cyan": "36", "green": "32", "magenta": "35", "yellow": "33",
    "blue": "34", "red": "31", "white": "37", "dim": "2",
}
_VERBOSITY = ("terse", "normal", "verbose")
_PERMISSION = ("ask", "auto", "skip")
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_MAX_STEPS_RANGE = (1, 500)


@dataclass(frozen=True)
class Agent:
    name: str
    model: str = DEFAULT_MODEL
    pool_gb: int = 5
    persona: str = "You are a helpful local coding agent."
    verbosity: str = "normal"
    show_tools: bool = True
    accent: str = "cyan"
    prompt: str = ""
    banner: str = ""
    tools: tuple[str, ...] = tuple(TOOLS)
    permission: str = "ask"
    max_steps: int = 40
    commands: dict[str, Any] = field(default_factory=dict)  # reserved for sub-project C

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Agent":
        """Build an Agent from a (possibly partial) dict, filling defaults and
        ignoring unknown keys (e.g. a stray ``effort`` from a future/foreign file)."""
        d = dict(data or {})
        tools = d.get("tools")
        name = str(d.get("name", ""))
        kw: dict[str, Any] = {
            "name": name,
            "model": str(d.get("model", DEFAULT_MODEL)) or DEFAULT_MODEL,
            "pool_gb": _as_int(d.get("pool_gb"), 5),
            "persona": str(d.get("persona", "You are a helpful local coding agent.")),
            "verbosity": str(d.get("verbosity", "normal")),
            "show_tools": bool(d.get("show_tools", True)),
            "accent": str(d.get("accent", "cyan")),
            "prompt": str(d.get("prompt", "") or f"{name} > "),
            "banner": str(d.get("banner", "")),
            "tools": tuple(tools) if isinstance(tools, (list, tuple)) else tuple(TOOLS),
            "permission": str(d.get("permission", "ask")),
            "max_steps": _as_int(d.get("max_steps"), 40),
            "commands": dict(d.get("commands") or {}),
        }
        return cls(**kw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "model": self.model, "pool_gb": self.pool_gb,
            "persona": self.persona, "verbosity": self.verbosity, "show_tools": self.show_tools,
            "accent": self.accent, "prompt": self.prompt, "banner": self.banner,
            "tools": list(self.tools), "permission": self.permission,
            "max_steps": self.max_steps, "commands": dict(self.commands),
        }

    def set(self, **fields: Any) -> "Agent":
        """Return a NEW validated copy with the given fields changed. ``memory`` is
        an alias for ``pool_gb`` (matches the `/agent set memory` command)."""
        if "memory" in fields:
            fields["pool_gb"] = _as_int(fields.pop("memory"), self.pool_gb)
        if "tools" in fields and isinstance(fields["tools"], list):
            fields["tools"] = tuple(fields["tools"])
        updated = replace(self, **fields)
        problems = validate(updated)
        if problems:
            raise ValueError("; ".join(problems))
        return updated

    # --- library API (proxies to agent_store / agent_runner; lazy imports) ----
    @classmethod
    def create(cls, name: str, **fields: Any) -> "Agent":
        from aether_agent import agent_store
        agent = cls.from_dict({"name": name, **fields})
        problems = validate(agent)
        if problems:
            raise ValueError("; ".join(problems))
        agent_store.create(agent)
        return agent

    @classmethod
    def load(cls, name: str) -> "Agent":
        from aether_agent import agent_store
        return agent_store.load(name)

    @classmethod
    def list(cls) -> list[str]:
        from aether_agent import agent_store
        return agent_store.list_agents()

    def save(self) -> None:
        from aether_agent import agent_store
        agent_store.save(self)

    def delete(self, purge: bool = False) -> None:
        from aether_agent import agent_store
        agent_store.delete(self.name, purge=purge)

    def run(self, task: str, **kw: Any) -> Iterator[dict[str, Any]]:
        from aether_agent import agent_runner
        yield from agent_runner.run(self, task, **kw)


def validate(agent: "Agent") -> list[str]:
    """Return a list of human-readable problems (empty = valid)."""
    p: list[str] = []
    if not _NAME_RE.match(agent.name or ""):
        p.append(f"name must be a slug [a-z0-9_-] (got {agent.name!r})")
    if not str(agent.model).strip():
        p.append("model must be a non-empty string")
    if not isinstance(agent.pool_gb, int) or agent.pool_gb < 1:
        p.append(f"pool_gb must be an int >= 1 (got {agent.pool_gb!r})")
    if agent.verbosity not in _VERBOSITY:
        p.append(f"verbosity must be one of {_VERBOSITY} (got {agent.verbosity!r})")
    if agent.permission not in _PERMISSION:
        p.append(f"permission must be one of {_PERMISSION} (got {agent.permission!r})")
    if agent.accent not in ACCENTS:
        p.append(f"accent must be one of {sorted(ACCENTS)} (got {agent.accent!r})")
    unknown = [t for t in agent.tools if t not in TOOLS]
    if unknown:
        p.append(f"tools must be a subset of the 8 canonical tools (unknown: {unknown})")
    lo, hi = _MAX_STEPS_RANGE
    if not isinstance(agent.max_steps, int) or not (lo <= agent.max_steps <= hi):
        p.append(f"max_steps must be an int in [{lo}, {hi}] (got {agent.max_steps!r})")
    if agent.commands:
        p.append("commands is reserved for a future release and must be empty")
    return p


def _as_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


__all__ = ["Agent", "validate", "ACCENTS"]
