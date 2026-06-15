# aether_agent/agent_runner.py
# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Run a task AS a custom agent.

Builds the agent's own ``Session`` (its persistent memory pool), an Ollama chat
on the agent's model, a persona system prompt, and a policy-wrapped ``Tools``
that enforces the agent's tool allowlist + permission mode — then drives
``agent.run_agent_events`` and yields its render-ready events. Session creation
is injectable (``session_factory``) so tests never touch numpy/disk.
"""
from __future__ import annotations

from typing import Any, Callable, Iterator, Optional

from aether_agent.adapter import OllamaChat
from aether_agent.agent import run_agent_events
from aether_agent.agent_profile import Agent
from aether_agent.tools import Tools, tool_schema

#: tools that change the workspace / run code — gated by permission mode.
DESTRUCTIVE = {"write_file", "run_shell", "git_commit"}
ConfirmFn = Callable[[str, dict], bool]


def _deny(_name: str, _args: dict) -> bool:
    return False


class _PolicyTools:
    """Wrap a real ``Tools`` with an allowlist + a permission gate. Same
    ``execute(name, args) -> str`` contract; refusals are returned as strings
    (never raised) so the agent loop reads them like any tool output."""

    def __init__(self, inner: Tools, allowed: set, permission: str, confirm: ConfirmFn) -> None:
        self._inner = inner
        self._allowed = set(allowed)
        self._permission = permission
        self._confirm = confirm
        self.test_cmd = getattr(inner, "test_cmd", "")  # the verify gate reads this attr

    def execute(self, name: str, args: dict) -> str:
        if name not in self._allowed:
            return f"[tool {name} not allowed for this agent]"
        if name in DESTRUCTIVE and self._permission != "skip":
            if not self._confirm(name, args):
                return f"[denied: {name} (permission={self._permission})]"
        return self._inner.execute(name, args)

    def run_tests(self, command: Optional[str] = None) -> str:
        return self._inner.run_tests(command)


def _default_session_factory(agent: Agent):
    from aether_agent import agent_store
    from aether_context import Session

    return Session(
        model=f"ollama/{agent.model}",
        pool_gb=agent.pool_gb,
        pool_dir=str(agent_store.agent_pool_dir(agent.name)),
        pool_mode="separate",
        pull=False,
        fallback_to_mock=True,
    )


def run(
    agent: Agent,
    task: str,
    *,
    cwd: str = ".",
    confirm: Optional[ConfirmFn] = None,
    llm: Any = None,
    session_factory: Optional[Callable[[Agent], Any]] = None,
    on_status: Optional[Callable[[str], None]] = None,
) -> Iterator[dict[str, Any]]:
    """Drive one task as ``agent``; yield render-ready events. ``llm`` and
    ``session_factory`` are injectable for tests (no real Ollama / Session)."""
    chat = llm if llm is not None else OllamaChat(model=agent.model)
    sess = (session_factory or _default_session_factory)(agent)
    allowed = set(agent.tools)
    schema = [s for s in tool_schema() if s["function"]["name"] in allowed]
    tools = _PolicyTools(Tools(cwd), allowed, agent.permission, confirm or _deny)
    try:
        yield from run_agent_events(
            task,
            llm=chat,
            tools=tools,
            cwd=cwd,
            pool_gb=agent.pool_gb,
            max_steps=agent.max_steps,
            sess=sess,
            schema=schema,
            system=agent.persona,
            git_checkpoint=False,
            verify_finish=False,
            on_status=on_status,
        )
    finally:
        try:
            sess.close()
        except Exception:  # noqa: BLE001 — teardown best-effort
            pass


__all__ = ["run", "DESTRUCTIVE", "ConfirmFn"]
