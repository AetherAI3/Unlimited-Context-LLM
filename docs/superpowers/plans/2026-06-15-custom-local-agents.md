# Custom Local Agents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user create named, shareable local agents — each with its own model, its own persistent Unlimited-Context memory pool, a persona, a visual identity, and tool/permission guardrails — and run/switch them with a unified `/agent` grammar plus an `Agent` library API.

**Architecture:** Four new stdlib-only modules in `aether_agent/` — `agent_profile.py` (the `Agent` dataclass + validation + the library API), `agent_store.py` (file-per-agent persistence + per-agent pool dirs), `agent_runner.py` (build a per-agent `Session`+policy-wrapped `Tools` and drive `run_agent_events`), `agent_slash.py` (the `/agent` grammar) — wired into `slash.py` and `repl.py`. One tiny additive change to `agent.py` (a `system=` param so a persona can override the default system prompt).

**Tech Stack:** Python 3.10+, stdlib only (`json`, `dataclasses`, `re`, `os`, `pathlib`, `shutil`, `subprocess`), `pytest`. No new runtime deps. Reuses `aether_context.Session`, `aether_agent.{adapter,tools,brains,agent,config,protocol,persona}`.

**Spec:** `docs/superpowers/specs/2026-06-15-custom-local-agents-design.md`
**Branch:** `feat/custom-local-agents` (off `main`, which has sub-project A).
**Gate:** `.venv/Scripts/python.exe -m pytest -q` AND `.venv/Scripts/python.exe -m ruff check aether_agent tests` (CI gates on ruff — no unused imports / F401).

---

## File Structure

| File | New/Mod | Responsibility |
|---|---|---|
| `aether_agent/agent_profile.py` | New | `Agent` frozen dataclass + defaults + `validate` + `from_dict`/`to_dict` + `set` + library classmethods (`create/load/list/delete`) + `run`. `ACCENTS` palette. |
| `aether_agent/agent_store.py` | New | `agents_dir`/`agent_path`/`agent_pool_dir` + `create/load/save/list_agents/delete/exists` (file-per-agent, fail-soft). |
| `aether_agent/agent_runner.py` | New | `_PolicyTools` (allowlist + permission) + `run(agent, task, ...)` -> event iterator over `run_agent_events`. |
| `aether_agent/agent_slash.py` | New | `dispatch_agent(line, *, active)` grammar + handlers (`/new-agent`, `/agents`, `/agent ...`). Config verbs pure; switch/run return action flags. |
| `aether_agent/agent.py` | Mod | add `system: Optional[str] = None` to `run_agent_events` (persona override; default = `SYSTEM_PROMPT`). |
| `aether_agent/slash.py` | Mod | register `new-agent`/`agents`/`agent`; add `active_agent` to `SlashContext`; help lines; move old cloud orchestrator list/switch to `/orchestrators`/`/orchestrator`. |
| `aether_agent/repl.py` | Mod | active-agent state; apply persona/accent/prompt/banner on switch; run turns as the active agent; handle `switch_agent`/`run_agent` flags. |
| `aether_agent/__init__.py` | Mod | re-export `Agent`. |

Reuse (no duplication): `config.config_dir()` (base of `agents_dir`), `protocol.TOOLS` (the canonical 8), `tools.tool_schema()`/`Tools`, `adapter.OllamaChat`, `agent.run_agent_events`, `Session`.

---

## Task 1: `agent_profile.py` — the Agent dataclass + validation

**Files:**
- Create: `aether_agent/agent_profile.py`
- Test: `tests/test_agent_profile.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_profile.py
from aether_agent import agent_profile
from aether_agent.agent_profile import Agent


def test_from_dict_fills_defaults_and_ignores_effort():
    a = Agent.from_dict({"name": "jane", "effort": "codepro"})  # effort ignored
    assert a.name == "jane"
    assert a.model  # a default model
    assert a.pool_gb == 5
    assert a.permission == "ask"
    assert set(a.tools)  # defaults to the canonical 8
    assert not hasattr(a, "effort")


def test_validate_catches_bad_fields():
    bad = Agent.from_dict({
        "name": "Bad Name!", "pool_gb": 0, "permission": "nope",
        "tools": ["read_file", "made_up_tool"], "max_steps": 9999, "accent": "chartreuse",
    })
    problems = agent_profile.validate(bad)
    joined = " ".join(problems).lower()
    assert "name" in joined
    assert "pool_gb" in joined or "pool" in joined
    assert "permission" in joined
    assert "tool" in joined
    assert "max_steps" in joined or "step" in joined
    assert "accent" in joined


def test_valid_agent_has_no_problems():
    a = Agent.from_dict({"name": "jane"})
    assert agent_profile.validate(a) == []


def test_roundtrip_and_set_returns_new_validated_copy():
    a = Agent.from_dict({"name": "jane"})
    d = a.to_dict()
    assert d["name"] == "jane" and "commands" in d and "effort" not in d
    b = a.set(pool_gb=10, accent="green")
    assert b.pool_gb == 10 and b.accent == "green"
    assert a.pool_gb == 5  # original unchanged (immutable)


def test_set_with_invalid_value_raises():
    a = Agent.from_dict({"name": "jane"})
    try:
        a.set(permission="bogus")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "permission" in str(e).lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_profile.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aether_agent.agent_profile'`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_profile.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add aether_agent/agent_profile.py tests/test_agent_profile.py
git commit -m "feat(agents): agent_profile — Agent dataclass + validation + library API surface"
```

---

## Task 2: `agent_store.py` — file-per-agent persistence

**Files:**
- Create: `aether_agent/agent_store.py`
- Test: `tests/test_agent_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_store.py
import pytest

from aether_agent import agent_store
from aether_agent.agent_profile import Agent


@pytest.fixture(autouse=True)
def _tmp_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))


def test_create_load_list_exists():
    a = Agent.from_dict({"name": "jane", "pool_gb": 7})
    agent_store.create(a)
    assert agent_store.exists("jane")
    assert agent_store.list_agents() == ["jane"]
    loaded = agent_store.load("jane")
    assert loaded.name == "jane" and loaded.pool_gb == 7


def test_create_duplicate_raises():
    agent_store.create(Agent.from_dict({"name": "jane"}))
    with pytest.raises(Exception):
        agent_store.create(Agent.from_dict({"name": "jane"}))


def test_load_missing_raises():
    with pytest.raises(Exception):
        agent_store.load("nope")


def test_pool_dir_path_layout():
    p = agent_store.agent_pool_dir("jane")
    assert p.name == "pool" and p.parent.name == "jane"
    assert "agents" in str(p)


def test_delete_keeps_pool_unless_purge():
    a = Agent.from_dict({"name": "jane"})
    agent_store.create(a)
    pool = agent_store.agent_pool_dir("jane")
    pool.mkdir(parents=True, exist_ok=True)
    agent_store.delete("jane", purge=False)
    assert not agent_store.exists("jane")
    assert pool.exists()  # pool kept
    agent_store.create(a)
    agent_store.delete("jane", purge=True)
    assert not pool.exists()


def test_corrupt_file_fails_soft():
    agent_store.agent_path("broken").write_text("{ not json", encoding="utf-8")
    with pytest.raises(Exception):
        agent_store.load("broken")
    assert "broken" in agent_store.list_agents()  # listing must not crash
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_store.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aether_agent.agent_store'`

- [ ] **Step 3: Write minimal implementation**

```python
# aether_agent/agent_store.py
# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""File-per-agent persistence.

Definitions live at ``<config>/agents/<name>.json``; each agent's memory pool is
separate local state at ``<config>/agents/<name>/pool/``. Sharing an agent =
copying its one ``<name>.json``. Honors ``AETHER_CONFIG_DIR`` (via ``config``).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from aether_agent.agent_profile import Agent
from aether_agent.config import config_dir


def agents_dir() -> Path:
    return Path(config_dir()) / "agents"


def agent_path(name: str) -> Path:
    d = agents_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.json"


def agent_pool_dir(name: str) -> Path:
    return agents_dir() / name / "pool"


def exists(name: str) -> bool:
    return agent_path(name).is_file()


def list_agents() -> list[str]:
    d = agents_dir()
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


def create(agent: Agent) -> None:
    if exists(agent.name):
        raise FileExistsError(f"agent already exists: {agent.name}")
    save(agent)


def save(agent: Agent) -> None:
    agent_path(agent.name).write_text(
        json.dumps(agent.to_dict(), indent=2) + "\n", encoding="utf-8"
    )


def load(name: str) -> Agent:
    path = agent_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"no such agent: {name}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        raise ValueError(f"agent file is corrupt: {name} ({e})") from e
    return Agent.from_dict(data)


def delete(name: str, *, purge: bool = False) -> None:
    path = agent_path(name)
    if path.is_file():
        path.unlink()
    if purge:
        pool_parent = agents_dir() / name
        if pool_parent.is_dir():
            shutil.rmtree(pool_parent, ignore_errors=True)


__all__ = [
    "agents_dir", "agent_path", "agent_pool_dir",
    "exists", "list_agents", "create", "save", "load", "delete",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_store.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add aether_agent/agent_store.py tests/test_agent_store.py
git commit -m "feat(agents): agent_store — file-per-agent CRUD + per-agent pool dirs"
```

---

## Task 3: `run_agent_events` persona override (`agent.py`)

**Files:**
- Modify: `aether_agent/agent.py` (add a `system` param)
- Test: `tests/test_run_agent_events_system.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_agent_events_system.py
from aether_agent import agent as agent_mod
from aether_agent.tools import Tools


class _FakeLLM:
    def __init__(self):
        self.seen_system = None

    def chat(self, messages, tools=None):
        self.seen_system = messages[0]["content"]
        return {"role": "assistant", "content": "ok", "tool_calls": []}


def test_system_param_overrides_default_persona(tmp_path):
    llm = _FakeLLM()
    tools = Tools(str(tmp_path), test_cmd="")  # empty test_cmd -> verify gate is a no-op
    events = list(agent_mod.run_agent_events(
        "hi", llm=llm, tools=tools, system="You are Jane.", verify_finish=False,
    ))
    assert llm.seen_system == "You are Jane."
    assert any(e["type"] == "done" for e in events)


def test_system_defaults_to_builtin_when_omitted(tmp_path):
    from aether_agent.persona import SYSTEM_PROMPT
    llm = _FakeLLM()
    tools = Tools(str(tmp_path), test_cmd="")
    list(agent_mod.run_agent_events("hi", llm=llm, tools=tools, verify_finish=False))
    assert llm.seen_system == SYSTEM_PROMPT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_run_agent_events_system.py -q`
Expected: FAIL — `TypeError: run_agent_events() got an unexpected keyword argument 'system'`

- [ ] **Step 3: Write minimal implementation** (edit `aether_agent/agent.py`)

Add the param to the `run_agent_events` signature (after the `on_status` line):

```python
    on_status: Optional[Callable[[str], None]] = None,
    system: Optional[str] = None,
) -> Iterator[dict[str, Any]]:
```

Change the messages init (currently `messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}, ...]`) to:

```python
    sys_prompt = system if system is not None else SYSTEM_PROMPT
    messages: list[dict] = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": task},
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_run_agent_events_system.py tests/test_brains.py -q`
Expected: PASS (new green; existing brains tests unaffected)

- [ ] **Step 5: Commit**

```bash
git add aether_agent/agent.py tests/test_run_agent_events_system.py
git commit -m "feat(agents): run_agent_events accepts a system= persona override (default unchanged)"
```

---

## Task 4: `agent_runner.py` — per-agent Session + policy tools

**Files:**
- Create: `aether_agent/agent_runner.py`
- Test: `tests/test_agent_runner.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_runner.py
from aether_agent import agent_runner
from aether_agent.agent_profile import Agent
from aether_agent.tools import Tools


def test_policy_tools_blocks_disallowed_and_allows_allowed(tmp_path):
    inner = Tools(str(tmp_path))
    pt = agent_runner._PolicyTools(inner, allowed={"read_file"}, permission="skip", confirm=lambda n, a: True)
    assert "not allowed" in pt.execute("write_file", {"path": "x", "content": "y"}).lower()
    out = pt.execute("read_file", {"path": "missing"})
    assert "no such file" in out.lower()  # delegated to inner


def test_policy_tools_ask_denies_destructive_without_confirm(tmp_path):
    inner = Tools(str(tmp_path))
    pt = agent_runner._PolicyTools(inner, allowed={"write_file"}, permission="ask", confirm=lambda n, a: False)
    assert "denied" in pt.execute("write_file", {"path": "x", "content": "y"}).lower()
    pt2 = agent_runner._PolicyTools(inner, allowed={"write_file"}, permission="ask", confirm=lambda n, a: True)
    assert "wrote" in pt2.execute("write_file", {"path": "x.txt", "content": "y"}).lower()


def test_run_builds_session_with_agent_pool_and_streams(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    captured = {}

    class _FakeSession:
        def remember(self, *a, **k): pass
        def status_dict(self): return {}
        def close(self): captured["closed"] = True

    def fake_session_factory(agent):
        captured["pool_gb"] = agent.pool_gb
        captured["model"] = agent.model
        return _FakeSession()

    class _FakeLLM:
        def chat(self, messages, tools=None):
            captured["system"] = messages[0]["content"]
            captured["schema_names"] = [t["function"]["name"] for t in (tools or [])]
            return {"role": "assistant", "content": "done", "tool_calls": []}

    a = Agent.from_dict({"name": "jane", "pool_gb": 9, "persona": "You are Jane.",
                         "tools": ["read_file", "web_search"], "permission": "skip"})
    events = list(agent_runner.run(
        a, "hello", cwd=str(tmp_path), llm=_FakeLLM(), session_factory=fake_session_factory,
    ))
    assert captured["pool_gb"] == 9 and captured["model"] == a.model
    assert captured["system"] == "You are Jane."
    assert set(captured["schema_names"]) == {"read_file", "web_search"}
    assert any(e["type"] == "done" for e in events)
    assert captured.get("closed") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_runner.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aether_agent.agent_runner'`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_runner.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add aether_agent/agent_runner.py tests/test_agent_runner.py
git commit -m "feat(agents): agent_runner — per-agent Session + allowlist/permission policy tools"
```

---

## Task 5: `agent_slash.py` — the `/agent` grammar

**Files:**
- Create: `aether_agent/agent_slash.py`
- Test: `tests/test_agent_slash.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_slash.py
import pytest

from aether_agent import agent_slash
from aether_agent.agent_profile import Agent


@pytest.fixture(autouse=True)
def _tmp_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))


def _mk(name="jane"):
    from aether_agent import agent_store
    agent_store.create(Agent.from_dict({"name": name}))


def test_new_agent_creates():
    res = agent_slash.dispatch_agent("/new-agent jane", active="")
    assert "jane" in res["text"]
    from aether_agent import agent_store
    assert agent_store.exists("jane")


def test_new_agent_duplicate_reports():
    _mk()
    res = agent_slash.dispatch_agent("/new-agent jane", active="")
    assert "exist" in res["text"].lower()


def test_agents_lists_with_active_marker():
    _mk("jane"); _mk("neo")
    res = agent_slash.dispatch_agent("/agents", active="neo")
    assert "jane" in res["text"] and "neo" in res["text"]


def test_agent_switch_returns_flag():
    _mk()
    res = agent_slash.dispatch_agent("/agent jane", active="")
    assert res.get("switch_agent") == "jane"


def test_agent_run_returns_flag():
    _mk()
    res = agent_slash.dispatch_agent('/agent jane fix the bug', active="")
    assert res.get("run_agent") == {"name": "jane", "task": "fix the bug"}


def test_agent_set_persists():
    _mk()
    res = agent_slash.dispatch_agent("/agent jane set memory 10", active="")
    assert "10" in res["text"]
    assert Agent.load("jane").pool_gb == 10


def test_agent_set_invalid_reports_not_crash():
    _mk()
    res = agent_slash.dispatch_agent("/agent jane set permission bogus", active="")
    assert "permission" in res["text"].lower()
    assert Agent.load("jane").permission == "ask"  # unchanged


def test_unknown_agent_is_friendly():
    res = agent_slash.dispatch_agent("/agent ghost", active="")
    assert "no" in res["text"].lower() or "unknown" in res["text"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_slash.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aether_agent.agent_slash'`

- [ ] **Step 3: Write minimal implementation**

```python
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
_VERBS = {"set", "show", "edit", "delete"}
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
    names = agent_store.list_agents()
    if not names:
        return _text("(no agents yet — create one with /new-agent <name>)")
    lines = ["agents (* active):"]
    for n in names:
        try:
            a = agent_store.load(n)
            mark = "*" if n == active else " "
            lines.append(f"{mark} {n}\t{a.model}\t{a.pool_gb}GB")
        except (ValueError, OSError):
            lines.append(f"  {n}\t(corrupt file)")
    lines.append("switch: /agent <name>   run: /agent <name> <task>")
    return _text("\n".join(lines))


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


__all__ = ["dispatch_agent", "SlashResult"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_slash.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add aether_agent/agent_slash.py tests/test_agent_slash.py
git commit -m "feat(agents): agent_slash — /new-agent, /agents, /agent <name> <verb> grammar"
```

---

## Task 6: wire `/agent*` into `slash.py`

**Files:**
- Modify: `aether_agent/slash.py`
- Test: `tests/test_slash_agents.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_slash_agents.py
import pytest

from aether_agent.slash import SlashContext, dispatch


@pytest.fixture(autouse=True)
def _tmp_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))


def test_slash_routes_agent_commands():
    res = dispatch(SlashContext(api=None), "/new-agent jane")
    assert "jane" in res["text"]
    res2 = dispatch(SlashContext(api=None, active_agent=""), "/agent jane")
    assert res2.get("switch_agent") == "jane"
    res3 = dispatch(SlashContext(api=None), "/agents")
    assert "jane" in res3["text"]


def test_help_lists_agent_commands():
    res = dispatch(SlashContext(api=None), "/help")
    assert "/agent" in res["text"] and "/new-agent" in res["text"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_slash_agents.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'active_agent'`

- [ ] **Step 3: Write minimal implementation** (edit `aether_agent/slash.py`)

(a) Add `active_agent` to `SlashContext` (after the `ollama: Any = None` line):

```python
    active_agent: str = ""  # name of the active custom agent (B); "" = none
```

(b) Rename the existing cloud-orchestrator handlers so `/agents` + `/agent` can mean custom agents:
- rename the existing `def _agents(...)` → `def _agents_orch(...)` (keep its body verbatim).
- rename the existing `def _agent(...)` → `def _agent_orch(...)` (keep its body verbatim).

(c) Add the three new handlers (after `_web`):

```python
def _new_agent(ctx: SlashContext, arg: str) -> SlashResult:
    from aether_agent.agent_slash import dispatch_agent
    return dispatch_agent(f"/new-agent {arg}".strip(), active=ctx.active_agent)


def _agents(ctx: SlashContext, arg: str) -> SlashResult:
    from aether_agent.agent_slash import dispatch_agent
    return dispatch_agent("/agents", active=ctx.active_agent)


def _agent_cmd(ctx: SlashContext, arg: str) -> SlashResult:
    from aether_agent.agent_slash import dispatch_agent
    return dispatch_agent(f"/agent {arg}".strip(), active=ctx.active_agent)
```

(d) Update `REGISTRY` — replace the old `"agents": _agents,` and `"agent": _agent,` entries with:

```python
    "agents": _agents,             # B: list custom local agents
    "agent": _agent_cmd,           # B: /agent <name> <verb>
    "new-agent": _new_agent,       # B: create
    "orchestrators": _agents_orch, # cloud orchestrators (was /agents)
    "orchestrator": _agent_orch,   # cloud orchestrator switch (was /agent)
```

(e) Update `_HELP_LINES` — replace the two orchestrator lines (`/agents`, `/agent <id>`) with:

```python
    "/agents            list your custom agents",
    "/agent <name>      switch agent (or: /agent <name> <task>)",
    "/new-agent <name>  create a custom agent",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_slash_agents.py tests/test_slash.py tests/test_slash_ollama.py -q`
Expected: PASS — if `tests/test_slash.py` asserted the old `/agents`/`/agent` orchestrator text, update those two assertions to use `/orchestrators` / `/orchestrator`.

- [ ] **Step 5: Commit**

```bash
git add aether_agent/slash.py tests/test_slash_agents.py tests/test_slash.py
git commit -m "feat(agents): register /new-agent /agents /agent; move cloud orchestrators to /orchestrators"
```

---

## Task 7: `repl.py` — active agent + run-as-agent + UX

**Files:**
- Modify: `aether_agent/repl.py`
- Test: `tests/test_repl_agents.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_repl_agents.py
import io

import pytest

from aether_agent import repl
from aether_agent.agent_profile import Agent


@pytest.fixture(autouse=True)
def _tmp_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))


def test_apply_agent_sets_prompt_accent_persona():
    a = Agent.from_dict({"name": "jane", "accent": "green", "prompt": "jane > ", "banner": "~J~"})
    state = repl._apply_agent(a)
    assert state["prompt"] == "jane > "
    assert "32" in state["accent_ansi"]  # green = 32
    assert state["banner"] == "~J~"


def test_run_agent_flag_streams_via_runner(monkeypatch):
    from aether_agent import agent_store
    agent_store.create(Agent.from_dict({"name": "jane"}))
    seen = {}

    def fake_run(agent, task, **kw):
        seen["name"] = agent.name
        seen["task"] = task
        yield {"type": "done", "text": "ran"}

    monkeypatch.setattr("aether_agent.agent_runner.run", fake_run)
    out = io.StringIO()
    repl._handle_agent_action({"run_agent": {"name": "jane", "task": "do x"}}, out)
    assert seen == {"name": "jane", "task": "do x"}
    assert "ran" in out.getvalue()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_repl_agents.py -q`
Expected: FAIL — `AttributeError: module 'aether_agent.repl' has no attribute '_apply_agent'`

- [ ] **Step 3: Write minimal implementation** (edit `aether_agent/repl.py`)

(a) Add import:

```python
from aether_agent.agent_profile import ACCENTS, Agent
```

(b) Add helpers (module level, near `_make_ctx`):

```python
def _apply_agent(agent: Agent) -> dict[str, Any]:
    """Compute the REPL UX state for an active agent (pure; cp1252-safe ANSI)."""
    return {
        "name": agent.name,
        "prompt": agent.prompt or f"{agent.name} > ",
        "accent_ansi": f"\x1b[{ACCENTS.get(agent.accent, '36')}m",
        "banner": agent.banner,
        "persona": agent.persona,
    }


def _handle_agent_action(res: dict[str, Any], out: Any) -> None:
    """Execute a /agent run action: stream the agent's turn via the runner."""
    from aether_agent import agent_runner, agent_store

    run = res.get("run_agent")
    if not run:
        return
    try:
        agent = agent_store.load(run["name"])
    except (ValueError, OSError) as e:
        _safe_write(out, f"\n[x] {e}\n")
        return
    for ev in agent_runner.run(agent, run["task"], confirm=lambda n, a: False):
        _render_event(ev, out)
```

(c) In `main()`, add active-agent state right after `ctx = _make_ctx(...)`:

```python
    active_agent: Optional[Agent] = None
    prompt_str = _PROMPT
```

(d) Change the prompt read in the loop from `input(_PROMPT)` to `input(prompt_str)`.

(e) In the slash-result handling (inside `if line.startswith("/"):`), AFTER the `text` write and BEFORE
`if res.get("setup")`, insert:

```python
                if res.get("switch_agent"):
                    try:
                        active_agent = Agent.load(res["switch_agent"])
                        ctx.active_agent = active_agent.name
                        st = _apply_agent(active_agent)
                        prompt_str = st["prompt"]
                        if st["banner"]:
                            _safe_write(out, st["accent_ansi"] + st["banner"] + "\x1b[0m\n")
                    except (ValueError, OSError) as e:
                        _safe_write(out, f"\n[x] {e}\n")
                if res.get("run_agent"):
                    _handle_agent_action(res, out)
```

(f) In the normal-turn branch (the non-slash path), run AS the active agent when one is set. Replace the
existing brain build+run lines with:

```python
            if active_agent is not None:
                from aether_agent import agent_runner
                for ev in agent_runner.run(active_agent, line, confirm=lambda n, a: False):
                    _render_event(ev, out)
                out.write("\n")
                continue
            brain = select_brain(
                authed=store.get() is not None,
                backend=backend,
                api=api,
                model=ctx.model or chosen_model or "",
            )
            _run_turn(brain, line, out)
            out.write("\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_repl_agents.py tests/test_repl_preflight.py -q`
Expected: PASS (new green; existing repl tests still green)

- [ ] **Step 5: Commit**

```bash
git add aether_agent/repl.py tests/test_repl_agents.py
git commit -m "feat(agents): REPL active-agent state — switch applies UX, turns run as the agent"
```

---

## Task 8: `Agent` library re-export + end-to-end library test

**Files:**
- Modify: `aether_agent/__init__.py`
- Test: `tests/test_agent_library.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_library.py
import pytest

from aether_agent import Agent  # the public library API


@pytest.fixture(autouse=True)
def _tmp_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))


def test_create_load_list_set_save_delete_roundtrip():
    a = Agent.create("jane", model="qwen2.5-coder:7b", pool_gb=8)
    assert a.pool_gb == 8
    assert Agent.list() == ["jane"]
    b = Agent.load("jane").set(pool_gb=12)
    b.save()
    assert Agent.load("jane").pool_gb == 12
    b.delete()
    assert Agent.list() == []


def test_run_streams_via_runner(monkeypatch):
    Agent.create("jane")

    def fake_run(agent, task, **kw):
        yield {"type": "done", "text": f"{agent.name}:{task}"}

    monkeypatch.setattr("aether_agent.agent_runner.run", fake_run)
    out = list(Agent.load("jane").run("hi"))
    assert out[-1]["text"] == "jane:hi"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_library.py -q`
Expected: FAIL — `ImportError: cannot import name 'Agent' from 'aether_agent'`

- [ ] **Step 3: Write minimal implementation** (edit `aether_agent/__init__.py`)

Append (do not remove existing content):

```python
from aether_agent.agent_profile import Agent  # noqa: E402  (public library API)
```

If `__init__.py` defines `__all__`, append `"Agent"` to it.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_library.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add aether_agent/__init__.py tests/test_agent_library.py
git commit -m "feat(agents): export Agent as the public library API (from aether_agent import Agent)"
```

---

## Task 9: Full-suite green + ruff + push

**Files:** full suite

- [ ] **Step 1: Run ruff (CI gates on it)**

Run: `.venv/Scripts/python.exe -m ruff check aether_agent tests`
Expected: `All checks passed!` — remove any F401/unused import and re-run.

- [ ] **Step 2: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS — all prior tests (476+) plus the ~32 new B tests. Zero regressions. If `test_slash.py`
asserted the old `/agents`/`/agent` orchestrator output, update it to `/orchestrators`/`/orchestrator`.

- [ ] **Step 3: Boot smoke (no Ollama needed)**

Run (bash): `AETHER_CONFIG_DIR="$(mktemp -d)" .venv/Scripts/python.exe -c "from aether_agent import Agent; a=Agent.create('jane', pool_gb=7); print('created', a.name, a.pool_gb); print('list', Agent.list())"`
Expected: `created jane 7` / `list ['jane']` — no crash.

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "test(agents): full-suite + ruff green after custom local agents (sub-project B)"
```

- [ ] **Step 5: Push the branch**

```bash
git push -u origin feat/custom-local-agents
```

---

## Self-review (author checklist — completed)

- **Spec coverage:** Agent dataclass + validation + no-effort (T1) · file-per-agent + per-agent pool (T2) · persona override seam (T3) · per-agent Session + allowlist + permission (T4) · `/agent` grammar incl. switch/run/set/show/edit/delete + `/new-agent` + `/agents` (T5,T6) · customizable UX applied on switch + run-as-agent (T7) · library API `from aether_agent import Agent` (T1 methods + T8 export) · offline tests per unit (every task). All spec sections map to a task.
- **Placeholder scan:** no TBD/TODO; every code step is complete. The `commands` field is reserved-and-validated-empty (C), not a placeholder.
- **Type consistency:** `Agent` fields + `from_dict`/`to_dict`/`set`/`validate` consistent across T1–T8; `agent_store` names (`create/load/save/list_agents/delete/exists/agents_dir/agent_path/agent_pool_dir`) identical across T2,T4,T5,T7,T8; `agent_runner.run(agent, task, *, cwd, confirm, llm, session_factory, on_status)` + `_PolicyTools` consistent across T4,T7,T8; `dispatch_agent(line, *, active)` + `switch_agent`/`run_agent` flag shapes consistent across T5,T6,T7; `run_agent_events(..., system=)` consistent across T3,T4. `ACCENTS` shared by T1 (validate) + T7 (repl).
- **Scope:** B only — C (`commands` authoring), D (multi-agent columns), full theme, effort tiers all explicitly out.
```
