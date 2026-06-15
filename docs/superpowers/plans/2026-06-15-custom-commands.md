# Custom Local Commands Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users (and the agent, via a gated `define_command` tool) save reusable prompt-macros on an agent and invoke them as bare slash commands (`/review src/api.py`), expanding into an agent turn.

**Architecture:** One new stdlib module `aether_agent/agent_commands.py` (name-validation + template expansion + add/remove/list), un-reserve B's `commands` field into a validated `{name: template}` map, resolve custom slashes in `slash.dispatch` after built-ins, and a local-only `define_command` tool in `agent_runner` (kept OUT of the 8-tool bridge protocol).

**Tech Stack:** Python 3.10+, stdlib only (`re`, `json`, `dataclasses`), `pytest`. No new deps. Reuses B (`agent_profile`, `agent_store`, `agent_slash`, `agent_runner`, `slash`, `tools`, `protocol`).

**Spec:** `docs/superpowers/specs/2026-06-15-custom-commands-design.md`
**Branch:** `feat/custom-commands` (off `main`).
**Gate:** `.venv/Scripts/python.exe -m pytest -q` AND `.venv/Scripts/python.exe -m ruff check aether_agent tests`.

---

## File Structure

| File | New/Mod | Responsibility |
|---|---|---|
| `aether_agent/agent_profile.py` | Mod | `RESERVED_COMMANDS` + `is_valid_command_name()` + `ALLOWED_AGENT_TOOLS`; `commands` validated as `{name: template}`; tools validated against `ALLOWED_AGENT_TOOLS`. |
| `aether_agent/agent_commands.py` | New | `valid_name` (re-uses profile) + `expand(template, args)` + `add/remove/list_commands`. |
| `aether_agent/agent_slash.py` | Mod | `cmd add|list|remove` verb + `run <cmd>` verb. |
| `aether_agent/slash.py` | Mod | `dispatch` resolves a bare `/<word>` against the active agent's commands after built-ins. |
| `aether_agent/agent_runner.py` | Mod | inject + handle the local-only `define_command` tool when allowed. |

Ordering matters: Task 1 (profile validation) MUST precede Task 2 (`agent_commands.add` calls `Agent.set(commands=...)`, which validates).

---

## Task 1: `agent_profile.py` — un-reserve `commands` + allow `define_command`

**Files:**
- Modify: `aether_agent/agent_profile.py`
- Test: `tests/test_agent_profile_commands.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_profile_commands.py
from aether_agent import agent_profile
from aether_agent.agent_profile import Agent


def test_commands_map_validates():
    good = Agent.from_dict({"name": "jane", "commands": {"review": "Review $1 for bugs"}})
    assert agent_profile.validate(good) == []


def test_reserved_or_empty_command_rejected():
    bad = Agent.from_dict({"name": "jane", "commands": {"help": "x"}})  # reserved name
    assert any("command" in p.lower() for p in agent_profile.validate(bad))
    empty = Agent.from_dict({"name": "jane", "commands": {"ok": "   "}})  # empty template
    assert any("template" in p.lower() for p in agent_profile.validate(empty))


def test_define_command_allowed_in_tools():
    a = Agent.from_dict({"name": "jane", "tools": ["read_file", "define_command"]})
    assert agent_profile.validate(a) == []  # define_command is an allowed agent tool


def test_is_valid_command_name():
    assert agent_profile.is_valid_command_name("review")
    assert not agent_profile.is_valid_command_name("help")        # reserved
    assert not agent_profile.is_valid_command_name("Bad Name")    # not a slug
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_profile_commands.py -q`
Expected: FAIL — `AttributeError: module 'aether_agent.agent_profile' has no attribute 'is_valid_command_name'` (and the reserved-name test fails because B still requires `commands` empty).

- [ ] **Step 3: Write minimal implementation** (edit `aether_agent/agent_profile.py`)

`re` is already imported at the top of `agent_profile.py`. Add near the other module constants (after `_MAX_STEPS_RANGE`):

```python
#: command names a custom command may NOT take (would shadow a built-in slash).
RESERVED_COMMANDS = frozenset({
    "help", "models", "model", "agents", "agent", "new-agent", "orchestrator",
    "orchestrators", "tier", "audit", "web", "clear", "exit", "quit",
    "pull", "doctor", "serve", "setup", "config",
})
_CMD_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

#: tools an AGENT may list (the 8 bridge tools + the local-only authoring tool).
ALLOWED_AGENT_TOOLS = tuple(TOOLS) + ("define_command",)


def is_valid_command_name(name: str) -> bool:
    return bool(_CMD_NAME_RE.match(name or "")) and name not in RESERVED_COMMANDS
```

Change the tools check in `validate()` from `if t not in TOOLS` to:

```python
    unknown = [t for t in agent.tools if t not in ALLOWED_AGENT_TOOLS]
    if unknown:
        p.append(f"tools must be a subset of {sorted(ALLOWED_AGENT_TOOLS)} (unknown: {unknown})")
```

Replace the `if agent.commands:` block in `validate()` with:

```python
    for cname, ctmpl in (agent.commands or {}).items():
        if not is_valid_command_name(cname):
            p.append(f"command name {cname!r} must be a slug and not a reserved built-in")
        if not isinstance(ctmpl, str) or not ctmpl.strip():
            p.append(f"command {cname!r} template must be a non-empty string")
```

Add `RESERVED_COMMANDS`, `ALLOWED_AGENT_TOOLS`, `is_valid_command_name` to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_profile_commands.py tests/test_agent_profile.py -q`
Expected: PASS (new green; the existing `test_agent_profile.py` still green — `commands` defaults to `{}`, which passes the loop trivially)

- [ ] **Step 5: Commit**

```bash
git add aether_agent/agent_profile.py tests/test_agent_profile_commands.py
git commit -m "feat(cmds): un-reserve Agent.commands as a validated {name:template} map + allow define_command tool"
```

---

## Task 2: `agent_commands.py` — name validation + expansion + add/remove/list

**Files:**
- Create: `aether_agent/agent_commands.py`
- Test: `tests/test_agent_commands.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_commands.py
import pytest

from aether_agent import agent_commands
from aether_agent.agent_profile import Agent


def test_valid_name():
    assert agent_commands.valid_name("review")
    assert not agent_commands.valid_name("help")       # reserved
    assert not agent_commands.valid_name("Bad Name")   # not a slug


def test_expand_positional_all_and_dollar():
    assert agent_commands.expand("Review $1 and $2", ["a.py", "b.py"]) == "Review a.py and b.py"
    assert agent_commands.expand("all: $*", ["x", "y", "z"]) == "all: x y z"
    assert agent_commands.expand("all: $@", ["x", "y"]) == "all: x y"
    assert agent_commands.expand("cost $$5 for $1", ["a"]) == "cost $5 for a"
    assert agent_commands.expand("missing $3 here", ["a"]) == "missing  here"  # absent -> ""
    assert agent_commands.expand("no placeholders", []) == "no placeholders"


def test_add_remove_list_roundtrip():
    a = Agent.from_dict({"name": "jane"})
    a = agent_commands.add(a, "review", "Review $1 for bugs")
    assert agent_commands.list_commands(a) == {"review": "Review $1 for bugs"}
    a = agent_commands.remove(a, "review")
    assert agent_commands.list_commands(a) == {}


def test_add_rejects_bad_name_and_empty_template():
    a = Agent.from_dict({"name": "jane"})
    with pytest.raises(ValueError):
        agent_commands.add(a, "help", "x")        # reserved
    with pytest.raises(ValueError):
        agent_commands.add(a, "ok", "   ")        # empty template
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_commands.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aether_agent.agent_commands'`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_commands.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add aether_agent/agent_commands.py tests/test_agent_commands.py
git commit -m "feat(cmds): agent_commands — name validation, template expansion, add/remove/list"
```

---

## Task 3: `agent_slash.py` — `cmd` + `run` verbs

**Files:**
- Modify: `aether_agent/agent_slash.py`
- Test: `tests/test_agent_slash_cmd.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_slash_cmd.py
import pytest

from aether_agent import agent_slash
from aether_agent.agent_profile import Agent


@pytest.fixture(autouse=True)
def _tmp_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))


def _mk(name="jane"):
    from aether_agent import agent_store
    agent_store.create(Agent.from_dict({"name": name}))


def test_cmd_add_list_remove():
    _mk()
    res = agent_slash.dispatch_agent("/agent jane cmd add review = Review $1 for bugs", active="")
    assert "review" in res["text"]
    assert Agent.load("jane").commands == {"review": "Review $1 for bugs"}
    res2 = agent_slash.dispatch_agent("/agent jane cmd list", active="")
    assert "review" in res2["text"]
    agent_slash.dispatch_agent("/agent jane cmd remove review", active="")
    assert Agent.load("jane").commands == {}


def test_cmd_add_requires_equals_and_rejects_reserved():
    _mk()
    res = agent_slash.dispatch_agent("/agent jane cmd add review Review $1", active="")  # no '='
    assert "=" in res["text"]
    res2 = agent_slash.dispatch_agent("/agent jane cmd add help = x", active="")  # reserved
    assert "command" in res2["text"].lower()
    assert Agent.load("jane").commands == {}  # nothing saved


def test_run_verb_expands_to_run_agent():
    _mk()
    agent_slash.dispatch_agent("/agent jane cmd add review = Review $1 for bugs", active="")
    res = agent_slash.dispatch_agent("/agent jane run review src/api.py", active="")
    assert res.get("run_agent") == {"name": "jane", "task": "Review src/api.py for bugs"}


def test_run_unknown_command_friendly():
    _mk()
    res = agent_slash.dispatch_agent("/agent jane run ghost x", active="")
    assert "no" in res["text"].lower() or "unknown" in res["text"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_slash_cmd.py -q`
Expected: FAIL — the `cmd`/`run` verbs aren't handled (a `cmd add ...` is currently treated as a one-shot task and returns a `run_agent` flag, not a text result), so the assertions fail.

- [ ] **Step 3: Write minimal implementation** (edit `aether_agent/agent_slash.py`)

Add `"cmd"` and `"run"` to `_VERBS`:

```python
_VERBS = {"set", "show", "edit", "delete", "cmd", "run"}
```

Add routing in `_agent()` (after the `delete` branch, before the one-shot-task fallthrough):

```python
    if verb == "cmd":
        return _cmd(name, rest[1:])
    if verb == "run":
        return _run_cmd(name, rest[1:])
```

Add the two handlers:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_slash_cmd.py tests/test_agent_slash.py -q`
Expected: PASS (new green; existing agent_slash tests still green)

- [ ] **Step 5: Commit**

```bash
git add aether_agent/agent_slash.py tests/test_agent_slash_cmd.py
git commit -m "feat(cmds): /agent <name> cmd add|list|remove + run <cmd> verbs"
```

---

## Task 4: `slash.py` — resolve bare `/<custom>` after built-ins

**Files:**
- Modify: `aether_agent/slash.py` (`dispatch`)
- Test: `tests/test_slash_custom_invoke.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_slash_custom_invoke.py
import pytest

from aether_agent.slash import SlashContext, dispatch
from aether_agent.agent_profile import Agent


@pytest.fixture(autouse=True)
def _tmp_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))


def _mk_with_cmd():
    from aether_agent import agent_store, agent_commands
    a = agent_commands.add(Agent.from_dict({"name": "jane"}), "review", "Review $1 for bugs")
    agent_store.create(a)


def test_builtin_wins_over_custom_same_name():
    res = dispatch(SlashContext(api=None, active_agent="jane"), "/help")
    assert "/exit" in res["text"]  # the built-in help text resolves, not a custom


def test_active_agent_custom_resolves():
    _mk_with_cmd()
    res = dispatch(SlashContext(api=None, active_agent="jane"), "/review src/api.py")
    assert res.get("run_agent") == {"name": "jane", "task": "Review src/api.py for bugs"}


def test_no_active_agent_unknown():
    _mk_with_cmd()
    res = dispatch(SlashContext(api=None, active_agent=""), "/review x")
    assert "unknown" in res["text"].lower()


def test_unknown_name_with_active_agent_is_unknown():
    _mk_with_cmd()
    res = dispatch(SlashContext(api=None, active_agent="jane"), "/ghost x")
    assert "unknown" in res["text"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_slash_custom_invoke.py -q`
Expected: FAIL — `test_active_agent_custom_resolves` fails (a custom `/review` currently returns the "unknown command" note).

- [ ] **Step 3: Write minimal implementation** (edit `aether_agent/slash.py`)

The unknown-command branch in `dispatch` currently is:

```python
    handler = REGISTRY.get(cmd)
    if handler is None:
        return _text(f"(unknown command: /{cmd}) — try /help")
    return handler(ctx, arg)
```

Change it to resolve a custom command BEFORE returning unknown:

```python
    handler = REGISTRY.get(cmd)
    if handler is not None:
        return handler(ctx, arg)
    custom = _resolve_custom_command(ctx, cmd, arg)
    if custom is not None:
        return custom
    return _text(f"(unknown command: /{cmd}) — try /help")
```

Add the resolver (near `dispatch`):

```python
def _resolve_custom_command(ctx: SlashContext, cmd: str, arg: str):
    """If an agent is active and has a command named ``cmd``, expand it into a
    run_agent action. Returns None when there is no match (caller -> unknown)."""
    if not getattr(ctx, "active_agent", ""):
        return None
    from aether_agent import agent_commands, agent_store

    try:
        agent = agent_store.load(ctx.active_agent)
    except (ValueError, OSError):
        return None
    cmds = agent_commands.list_commands(agent)
    if cmd not in cmds:
        return None
    task = agent_commands.expand(cmds[cmd], arg.split())
    return {"run_agent": {"name": ctx.active_agent, "task": task}}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_slash_custom_invoke.py tests/test_slash.py tests/test_slash_agents.py -q`
Expected: PASS (new green; existing slash tests still green — built-ins still resolve first)

- [ ] **Step 5: Commit**

```bash
git add aether_agent/slash.py tests/test_slash_custom_invoke.py
git commit -m "feat(cmds): resolve bare /<custom> against the active agent after built-ins"
```

---

## Task 5: `agent_runner.py` — the local-only `define_command` tool

**Files:**
- Modify: `aether_agent/agent_runner.py`
- Test: `tests/test_agent_runner_define_command.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_runner_define_command.py
import pytest

from aether_agent import agent_runner
from aether_agent.agent_profile import Agent


@pytest.fixture(autouse=True)
def _tmp_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))


def test_define_command_schema_only_when_allowed():
    with_tool = Agent.from_dict({"name": "jane", "tools": ["read_file", "define_command"]})
    without = Agent.from_dict({"name": "jane", "tools": ["read_file"]})
    assert "define_command" in agent_runner._offered_tool_names(with_tool)
    assert "define_command" not in agent_runner._offered_tool_names(without)


def test_define_command_writes_and_reloads():
    from aether_agent import agent_store
    a = Agent.from_dict({"name": "jane", "tools": ["read_file", "define_command"]})
    agent_store.create(a)
    pt = agent_runner._PolicyTools(
        inner=None, allowed=set(a.tools), permission="skip", confirm=lambda n, x: True, agent_name="jane",
    )
    out = pt.execute("define_command", {"name": "review", "template": "Review $1 for bugs"})
    assert "defined" in out.lower()
    assert agent_store.load("jane").commands == {"review": "Review $1 for bugs"}


def test_define_command_refused_when_not_allowed():
    pt = agent_runner._PolicyTools(
        inner=None, allowed={"read_file"}, permission="skip", confirm=lambda n, x: True, agent_name="jane",
    )
    out = pt.execute("define_command", {"name": "review", "template": "x"})
    assert "not allowed" in out.lower()


def test_define_command_rejects_reserved_name():
    from aether_agent import agent_store
    agent_store.create(Agent.from_dict({"name": "jane", "tools": ["define_command"]}))
    pt = agent_runner._PolicyTools(
        inner=None, allowed={"define_command"}, permission="skip", confirm=lambda n, x: True, agent_name="jane",
    )
    out = pt.execute("define_command", {"name": "help", "template": "x"})
    assert "refused" in out.lower()
    assert agent_store.load("jane").commands == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_runner_define_command.py -q`
Expected: FAIL — `AttributeError: module 'aether_agent.agent_runner' has no attribute '_offered_tool_names'` / `_PolicyTools.__init__() got an unexpected keyword argument 'agent_name'`.

- [ ] **Step 3: Write minimal implementation** (edit `aether_agent/agent_runner.py`)

Add a module-level helper:

```python
def _offered_tool_names(agent) -> list[str]:
    """The tool names offered to the model for this agent: the canonical tools it
    allows, plus the local-only define_command when the agent opts in."""
    from aether_agent.protocol import TOOLS
    names = [t for t in agent.tools if t in TOOLS]
    if "define_command" in agent.tools:
        names.append("define_command")
    return names
```

Change `_PolicyTools.__init__` to accept `agent_name`:

```python
    def __init__(self, inner, allowed: set, permission: str, confirm: ConfirmFn, agent_name: str = "") -> None:
        self._inner = inner
        self._allowed = set(allowed)
        self._permission = permission
        self._confirm = confirm
        self._agent_name = agent_name
        self.test_cmd = getattr(inner, "test_cmd", "")
```

In `_PolicyTools.execute`, handle `define_command` BEFORE the allowlist/inner delegation:

```python
    def execute(self, name: str, args: dict) -> str:
        if name == "define_command":
            return self._define_command(args)
        if name not in self._allowed:
            return f"[tool {name} not allowed for this agent]"
        if name in DESTRUCTIVE and self._permission != "skip":
            if not self._confirm(name, args):
                return f"[denied: {name} (permission={self._permission})]"
        return self._inner.execute(name, args)

    def _define_command(self, args: dict) -> str:
        if "define_command" not in self._allowed:
            return "[tool define_command not allowed for this agent]"
        from aether_agent import agent_commands, agent_store

        cname = str(args.get("name", ""))
        template = str(args.get("template", ""))
        try:
            updated = agent_commands.add(agent_store.load(self._agent_name), cname, template)
        except (ValueError, TypeError, OSError) as e:
            return f"[define_command refused: {e}]"
        agent_store.save(updated)
        return f"[defined command /{cname}]"
```

In `run()`, build the schema from the offered tools (so define_command is advertised when allowed) and
pass `agent_name`. Replace the `schema = [...]` and `tools = _PolicyTools(...)` lines with:

```python
    offered = set(_offered_tool_names(agent))
    schema = [s for s in tool_schema() if s["function"]["name"] in offered]
    if "define_command" in offered:
        schema.append({
            "type": "function",
            "function": {
                "name": "define_command",
                "description": "Save a reusable slash-command (prompt macro) on this agent. "
                               "args: {name, template}. template may use $1..$9 and $*.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "template": {"type": "string"}},
                    "required": ["name", "template"],
                },
            },
        })
    tools = _PolicyTools(Tools(cwd), allowed=set(agent.tools), permission=agent.permission,
                         confirm=confirm or _deny, agent_name=agent.name)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_runner_define_command.py tests/test_agent_runner.py -q`
Expected: PASS (new green; existing agent_runner tests still green — they construct `_PolicyTools(inner, allowed, permission, confirm)` positionally and `agent_name` defaults to "")

- [ ] **Step 5: Commit**

```bash
git add aether_agent/agent_runner.py tests/test_agent_runner_define_command.py
git commit -m "feat(cmds): local-only define_command tool (opt-in, macro-only, bounded self-modification)"
```

---

## Task 6: Full-suite + ruff + push

**Files:** full suite

- [ ] **Step 1: Ruff**

Run: `.venv/Scripts/python.exe -m ruff check aether_agent tests`
Expected: `All checks passed!` — remove any F401 and re-run.

- [ ] **Step 2: Full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS — all prior tests (506+) plus the ~20 new C tests. Zero regressions.

- [ ] **Step 3: Boot smoke (no Ollama)**

Run (bash): `AETHER_CONFIG_DIR="$(mktemp -d)" .venv/Scripts/python.exe -c "
from aether_agent import Agent
from aether_agent.slash import SlashContext, dispatch
Agent.create('jane')
print(dispatch(SlashContext(api=None), '/agent jane cmd add review = Review \$1 for bugs')['text'])
print('invoke ->', dispatch(SlashContext(api=None, active_agent='jane'), '/review x.py').get('run_agent'))
"`
Expected: `added /review = ...` / `invoke -> {'name': 'jane', 'task': 'Review x.py for bugs'}`.

- [ ] **Step 4: Commit any fixes + push**

```bash
git add -A && git commit -m "test(cmds): full-suite + ruff green after custom commands (sub-project C)"
git push -u origin feat/custom-commands
```

---

## Self-review (author checklist — completed)

- **Spec coverage:** prompt-macro storage + validation (T1) · name-validation + expansion + add/remove/list (T2) · manual `cmd` + `run` verbs (T3) · bare-slash resolution after built-ins (T4) · gated local-only `define_command` (T5) · offline tests per unit. All spec sections map to a task.
- **Placeholder scan:** no TBD/TODO; complete code each step.
- **Type consistency:** `agent_commands.{valid_name,expand,add,remove,list_commands}` consistent across T2–T5; `agent_profile.{is_valid_command_name,RESERVED_COMMANDS,ALLOWED_AGENT_TOOLS}` across T1,T2; `_PolicyTools(..., agent_name="")` + `_offered_tool_names` + `_define_command` across T5; the `run_agent` flag shape (`{"name","task"}`) matches B's repl handler. `define_command` is NOT added to `protocol.TOOLS` (bridge stays 8).
- **Scope:** C only — shell bindings, global commands, typed args, D all out.
```
