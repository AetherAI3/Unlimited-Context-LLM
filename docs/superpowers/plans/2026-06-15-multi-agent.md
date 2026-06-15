# Multi-Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `/agents` column dashboard + the ability to run several agents at once (`/agent jane t \ /agent neo t`) concurrently, with live per-agent labeled output and a write-serialized shared workspace.

**Architecture:** Two new stdlib modules — `aether_agent/agents_view.py` (pure column renderer) and `aether_agent/multi_runner.py` (thread-per-agent orchestrator over a `queue.Queue`) — plus a `write_lock` param on `agent_runner.run`/`_PolicyTools`, a columns upgrade to `/agents`, and ` \ `-multi-run parsing + labeled rendering in `repl.py`.

**Tech Stack:** Python 3.10+, stdlib only (`threading`, `queue`, `dataclasses`), `pytest`. No new deps. Reuses B (`agent_profile`, `agent_store`, `agent_runner`, `agent_slash`, `repl`, `ACCENTS`).

**Spec:** `docs/superpowers/specs/2026-06-15-multi-agent-design.md`
**Branch:** `feat/multi-agent` (off `main`; build AFTER C if you want command counts in the dashboard, but D does not depend on C — the `write_lock` edit is orthogonal to C's `define_command` edit on `_PolicyTools`).
**Gate:** `.venv/Scripts/python.exe -m pytest -q` AND `.venv/Scripts/python.exe -m ruff check aether_agent tests`.

---

## File Structure

| File | New/Mod | Responsibility |
|---|---|---|
| `aether_agent/agents_view.py` | New | pure `render_agents_columns(rows, active, width)`. |
| `aether_agent/agent_runner.py` | Mod | `run(..., write_lock=None)`; `_PolicyTools` serializes DESTRUCTIVE tools through the lock when set. |
| `aether_agent/multi_runner.py` | New | `run_many(jobs, *, emit, confirm, cwd)` — threads + queue + summaries; caps at 8. |
| `aether_agent/agent_slash.py` | Mod | `/agents` -> the column view. |
| `aether_agent/repl.py` | Mod | parse ` \ ` multi-runs -> `run_many` + labeled rendering. |

---

## Task 1: `agents_view.py` — column dashboard renderer (pure)

**Files:**
- Create: `aether_agent/agents_view.py`
- Test: `tests/test_agents_view.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agents_view.py
from aether_agent import agents_view


def _rows(*names):
    return [{"name": n, "model": "qwen2.5-coder:7b", "pool_gb": 5, "n_commands": 2} for n in names]


def test_empty():
    assert "no agents" in agents_view.render_agents_columns([]).lower()


def test_single_card_has_fields_and_active_marker():
    out = agents_view.render_agents_columns(_rows("jane"), active="jane")
    assert "jane" in out and "qwen2.5-coder:7b" in out
    assert "5 GB" in out and "2 cmds" in out
    assert "*" in out  # active marker


def test_multiple_cards_wrap_to_width():
    out = agents_view.render_agents_columns(_rows("a", "b", "c", "d"), width=50)
    lines = out.splitlines()
    assert any("a" in line and "b" in line for line in lines)  # a,b side by side
    assert any("c" in line and "d" in line for line in lines)  # c,d on the next block


def test_cp1252_safe():
    agents_view.render_agents_columns(_rows("jane", "neo"), active="neo").encode("cp1252")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agents_view.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aether_agent.agents_view'`

- [ ] **Step 3: Write minimal implementation**

```python
# aether_agent/agents_view.py
# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""The `/agents` column dashboard — a pure, cp1252-safe renderer.

`render_agents_columns(rows, active, width)` lays out agent cards side by side,
N-per-row to the terminal width. No ANSI/color here (keeps width math correct and
the output testable); the active agent is marked with `*`.
"""
from __future__ import annotations

from typing import Any

_CARD_W = 22


def render_agents_columns(rows: list[dict], active: str = "", width: int = 80) -> str:
    if not rows:
        return "(no agents yet - create one with /new-agent <name>)"
    cards = [_card(r, active) for r in rows]
    per_row = max(1, width // (_CARD_W + 2))
    lines: list[str] = []
    for i in range(0, len(cards), per_row):
        group = cards[i:i + per_row]
        height = max(len(c) for c in group)
        for c in group:
            c.extend([""] * (height - len(c)))
        for row_line in range(height):
            lines.append("  ".join(c[row_line].ljust(_CARD_W) for c in group).rstrip())
        lines.append("")  # blank line between card rows
    return "\n".join(lines).rstrip()


def _card(r: dict[str, Any], active: str) -> list[str]:
    name = str(r.get("name", ""))
    mark = "*" if name and name == active else " "
    model = str(r.get("model", ""))
    pool = r.get("pool_gb", 0)
    n = r.get("n_commands", 0)
    return [
        f"{mark} {name}"[:_CARD_W],
        f"  {model}"[:_CARD_W],
        f"  {pool} GB . {n} cmds"[:_CARD_W],
    ]


__all__ = ["render_agents_columns"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agents_view.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add aether_agent/agents_view.py tests/test_agents_view.py
git commit -m "feat(multi): agents_view — pure column dashboard renderer"
```

---

## Task 2: `agent_runner.py` — `write_lock` serializes destructive tools

**Files:**
- Modify: `aether_agent/agent_runner.py`
- Test: `tests/test_agent_runner_lock.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_runner_lock.py
from aether_agent import agent_runner


class _RecordingLock:
    def __init__(self):
        self.acquired = 0
    def __enter__(self):
        self.acquired += 1
        return self
    def __exit__(self, *a):
        return False


class _FakeInner:
    test_cmd = ""
    def execute(self, name, args):
        return f"[ran {name}]"


def test_destructive_acquires_lock_read_does_not():
    lock = _RecordingLock()
    pt = agent_runner._PolicyTools(
        _FakeInner(), allowed={"write_file", "read_file"}, permission="skip",
        confirm=lambda n, a: True, write_lock=lock,
    )
    pt.execute("read_file", {"path": "x"})
    assert lock.acquired == 0          # reads are not locked
    pt.execute("write_file", {"path": "x", "content": "y"})
    assert lock.acquired == 1          # destructive acquired the lock


def test_no_lock_means_no_locking():
    pt = agent_runner._PolicyTools(
        _FakeInner(), allowed={"write_file"}, permission="skip", confirm=lambda n, a: True,
    )
    assert "[ran write_file]" in pt.execute("write_file", {"path": "x", "content": "y"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_runner_lock.py -q`
Expected: FAIL — `_PolicyTools.__init__() got an unexpected keyword argument 'write_lock'`

- [ ] **Step 3: Write minimal implementation** (edit `aether_agent/agent_runner.py`)

Add `write_lock=None` to `_PolicyTools.__init__` and store it (keep any existing params like `agent_name`
from sub-project C if present — this is an additive param):

```python
    def __init__(self, inner, allowed: set, permission: str, confirm: ConfirmFn,
                 agent_name: str = "", write_lock=None) -> None:
        self._inner = inner
        self._allowed = set(allowed)
        self._permission = permission
        self._confirm = confirm
        self._agent_name = agent_name
        self._write_lock = write_lock
        self.test_cmd = getattr(inner, "test_cmd", "")
```

(If sub-project C is not yet built, `agent_name` is also new — include it; if C is built, just add
`write_lock` to the existing signature.)

In `execute`, after the permission gate passes and BEFORE delegating to the inner tool, wrap a
DESTRUCTIVE call in the lock when one is set:

```python
    def execute(self, name: str, args: dict) -> str:
        if name not in self._allowed:
            return f"[tool {name} not allowed for this agent]"
        if name in DESTRUCTIVE and self._permission != "skip":
            if not self._confirm(name, args):
                return f"[denied: {name} (permission={self._permission})]"
        if name in DESTRUCTIVE and self._write_lock is not None:
            with self._write_lock:
                return self._inner.execute(name, args)
        return self._inner.execute(name, args)
```

(If C's `define_command` short-circuit exists, keep it as the FIRST line of `execute`.)

In `run()`, add a `write_lock=None` param and pass it into `_PolicyTools`:

```python
def run(agent, task, *, cwd=".", confirm=None, llm=None, session_factory=None,
        on_status=None, write_lock=None):
    ...
    tools = _PolicyTools(Tools(cwd), allowed=set(agent.tools), permission=agent.permission,
                         confirm=confirm or _deny, agent_name=agent.name, write_lock=write_lock)
```

(Drop `agent_name=agent.name` if C is not built. The `write_lock=write_lock` addition is the D change.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_runner_lock.py tests/test_agent_runner.py -q`
Expected: PASS (new green; existing agent_runner tests still green — `write_lock` defaults to None)

- [ ] **Step 5: Commit**

```bash
git add aether_agent/agent_runner.py tests/test_agent_runner_lock.py
git commit -m "feat(multi): agent_runner write_lock serializes destructive tools (one writer)"
```

---

## Task 3: `multi_runner.py` — concurrent orchestrator

**Files:**
- Create: `aether_agent/multi_runner.py`
- Test: `tests/test_multi_runner.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multi_runner.py
from aether_agent import multi_runner
from aether_agent.agent_profile import Agent


def test_run_many_streams_labeled_events_and_summaries(monkeypatch):
    scripts = {
        "jane": [{"type": "monologue", "text": "j1"}, {"type": "tool_call", "name": "read_file", "args": {}},
                 {"type": "done", "text": "jane done", "ok": True}],
        "neo": [{"type": "monologue", "text": "n1"}, {"type": "done", "text": "neo done", "ok": True}],
    }

    def fake_run(agent, task, **kw):
        for ev in scripts[agent.name]:
            yield ev

    monkeypatch.setattr("aether_agent.agent_runner.run", fake_run)
    seen = []
    jobs = [(Agent.from_dict({"name": "jane"}), "t1"), (Agent.from_dict({"name": "neo"}), "t2")]
    summaries = multi_runner.run_many(jobs, emit=lambda label, ev: seen.append((label, ev.get("type"))))
    labels = {label for label, _ in seen}
    assert labels == {"jane", "neo"}
    assert ("jane", "monologue") in seen and ("neo", "done") in seen
    by_name = {s["name"]: s for s in summaries}
    assert by_name["jane"]["tool_calls"] == 1 and by_name["jane"]["ok"] is True
    assert by_name["neo"]["summary"] == "neo done"


def test_cap_at_8(monkeypatch):
    def fake_run(agent, task, **kw):
        yield {"type": "done", "text": "x", "ok": True}
    monkeypatch.setattr("aether_agent.agent_runner.run", fake_run)
    jobs = [(Agent.from_dict({"name": f"a{i}"}), "t") for i in range(12)]
    summaries = multi_runner.run_many(jobs, emit=lambda l, e: None)
    assert len(summaries) == 8  # capped


def test_shared_write_lock_passed_to_each_runner(monkeypatch):
    locks_seen = []

    def fake_run(agent, task, *, write_lock=None, **kw):
        locks_seen.append(write_lock)
        yield {"type": "done", "text": "x", "ok": True}

    monkeypatch.setattr("aether_agent.agent_runner.run", fake_run)
    jobs = [(Agent.from_dict({"name": "a"}), "t"), (Agent.from_dict({"name": "b"}), "t")]
    multi_runner.run_many(jobs, emit=lambda l, e: None)
    assert locks_seen[0] is locks_seen[1] is not None  # same shared lock for all jobs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_multi_runner.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aether_agent.multi_runner'`

- [ ] **Step 3: Write minimal implementation**

```python
# aether_agent/multi_runner.py
# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Run several agents at once (sub-project D).

A daemon thread per ``(agent, task)`` runs ``agent_runner.run`` (Ollama is HTTP
I/O-bound, so the GIL releases during the request -> real parallelism). Every
event is pushed to a single ``queue.Queue`` that the MAIN thread drains and hands
to ``emit(label, event)`` -- so all terminal writes happen on one thread (no
garbled output). A shared ``write_lock`` (passed into each runner) serializes
destructive tools across agents. Bounded to 8 concurrent.
"""
from __future__ import annotations

import queue
import threading
from typing import Callable

from aether_agent import agent_runner

MAX_CONCURRENT = 8
Emit = Callable[[str, dict], None]
ConfirmFn = Callable[[str, dict], bool]


def _deny(_name: str, _args: dict) -> bool:
    return False


def run_many(jobs, *, emit: Emit, confirm: "ConfirmFn | None" = None, cwd: str = ".") -> list[dict]:
    """Run ``jobs`` (list of ``(Agent, task)``) concurrently; ``emit(label, ev)``
    is called on the main thread per event. Returns a summary dict per agent."""
    jobs = list(jobs)[:MAX_CONCURRENT]
    write_lock = threading.Lock()
    q: "queue.Queue[tuple[str, dict]]" = queue.Queue()
    summaries: dict[str, dict] = {
        a.name: {"name": a.name, "ok": False, "summary": "", "tool_calls": 0} for a, _ in jobs
    }

    def worker(agent, task) -> None:
        try:
            for ev in agent_runner.run(
                agent, task, cwd=cwd, confirm=confirm or _deny, write_lock=write_lock
            ):
                q.put((agent.name, ev))
        except Exception as e:  # noqa: BLE001 — surface as an error event, never crash the thread
            q.put((agent.name, {"type": "error", "msg": str(e)}))
        finally:
            q.put((agent.name, {"type": "_thread_done"}))

    threads = [threading.Thread(target=worker, args=(a, t), daemon=True) for a, t in jobs]
    for th in threads:
        th.start()

    remaining = len(jobs)
    while remaining > 0:
        label, ev = q.get()
        etype = ev.get("type")
        if etype == "_thread_done":
            remaining -= 1
            continue
        if etype == "tool_call":
            summaries[label]["tool_calls"] += 1
        elif etype == "done":
            summaries[label]["ok"] = bool(ev.get("ok", True))
            summaries[label]["summary"] = str(ev.get("text", ""))
        emit(label, ev)

    return [summaries[a.name] for a, _ in jobs]


__all__ = ["run_many", "MAX_CONCURRENT"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_multi_runner.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add aether_agent/multi_runner.py tests/test_multi_runner.py
git commit -m "feat(multi): multi_runner — thread-per-agent concurrent runs over a queue (labeled, write-locked, capped 8)"
```

---

## Task 4: `agent_slash.py` — `/agents` column dashboard

**Files:**
- Modify: `aether_agent/agent_slash.py` (`_agents`)
- Test: `tests/test_agent_slash_agents_columns.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_slash_agents_columns.py
import pytest

from aether_agent import agent_slash
from aether_agent.agent_profile import Agent


@pytest.fixture(autouse=True)
def _tmp_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))


def test_agents_renders_columns():
    from aether_agent import agent_store
    agent_store.create(Agent.from_dict({"name": "jane", "model": "qwen2.5-coder:7b", "pool_gb": 5}))
    agent_store.create(Agent.from_dict({"name": "neo", "model": "qwen3-coder:30b", "pool_gb": 10}))
    out = agent_slash.dispatch_agent("/agents", active="neo")["text"]
    assert "jane" in out and "neo" in out
    assert "qwen2.5-coder:7b" in out and "qwen3-coder:30b" in out
    assert "GB" in out  # the card stats line


def test_agents_empty():
    out = agent_slash.dispatch_agent("/agents", active="")["text"]
    assert "no agents" in out.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_slash_agents_columns.py -q`
Expected: FAIL — the current `_agents` returns the B simple-list text (a single line per agent with the
`switch:` footer), so the multi-line card-layout assertions do not all hold.

- [ ] **Step 3: Write minimal implementation** (edit `aether_agent/agent_slash.py`)

Replace the body of `_agents(active)` with the column build:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_slash_agents_columns.py tests/test_agent_slash.py tests/test_slash_agents.py -q`
Expected: PASS — update any existing test that asserted the old B simple-list `/agents` text (e.g. the
"switch:" footer) to assert only on agent names/models that appear in the column output too.

- [ ] **Step 5: Commit**

```bash
git add aether_agent/agent_slash.py tests/test_agent_slash_agents_columns.py
git commit -m "feat(multi): /agents renders the column dashboard via agents_view"
```

---

## Task 5: `repl.py` — parse ` \ ` multi-runs + labeled rendering

**Files:**
- Modify: `aether_agent/repl.py`
- Test: `tests/test_repl_multi.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_repl_multi.py
import io

import pytest

from aether_agent import repl
from aether_agent.agent_profile import Agent


@pytest.fixture(autouse=True)
def _tmp_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))


def test_parse_multi_two_runs():
    from aether_agent import agent_store
    agent_store.create(Agent.from_dict({"name": "jane"}))
    agent_store.create(Agent.from_dict({"name": "neo"}))
    jobs = repl._parse_multi("/agent jane fix tests \\ /agent neo write docs")
    assert jobs is not None
    assert [(a.name, t) for a, t in jobs] == [("jane", "fix tests"), ("neo", "write docs")]


def test_parse_multi_single_returns_none():
    from aether_agent import agent_store
    agent_store.create(Agent.from_dict({"name": "jane"}))
    assert repl._parse_multi("/agent jane fix tests") is None  # no ' \\ ' -> not multi


def test_handle_multi_streams_labeled(monkeypatch):
    from aether_agent import agent_store
    agent_store.create(Agent.from_dict({"name": "jane"}))
    agent_store.create(Agent.from_dict({"name": "neo"}))

    def fake_run_many(jobs, *, emit, **kw):
        for a, _ in jobs:
            emit(a.name, {"type": "done", "text": f"{a.name} ok"})
        return [{"name": a.name, "ok": True, "summary": f"{a.name} ok", "tool_calls": 0} for a, _ in jobs]

    monkeypatch.setattr("aether_agent.multi_runner.run_many", fake_run_many)
    out = io.StringIO()
    jobs = repl._parse_multi("/agent jane t \\ /agent neo t")
    repl._handle_multi(jobs, out)
    text = out.getvalue()
    assert "[jane]" in text and "[neo]" in text
    assert "done" in text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_repl_multi.py -q`
Expected: FAIL — `AttributeError: module 'aether_agent.repl' has no attribute '_parse_multi'`

- [ ] **Step 3: Write minimal implementation** (edit `aether_agent/repl.py`)

Add the parser + handlers (module level):

```python
def _parse_multi(line: str):
    """Parse a ` \\ `-joined multi-agent run line into [(Agent, task), ...].
    Returns None when it is not a 2+ agent-run line."""
    if " \\ " not in line:
        return None
    from aether_agent import agent_store

    jobs = []
    for seg in line.split(" \\ "):
        s = seg.strip()
        if s.startswith("/"):
            s = s[1:]
        parts = s.split()
        if len(parts) < 3 or parts[0].lower() != "agent":
            return None  # not all segments are agent runs -> not a multi-run
        if parts[2].lower() in {"set", "show", "edit", "delete", "cmd", "run"}:
            return None  # a verb, not a task
        name = parts[1]
        task = " ".join(parts[2:])
        try:
            jobs.append((agent_store.load(name), task))
        except (ValueError, OSError):
            return None
    return jobs if len(jobs) >= 2 else None


def _handle_multi(jobs, out) -> None:
    """Run jobs concurrently, rendering labeled interleaved output + a summary."""
    from aether_agent import multi_runner
    from aether_agent.agent_profile import ACCENTS

    accents = {a.name: f"\x1b[{ACCENTS.get(a.accent, '36')}m" for a, _ in jobs}

    def emit(label: str, ev: dict) -> None:
        _render_labeled(label, ev, accents.get(label, ""), out)

    summaries = multi_runner.run_many(jobs, emit=emit, confirm=lambda n, a: False)
    _safe_write(out, "\n--- done ---\n")
    for s in summaries:
        _safe_write(out, f"{s['name']}: {s['summary'] or ('ok' if s['ok'] else 'stopped')}\n")


def _render_labeled(label: str, ev: dict, accent_ansi: str, out) -> None:
    """Render one event prefixed by [label] (accent-colored), cp1252-safe."""
    tag = f"{accent_ansi}[{label}]\x1b[0m " if accent_ansi else f"[{label}] "
    etype = ev.get("type")
    if etype in ("monologue", "done"):
        text = str(ev.get("text", ""))
        if text:
            _safe_write(out, tag + _first_line(text, 120) + "\n")
    elif etype == "tool_call":
        _safe_write(out, tag + f"- {ev.get('name', '')}({_fmt_args(ev.get('args', {}))})\n")
    elif etype == "tool_result":
        _safe_write(out, tag + f"- {ev.get('name', '')} -> {_first_line(str(ev.get('output', '')))}\n")
    elif etype == "error":
        _safe_write(out, tag + f"[x] {ev.get('msg', 'error')}\n")
```

In `main()`, BEFORE the `if line.startswith("/"):` slash handling, add the multi-run branch:

```python
        jobs = _parse_multi(line)
        if jobs is not None:
            _handle_multi(jobs, out)
            out.write("\n")
            continue
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_repl_multi.py tests/test_repl_agents.py tests/test_repl_preflight.py -q`
Expected: PASS (new green; existing repl tests still green)

- [ ] **Step 5: Commit**

```bash
git add aether_agent/repl.py tests/test_repl_multi.py
git commit -m "feat(multi): REPL parses ' \\ ' multi-agent runs -> concurrent labeled streaming"
```

---

## Task 6: Full-suite + ruff + push

**Files:** full suite

- [ ] **Step 1: Ruff**

Run: `.venv/Scripts/python.exe -m ruff check aether_agent tests`
Expected: `All checks passed!` — remove any F401 and re-run.

- [ ] **Step 2: Full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS — all prior tests plus the ~14 new D tests. Zero regressions.

- [ ] **Step 3: Boot smoke (no Ollama)**

Run (bash): `AETHER_CONFIG_DIR="$(mktemp -d)" .venv/Scripts/python.exe -c "
from aether_agent import Agent
from aether_agent.slash import SlashContext, dispatch
Agent.create('jane'); Agent.create('neo')
print(dispatch(SlashContext(api=None, active_agent='neo'), '/agents')['text'])
"`
Expected: a column dashboard listing `jane` and `neo` with `*` on `neo`.

- [ ] **Step 4: Commit any fixes + push**

```bash
git add -A && git commit -m "test(multi): full-suite + ruff green after multi-agent (sub-project D)"
git push -u origin feat/multi-agent
```

---

## Self-review (author checklist — completed)

- **Spec coverage:** column dashboard renderer (T1) · write-lock serialization (T2) · concurrent thread orchestrator + cap + summaries (T3) · `/agents` columns (T4) · ` \ ` parse + labeled interleaved streaming (T5) · offline tests per unit. All spec sections map to a task.
- **Placeholder scan:** no TBD/TODO; complete code each step.
- **Type consistency:** `multi_runner.run_many(jobs, *, emit, confirm, cwd)` + the `(label, event)` queue shape consistent across T3,T5; `_PolicyTools(..., write_lock=None)` + `run(..., write_lock=None)` across T2,T3; `agents_view.render_agents_columns(rows, active, width)` + the `rows` dict shape (`name/model/pool_gb/n_commands`) across T1,T4; `repl._parse_multi`/`_handle_multi`/`_render_labeled` across T5. `write_lock` is additive + orthogonal to C's `define_command` edit on `_PolicyTools`.
- **Scope:** D only — full-screen panes, worktree isolation, >8 concurrent, cross-agent messaging all out.
```
