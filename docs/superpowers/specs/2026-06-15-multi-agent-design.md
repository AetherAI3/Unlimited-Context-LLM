# Design Spec — Multi-Agent (sub-project D)

**Date:** 2026-06-15
**Repo:** `unlimited-context-llm` (Python) · build branch (later): `feat/multi-agent` (off `main`)

## Sub-project D of 4 (final)

Part of "Claude Code for local". A = Ollama-wrapped terminal (merged). B = custom local agents (merged).
C = custom commands (spec+plan written). **D = multi-agent**: (1) `/agents` shows a **column dashboard**
of the user's agents, and (2) run **several agents at once** concurrently with live labeled output.

## Goal

```
/agents                                  -> a column dashboard of all agents + stats
/agent jane fix the tests \ /agent neo write the docs
   -> jane and neo run AT THE SAME TIME, output streamed live, labeled per agent
```

## Locked decisions (from brainstorm)

1. **Labeled interleaved stream** for concurrent output. Run N agents in daemon **threads** (`OllamaChat`
   is HTTP I/O-bound -> the GIL releases during the request -> real parallelism, the same pattern the
   engine's prefetch already uses). Each event is printed `[name] ...` colored by the agent's accent,
   streamed live; a per-agent summary at the end. **No full-screen pane TUI** (the line-based REPL has no
   raw-mode renderer; that lives in the TS host).
2. **Shared cwd, write-gated.** Concurrent agents share one cwd; reads/search/web run free in parallel,
   but `write_file`/`run_shell`/`git_commit` are serialized through one shared lock (one writer at a
   time) so edits can't interleave mid-write. Each agent keeps its **own** memory pool (isolated).
3. **`/agents` = static columns** (agent cards side by side). **Grammar** = ` \ `-separated
   `/agent <name> <task>` segments launch concurrently.
4. Cap concurrent agents at **8**. stdlib-only (`threading`, `queue`), cp1252-safe, ruff-clean. Reuses
   B/C (`agent_profile`, `agent_store`, `agent_runner`, `agent_slash`, `repl`, `ACCENTS`).

## Architecture

| File | Change |
|---|---|
| `aether_agent/agents_view.py` | **New**: pure `render_agents_columns(rows, active, width)` -- side-by-side agent cards (`name`/`model`/`Npool GB`/`Ncmds`/`* active`), wrapped N-per-row to terminal width. |
| `aether_agent/multi_runner.py` | **New**: `run_many(jobs, emit, confirm)` -- a daemon thread per `(agent, task)` running `agent_runner.run(..., write_lock=shared)`, pushing `(label, event)` to a `queue.Queue`; the caller's `emit(label, event)` is called on the main thread as events drain; returns per-agent summaries. Caps at 8. |
| `aether_agent/agent_runner.py` | **Mod**: `run(..., write_lock=None)`; `_PolicyTools` takes `write_lock` and acquires it around DESTRUCTIVE tools when set. |
| `aether_agent/agent_slash.py` | **Mod**: `/agents` returns `agents_view.render_agents_columns(...)` instead of the B simple list. |
| `aether_agent/repl.py` | **Mod**: split a REPL line on ` \ `; if >=2 segments parse to `/agent <name> <task>` runs, launch them via `multi_runner.run_many` with labeled rendering; otherwise the existing single path. |

## Concurrency model (`multi_runner.py`)

`run_many(jobs: list[tuple[Agent, str]], *, emit, confirm=None, cwd=".") -> list[dict]`:
- `jobs` capped to the first 8.
- one shared `write_lock = threading.Lock()` and a `queue.Queue()`.
- per job, a daemon thread runs:
  `for ev in agent_runner.run(agent, task, cwd=cwd, confirm=confirm or _deny, write_lock=write_lock): q.put((agent.name, ev))`,
  then `q.put((agent.name, {"type": "_thread_done"}))`.
- the main thread drains `q`: for each `(label, ev)` with a real type, call `emit(label, ev)`; track
  `_thread_done` to know when all threads finished; collect a small summary per agent (final `done`
  text + a tool-call count).
- returns `[{"name", "ok", "summary", "tool_calls"}]`.
- `emit` runs on the MAIN thread only (no concurrent terminal writes) -- the queue is the single
  serialization point, so rendering is race-free.

## Write-gating (`agent_runner._PolicyTools`)

- `run(agent, task, *, ..., write_lock=None)` threads `write_lock` into `_PolicyTools`.
- `_PolicyTools.execute`: for a DESTRUCTIVE tool (`write_file`/`run_shell`/`git_commit`), if a
  `write_lock` is set, wrap the inner `execute` in `with self._write_lock:` so only one concurrent agent
  writes at a time. Reads/search/web are NOT locked (parallel). When `write_lock` is None (single-agent
  path), behavior is unchanged.
- The existing per-agent permission gate + `Tools` cwd path-jail still apply underneath.

## Labeled rendering (`repl.py`)

`emit(label, ev)` -> `_render_labeled(label, ev, accent_ansi, out)`:
- prefix every output line with `[label] ` where `[label]` is colored by the agent's accent (reuse
  `agent_profile.ACCENTS`), reset after.
- render `monologue`/`tool_call`/`tool_result`/`done` the same way single runs do, but prefixed.
- cp1252-safe (`_safe_write`).
After `run_many` returns, print a `--- done ---` block: one line per agent (`name: <summary>`).

## `/agents` dashboard (`agents_view.py`)

`render_agents_columns(rows: list[dict], active: str = "", width: int = 80) -> str`:
- `rows` = `[{"name", "model", "pool_gb", "n_commands"}]` (built by `agent_slash._agents` from
  `agent_store.list_agents()` + `load`).
- each agent = a fixed-width card (e.g. 24 cols):
  ```
  * jane            <- accent, * = active
    qwen2.5-coder:7b
    5 GB . 2 cmds
  ```
- lay cards side by side, `floor(width / card_width)` per row, wrapping to more rows as needed.
- pure + deterministic + cp1252-safe (ASCII separators). No live refresh.

## Grammar (`repl.py`)

- Split the raw line on ` \ ` (space-backslash-space). If the result has >=2 parts AND each part (after
  trimming a leading `/`) starts with `agent <name> <task...>` (a run, not a verb like set/show) ->
  build `jobs` = `[(Agent.load(name), task)]` and call `run_many`.
- A single `/agent <name> <task>` (no ` \ `) stays the existing B single-run path.
- A bad segment (unknown agent / not a run) -> a friendly error for that segment; the valid ones still
  run. If <2 valid run-jobs, fall back to dispatching each segment normally (sequential).

## Security / safety

- Concurrency is bounded (<=8) so a stray line can't fork 50 Ollama calls.
- The shared write-lock prevents torn/interleaved file writes; per-agent permission + path-jail + web
  SSRF guard are unchanged and still enforced inside each thread.
- All terminal writes happen on the main thread (queue-serialized) -- no interleaved/garbled output.
- Each agent's memory pool is separate (`pool_mode="separate"`, distinct `pool_dir`) -- no cross-agent
  memory bleed.
- cp1252-safe throughout.

## Testing (`pytest -q` + ruff, offline)

- `tests/test_agents_view.py` -- `render_agents_columns`: single + multiple agents, active marker, width
  wrapping (cards per row), empty list, cp1252 encodability. Pure, deterministic.
- `tests/test_agent_runner_lock.py` -- with a `write_lock` injected (a recording lock), a DESTRUCTIVE
  tool acquires it and a read tool does not; without a lock, behavior is unchanged.
- `tests/test_multi_runner.py` -- `run_many` with a FAKE `agent_runner.run` (monkeypatched) yielding a
  scripted event list per agent: assert every agent's events reach `emit` labeled with its name, all
  agents complete, summaries returned; the >8 cap drops extras; a fake destructive tool through a shared
  recording lock proves one-writer-at-a-time.
- `tests/test_repl_multi.py` -- a ` \ `-joined line parses into 2 jobs and calls `run_many`
  (monkeypatched); a single `/agent` line does NOT; an unknown-agent segment degrades gracefully.
- `tests/test_agent_slash_agents_columns.py` -- `/agents` returns the column view containing each
  agent's name + model.

## Out of scope (D)

- Live full-screen column **panes** (raw-mode TUI) -- labeled-interleaved instead.
- Per-agent worktree isolation (shared cwd + write-lock instead).
- More than 8 concurrent agents.
- Cross-agent coordination/messaging (each runs its task independently).
- A live-refreshing dashboard (static snapshot only).
