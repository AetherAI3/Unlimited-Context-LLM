# Design Spec — Custom Local Agents (sub-project B)

**Date:** 2026-06-15
**Repo:** `unlimited-context-llm` (Python) · branch `feat/custom-local-agents` (off `main`, which now
has sub-project A merged)

## Sub-project B of 4

Part of the "Claude Code for local" vision. A = Ollama-wrapped terminal (**merged to main**). This is
**B = custom local agents**: a user creates named, shareable agents, each with its own model, its own
Unlimited-Context memory pool, a persona, a visual identity, and tool/permission guardrails.

- **C** (custom local *command* authoring) and **D** (multi-agent `/agents` column dashboard +
  concurrent `/agent n1 [t] \ /agent n2 [t]`) are **separate later builds**. B keeps the `commands`
  field as a reserved empty map and ships a simple `/agents` list, so C/D layer on cleanly.

## Goal

Let a user run `/new-agent jane`, customize her (model, memory, persona, look, tools), and then
`/agent jane <task>` or switch to her — each agent a self-contained, shareable file with its own
persistent memory. Also expose an `Agent` library API so apps embed agents in two lines.

## Locked decisions (from brainstorm)

1. **One file per agent.** Definition at `~/.config/aether/agents/<name>.json` (human-editable,
   shareable by copying one small file). Memory pool is **separate local state** at
   `~/.config/aether/agents/<name>/pool/` (not shared by default — it's big).
2. **Customization surface:** persona+behavior, visual identity (accent / prompt symbol / banner),
   and tool+permission policy. **No full color theme** (single accent suffices).
3. **No effort tiers on local agents** — effort (`low/med/max/codepro`) is an **AetherCloud-only**
   concept; `max_steps` is the local depth knob.
4. **Unified `/agent <name> <verb>` grammar** (scales to C/D).
5. stdlib-only, no new runtime deps. cp1252-safe output. Reuses the engine (`Session` per-agent
   `pool_dir`/`pool_gb`/`separate`), `brains`, `Tools`, and A's `onboarding`/`ollama_ctl`.

## Architecture — new `aether_agent/` modules

| Module | Responsibility |
|---|---|
| `agent_profile.py` | `Agent` dataclass (all fields + defaults) + `from_dict`/`to_dict` + `validate()`. The library API surface. |
| `agent_store.py` | path helpers (`agents_dir`, `agent_path(name)`, `agent_pool_dir(name)`) + `create/load/save/list/delete/exists` (file-per-agent, fail-soft on corrupt). |
| `agent_runner.py` | build `Session(pool_dir=<agent pool>, pool_gb, pool_mode="separate")` + `Tools(allowlist)` + persona; run a task -> event stream (reusing `brains`/`agent.run_agent_events`); enforce `permission` + `max_steps`. |
| `agent_slash.py` | parse + handle `/new-agent`, `/agents`, `/agent <name> [verb ...]`. Pure handlers returning `SlashResult`. |

Wires into: `slash.py` (register the new commands + add `agents`/`active_agent` to `SlashContext`),
`repl.py` (active-agent state; apply persona/accent/prompt/banner on switch; status bar shows the
active agent). Re-export `Agent` from `aether_agent/__init__.py` for `from aether_agent import Agent`.

## Agent definition (`agent_profile.py`)

`~/.config/aether/agents/<name>.json`, every field default-filled on load (forward-compat). Synthetic
example values shown:

```json
{
  "name": "jane",
  "model": "qwen2.5-coder:7b",
  "pool_gb": 5,
  "persona": "You are Jane, a blunt coding assistant.",
  "verbosity": "normal",
  "show_tools": true,
  "accent": "cyan",
  "prompt": "jane > ",
  "banner": "~ Jane ~",
  "tools": ["read_file", "write_file", "run_shell", "run_tests", "repo_search", "git_commit", "web_search", "web_fetch"],
  "permission": "ask",
  "max_steps": 40,
  "commands": {}
}
```

`validate(agent) -> list[str]` (returns problems; empty = ok):
- `name` = a slug (`[a-z0-9][a-z0-9_-]*`), non-empty.
- `model` = non-empty string.
- `pool_gb` = int >= 1.
- `verbosity` in `{terse, normal, verbose}`; `permission` in `{ask, auto, skip}`.
- `accent` in a fixed palette (`{cyan, green, magenta, yellow, blue, red, white, dim}`).
- `tools` subset of the canonical 8 (`protocol.TOOLS`); unknown tool = error.
- `max_steps` = int in `[1, 500]`.
- `prompt`/`banner`/`persona` = strings; rendered cp1252-safe.
- `commands` = dict (reserved for C; must be empty in B — a non-empty `commands` is allowed to load
  but is ignored by B and flagged as "reserved for a future release").

No `effort` field. `Agent` is a frozen dataclass with `set(**fields)` returning a NEW validated copy
(immutability per house style).

## Per-agent memory pool (`agent_store.py` + `agent_runner.py`)

- `agent_pool_dir(name)` = `agents_dir()/<name>/pool/` (created lazily on first run).
- `agent_runner.run(agent, task, ...)` opens `Session(model=f"ollama/{agent.model}", pool_gb=agent.pool_gb,
  pool_dir=agent_pool_dir(agent.name), pool_mode="separate")` -> each agent has its **own persistent
  Unlimited-Context memory** across sessions (the engine reopens the pool dir).
- `/agent <name> set memory <gb>` updates `pool_gb` and saves; the engine re-indexes non-destructively
  on next open (it already supports resize/reopen).
- Deleting an agent removes its `<name>.json`; its `<name>/pool/` is kept unless `delete ... --purge`.

## Commands (`agent_slash.py`)

| Command | Action |
|---|---|
| `/new-agent <name>` | Create with defaults if absent (then customize via `set`); error if the name exists or is invalid. |
| `/agents` | List agents: `name  -  model  -  Npool GB  -  * active`. Simple list (D adds columns). |
| `/agent <name>` | Switch the active agent -> REPL applies its persona/model/accent/prompt/banner. |
| `/agent <name> <task>` | Run `<task>` once on `<name>` **without switching** (build a runner, render the event stream). The seam **D** parallelizes. |
| `/agent <name> set <key> <val>` | Configure (`model, memory->pool_gb, persona, accent, prompt, banner, verbosity, show_tools, permission, max_steps, tools`); validate, save, report. |
| `/agent <name> edit` | Open `<name>.json` in `$EDITOR` (best-effort; fallback: print the path). |
| `/agent <name> show` | Print the agent's resolved config. |
| `/agent <name> delete [--purge]` | Remove the agent file (and its pool with `--purge`); confirm-gated. |

Grammar parse: split the line after `/agent` into `name` + `rest`. If `rest` is empty -> switch. If
`rest[0]` in known verbs (`set/edit/show/delete`) -> that verb. Else -> treat `rest` as a one-shot task.
Unknown agent name -> helpful error (never raises). Handlers are pure (return `SlashResult`), so the
whole surface is unit-testable with no TTY/Ollama.

## Customizable UX application (`repl.py`)

On `/agent <name>` switch, the REPL stores the active agent and applies:
- **prompt** symbol (`agent.prompt`) replaces the default `aether > `.
- **accent** -> an ANSI color from the fixed palette (cp1252-safe; degrades to no-color on dumb terms).
- **banner** printed once on switch (cp1252-safe; skipped if empty).
- **persona** -> the system prompt for that agent's turns.
- **verbosity** + **show_tools** -> render filter (terse hides tool lines + trims monologue; verbose
  shows reasoning).
- **tools** / **permission** / **max_steps** -> passed into `agent_runner`.
With no active agent, the REPL behaves exactly as today (the A baseline). The status bar shows the
active agent name + model + pool.

## Library API (`from aether_agent import Agent`)

```python
a = Agent.create("jane", model="qwen2.5-coder:7b", pool_gb=10, persona="...")  # validates + saves
a = Agent.load("jane")
a = a.set(pool_gb=10, permission="auto"); a.save()
for ev in a.run("summarize the repo"): ...   # its own persistent memory; yields brain events
names = Agent.list()                          # -> ["jane", "neo"]
a.delete(purge=False)
```

`Agent.run` builds a runner under the hood; `create`/`load`/`save`/`list`/`delete` proxy `agent_store`.
Two-line embed for an app: `Agent.load("jane").run(task)`.

## Permission modes (`agent_runner.py` tool loop)

- `ask` — confirm before `write_file` / `run_shell` / `git_commit` (TTY prompt; non-TTY -> deny + note).
- `auto` — read-class tools run freely; destructive tools confirmed (TTY) or denied (non-TTY).
- `skip` — fully autonomous (no prompts).
The existing `Tools` path-jail (cwd) and `web` SSRF guards still apply underneath, regardless of mode.
The library default uses the agent's stored mode.

## Testing (`pytest -q`, offline — no Ollama/network)

- `tests/test_agent_profile.py` — defaults fill-in; `validate()` catches bad enum / non-subset tools /
  `pool_gb<1` / bad name / `max_steps` out of range; `from_dict`/`to_dict` roundtrip; `set()` returns a
  new validated copy; an `effort` key (if present in a file) is ignored.
- `tests/test_agent_store.py` — `create/load/save/list/delete/exists` under a tmp `AETHER_CONFIG_DIR`;
  path layout (`<name>.json`, `<name>/pool/`); corrupt JSON -> fail-soft (raises a clear error, not a
  crash); `delete --purge` removes the pool dir, default keeps it.
- `tests/test_agent_runner.py` — builds a `Session` with the right `pool_dir`/`pool_gb`/`separate`
  (inject a fake `Session` + fake brain); `tools` allowlist is enforced (a disallowed tool is refused);
  `permission=skip` runs without prompting; `max_steps` is passed through.
- `tests/test_agent_slash.py` — grammar matrix: `/agent jane` (switch), `/agent jane "task"` (run),
  `/agent jane set memory 10` (config+save), `/agent jane set tools ...`, `/new-agent jane`, `/agents`,
  unknown verb/name; dispatch with a fake store + fake runner (no Ollama).
- `tests/test_agent_library.py` — `Agent.create/load/list/set/save/delete` roundtrip; `run()` over a
  fake runner yields events.
- `tests/test_repl_agent_switch.py` — light: switching applies the agent's prompt + accent + persona
  (monkeypatched store/runner, no real turn).

## Out of scope (B)

- Custom command **authoring** UX (**C**) — the `commands` field exists but is empty/ignored.
- Multi-agent concurrency + `/agents` **column dashboard** (**D**) — B ships a simple list.
- Full per-role color theme (deferred; single accent only).
- Effort tiers (**AetherCloud-only**, never local).
- Sharing/installing an agent FROM a URL/registry (just copy the JSON file in v1).
