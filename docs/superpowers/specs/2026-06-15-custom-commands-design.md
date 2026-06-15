# Design Spec — Custom Local Commands (sub-project C)

**Date:** 2026-06-15
**Repo:** `unlimited-context-llm` (Python) · build branch (later): `feat/custom-commands` (off `main`)

## Sub-project C of 4

Part of "Claude Code for local". A = Ollama-wrapped terminal (merged). B = custom local agents (merged;
reserved a per-agent `commands` dict, currently validated-empty). **C = custom command authoring**:
users -- and the agent itself, via a tool -- create reusable slash commands for an agent. **D**
(multi-agent `/agents` column dashboard + concurrent runs) is the separate next build.

## Goal

Let a user (or an agent) save a reusable **prompt-macro** on an agent -- a templated instruction that
expands into an agent turn -- and invoke it as a plain slash command:

```
/agent jane cmd add review = Review $1 for bugs and suggest fixes
# jane active:
/review src/api.py   ->   runs jane on "Review src/api.py for bugs and suggest fixes"
```

## Locked decisions (from brainstorm)

1. **Prompt-macro only.** A command = a string template that expands into an agent turn. No shell/script
   bindings (no new execution surface). The expanded turn is still gated by the agent's tool-allowlist +
   permission.
2. **Both authoring paths.** Manual (`/agent <name> cmd add|list|remove`) AND a local-only
   `define_command(name, template)` tool the agent may call mid-conversation (offered only when the
   agent's `tools` include it). The tool writes only a macro (bounded self-modification).
3. **Bare `/<name>` invocation, built-ins win.** Custom commands resolve as plain slashes only when an
   agent is active AND the name is not a built-in. Built-ins always take precedence; an unknown name
   with no matching custom command -> "unknown command". Cannot shadow a built-in.
4. stdlib-only, cp1252-safe, ruff-clean. Reuses B (`agent_profile`, `agent_store`, `agent_slash`,
   `agent_runner`, `slash`).

## Architecture

| File | Change |
|---|---|
| `aether_agent/agent_commands.py` | **New**: `valid_name(name)` (slug + not a reserved built-in), `expand(template, args)->str`, `add(agent, name, template)->Agent` / `remove(agent, name)->Agent` / `list_commands(agent)->dict`. |
| `aether_agent/agent_profile.py` | **Mod**: `commands` becomes a validated `dict[str, str]` (name->template); drop the "must be empty" rule; add `ALLOWED_AGENT_TOOLS = tuple(TOOLS) + ("define_command",)` so an agent may list `define_command` in `tools` without breaking the bridge's canonical 8. |
| `aether_agent/agent_slash.py` | **Mod**: add the `cmd` verb -> `/agent <name> cmd add|list|remove`; and `/agent <name> run <cmd> [args]` (run a command without switching). |
| `aether_agent/slash.py` `dispatch` | **Mod**: when `cmd` is not in `REGISTRY` and `ctx.active_agent` is set and the active agent has a command of that name -> return `{"run_agent": {"name": active, "task": expand(template, args)}}`. Built-ins resolve first; truly-unknown still returns the friendly note. |
| `aether_agent/agent_runner.py` | **Mod**: when `"define_command" in agent.tools`, inject its schema into the offered tools and handle the call in `_PolicyTools` (write the macro to the running agent's file via `agent_commands.add` + `agent_store.save`, then return a confirmation). `define_command` is **local-only** -- NOT added to `protocol.TOOLS` (the 8-tool bridge stays lockstep with aether-code). |

## Command storage + schema

`Agent.commands` = `{name: template}`:
- `name`: a slug (`[a-z0-9][a-z0-9_-]*`) and **not** a reserved built-in. Reserved set (rejected):
  `help, models, model, agents, agent, new-agent, orchestrator, orchestrators, tier, audit, web, clear,
  exit, quit, pull, doctor, serve, setup, config`.
- `template`: a non-empty string. Placeholders: `$1..$9` (positional args), `$*`/`$@` (all args joined
  by space), `$$` -> literal `$`. Any unfilled `$N` -> empty string.

`agent_profile.validate` now accepts a non-empty `commands` only if every key passes `valid_name` and
every value is a non-empty string (else a problem is reported). The `define_command` tool is allowed in
`tools` via `ALLOWED_AGENT_TOOLS` (the bridge schema in `headless.py` still advertises only the 8).

## Template expansion (`agent_commands.expand`)

`expand(template: str, args: list[str]) -> str`:
- `$$` -> `$` (done first, protected).
- `$1`..`$9` -> the i-th arg (1-based) or `""` if absent.
- `$*` and `$@` -> all args joined by a single space.
- Everything else is literal.
- Pure, deterministic, no shell/eval. Result is the user-turn text passed to the agent.

## Manual authoring (`agent_slash`, `cmd` verb)

- `/agent jane cmd add <name> = <template>` (everything after `=` is the template; `=` required).
- `/agent jane cmd list` -> the agent's commands (`name -> template`), or "(none)".
- `/agent jane cmd remove <name>`.
Each mutates the loaded `Agent` via `agent_commands` and saves through `agent_store`. Invalid name /
empty template / reserved name -> a friendly error, no crash, no save.

## Agent authoring (`define_command` tool)

- Offered to the model ONLY when `"define_command" in agent.tools` (default agents do NOT include it;
  the user opts in via `/agent jane set tools "<8 tools> define_command"`).
- Signature advertised to the model: `define_command(name: str, template: str)`.
- Handled in `agent_runner._PolicyTools.execute` (it knows the running agent): validates name+template,
  writes via `agent_commands.add` + `agent_store.save`, reloads so the command is usable immediately,
  returns `"[defined command /<name>]"` or `"[define_command refused: <reason>]"`.
- **Bounded self-modification:** it can ONLY add a prompt-macro to the agent's own `commands` -- it
  cannot change model/tools/permission/pool, and a macro grants no new execution capability (the macro's
  turn is still gated by the agent's existing allowlist + permission).

## Invocation (`slash.dispatch`)

Resolution order for a `/<word> args` line:
1. If `<word>` is a built-in in `REGISTRY` -> run it (built-ins always win).
2. Else if `ctx.active_agent` is set and the active agent's `commands` has `<word>` ->
   `{"run_agent": {"name": active, "task": expand(template, args)}}` (the REPL runs one turn).
3. Else -> "(unknown command: /<word>) - try /help".
`/agent <name> run <cmd> [args]` runs `<name>`'s command without switching (same expansion).

## Security

- Macros are templated **prompts** -- no shell, no eval, no file/network side effects at expansion time.
- The resulting turn runs through `agent_runner` -> still bounded by the agent's tool-allowlist +
  permission gate + the `Tools` cwd path-jail + `web` SSRF guard.
- `define_command` is opt-in (allowlist), macro-only, and cannot escalate the agent's capabilities.
- Reserved-name list prevents a custom command (user- or agent-authored) from shadowing a built-in.
- cp1252-safe rendering throughout.

## Testing (`pytest -q` + ruff, offline)

- `tests/test_agent_commands.py` -- `valid_name` (slug ok; reserved rejected; bad chars rejected);
  `expand` matrix (`$1..$9`, `$*`, `$@`, `$$`, missing args, no placeholders); `add`/`remove`/`list`
  return new Agents and round-trip.
- `tests/test_agent_profile_commands.py` -- `validate` accepts a good `{name: template}` map; rejects a
  reserved name, an empty template, a non-string value; `define_command` is accepted in `tools`.
- `tests/test_agent_slash_cmd.py` -- `/agent jane cmd add|list|remove`; reserved/invalid -> friendly
  error, no save; `/agent jane run review x` -> run_agent with expanded task.
- `tests/test_slash_custom_invoke.py` -- built-in wins over a custom of the same name; active agent's
  custom `/review x` -> run_agent expansion; no active agent -> unknown; unknown name -> unknown.
- `tests/test_agent_runner_define_command.py` -- `define_command` injected only when allowed; writes +
  reloads the agent; refused (string, no save) when not in `tools`; cannot alter other fields.

## Out of scope (C)

- Shell/script command bindings (macro-only).
- Global (non-agent) commands.
- Named/typed/optional args (positional `$1..$9` + `$*` only).
- Multi-agent concurrency + `/agents` columns (**D**).
- Sharing a command separately from its agent (commands travel inside the agent JSON).
