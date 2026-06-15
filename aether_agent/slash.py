# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""In-REPL slash commands (Claude-Code style).

Mirror of aether-code ``src/commands/slash.ts``. The interactive ``aether``
session routes any line starting with ``/`` here. A pure registry maps each
command to a handler; ``dispatch(ctx, line)`` runs the handler and returns a
small dict the REPL acts on::

    {'exit'?: bool, 'restart'?: {'model'|'agent': str}, 'text'?: str}

Handlers NEVER touch a TTY directly — they return ``text`` for the REPL to print
— so the whole surface is unit-testable with no terminal, no socket, no Ollama.

Commands::

    /help                 list commands
    /models               chat models (cloud catalog if authed, else Ollama hint)
    /model <tag>          switch model (returns a restart so the REPL clears context)
    /agents               list orchestrators (Neo/Kronus) — authed only
    /agent <id>           switch orchestrator
    /tier                 plan tier + default (authed) or "local" (offline)
    /audit [n]            recent Aether audit trail (authed) or a note (offline)
    /clear                clear the screen
    /web <query>          run web_search inline (offline-friendly network tool)
    /exit | /quit         leave the REPL
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from aether_agent.transport import AGENTS_PATH, AUDIT_TRAIL_PATH, MODELS_PATH

#: A handler takes (ctx, arg) and returns the dispatch result dict.
SlashResult = dict[str, Any]
Handler = Callable[["SlashContext", str], SlashResult]


@dataclass
class SlashContext:
    """Mutable REPL state a slash handler may read or update.

    ``api``   — a transport.ApiClient (anything exposing ``get_json(path)``).
    ``authed``— True when a token is present (cloud) else local Ollama.
    ``model`` — the active model tag; ``/model`` mutates it in place.
    ``agent`` — the active orchestrator id (or ""); ``/agent`` mutates it.
    ``web``   — the network-tool surface for ``/web`` (defaults to the real
                ``aether_agent.web`` module; injectable in tests).
    """

    api: Any
    authed: bool = False
    model: str = ""
    agent: str = ""
    web: Any = None
    ollama: Any = None  # an ollama_ctl.OllamaCtl (local lifecycle); None when cloud-only
    active_agent: str = ""  # name of the active custom agent (B); "" = none


# --- help text -------------------------------------------------------------
_HELP_LINES = (
    "/models            list chat models",
    "/model <tag>       switch model (clears context)",
    "/agents            list your custom agents",
    "/agent <name>      switch agent (or: /agent <name> <task>)",
    "/new-agent <name>  create a custom agent",
    "/tier              plan tier + default",
    "/audit [n]         recent Aether audit trail",
    "/web <query>       search the web inline",
    "/pull <tag>        download/update a local model",
    "/doctor            check the local Ollama setup",
    "/serve             show the Ollama daemon state",
    "/setup             re-run first-run setup",
    "/clear             clear screen",
    "/exit              leave (also /quit)",
)


def _text(s: str) -> SlashResult:
    return {"text": s}


def _items(payload: Any, *keys: str) -> list[dict[str, Any]]:
    """Pull the first present list key out of an API payload (defensive)."""
    if not isinstance(payload, dict):
        return []
    for k in keys:
        v = payload.get(k)
        if isinstance(v, list):
            return [i for i in v if isinstance(i, dict)]
    return []


# --- handlers --------------------------------------------------------------
def _help(ctx: SlashContext, arg: str) -> SlashResult:
    return _text("\n".join(_HELP_LINES))


def _exit(ctx: SlashContext, arg: str) -> SlashResult:
    return {"exit": True}


def _models(ctx: SlashContext, arg: str) -> SlashResult:
    if not ctx.authed and ctx.ollama is not None:
        return _ollama_models(ctx)
    if not ctx.authed:
        return _text("(local Ollama — set a model with /model <tag>)")
    payload = ctx.api.get_json(MODELS_PATH)
    models = _items(payload, "models")
    tier = (payload or {}).get("tier", "") if isinstance(payload, dict) else ""
    lines: list[str] = []
    if tier:
        lines.append(f"tier: {tier}")
    for i, m in enumerate(models, 1):
        mid = str(m.get("id", ""))
        label = str(m.get("label", "") or "")
        mark = "›" if mid and mid == ctx.model else " "
        lines.append(f"{mark} {i:>2}. {mid}\t{label}".rstrip())
    lines.append("switch: /model <tag>")
    return _text("\n".join(lines))


def _model(ctx: SlashContext, arg: str) -> SlashResult:
    tag = arg.strip()
    if not tag:
        return _text("usage: /model <tag>")
    ctx.model = tag
    return {"restart": {"model": tag}, "text": f"model -> {tag} (context cleared)"}


def _agents_orch(ctx: SlashContext, arg: str) -> SlashResult:
    if not ctx.authed:
        return _text("(orchestrators are a cloud feature — log in with: aether auth login)")
    payload = ctx.api.get_json(AGENTS_PATH)
    agents = _items(payload, "agents", "orchestrators", "models")
    if not agents:
        return _text("(no orchestrators available)")
    lines = []
    for i, a in enumerate(agents, 1):
        aid = str(a.get("id", ""))
        label = str(a.get("label", "") or "")
        mark = "›" if aid and aid == ctx.agent else " "
        lines.append(f"{mark} {i:>2}. {aid}\t{label}".rstrip())
    lines.append("switch: /agent <id>")
    return _text("\n".join(lines))


def _agent_orch(ctx: SlashContext, arg: str) -> SlashResult:
    aid = arg.strip()
    if not aid:
        return _text("usage: /agent <id>")
    if not ctx.authed:
        return _text("(orchestrators are a cloud feature — log in with: aether auth login)")
    ctx.agent = aid
    return {"restart": {"agent": aid}, "text": f"orchestrator -> {aid} (context cleared)"}


def _tier(ctx: SlashContext, arg: str) -> SlashResult:
    if not ctx.authed:
        return _text("tier: local (offline Ollama — no plan; log in for cloud models)")
    payload = ctx.api.get_json(MODELS_PATH)
    if not isinstance(payload, dict):
        payload = {}
    tier = str(payload.get("tier", "") or "?")
    default = str(payload.get("default", "") or "?")
    n_models = len(_items(payload, "models"))
    return _text(f"tier: {tier}   default: {default}   models: {n_models}")


def _audit(ctx: SlashContext, arg: str) -> SlashResult:
    if not ctx.authed:
        return _text(
            "(audit trail is a cloud feature — turns are signed server-side; "
            "log in with: aether auth login)"
        )
    n = _to_int(arg.strip(), default=10)
    path = f"{AUDIT_TRAIL_PATH}?limit={n}"
    payload = ctx.api.get_json(path)
    entries = _items(payload, "entries")
    if not entries:
        return _text("(no audit entries)")
    lines = []
    for e in entries:
        ts = str(e.get("timestamp", "") or "")
        etype = str(e.get("event_type", "") or "")
        chash = str(e.get("commitment_hash", "") or "-")
        oid = str(e.get("order_id", "") or "")
        lines.append("\t".join(x for x in (ts, etype, chash, oid) if x is not None))
    return _text("\n".join(lines))


def _clear(ctx: SlashContext, arg: str) -> SlashResult:
    # ANSI: clear screen + home cursor.
    return _text("\x1b[2J\x1b[H")


def _web(ctx: SlashContext, arg: str) -> SlashResult:
    query = arg.strip()
    if not query:
        return _text("usage: /web <query>")
    web = ctx.web
    if web is None:
        from aether_agent import web as web  # lazy default — real network tool
    try:
        out = web.web_search(query)
    except Exception as e:  # noqa: BLE001 — a tool error must not crash the REPL
        out = f"[web_search error: {e}]"
    return _text(out)


def _new_agent(ctx: SlashContext, arg: str) -> SlashResult:
    from aether_agent.agent_slash import dispatch_agent
    return dispatch_agent(f"/new-agent {arg}".strip(), active=ctx.active_agent)


def _agents(ctx: SlashContext, arg: str) -> SlashResult:
    from aether_agent.agent_slash import dispatch_agent
    return dispatch_agent("/agents", active=ctx.active_agent)


def _agent_cmd(ctx: SlashContext, arg: str) -> SlashResult:
    from aether_agent.agent_slash import dispatch_agent
    return dispatch_agent(f"/agent {arg}".strip(), active=ctx.active_agent)


def _ollama_models(ctx: SlashContext) -> SlashResult:
    from aether_agent.hardware import CATALOG, best_fit, detect_resources

    ctl = ctx.ollama
    installed = {m["tag"] for m in ctl.list_models()} if ctl else set()
    rec, _ = best_fit(detect_resources())
    lines = ["local models (* installed, › current, ! recommended):"]
    for m in CATALOG:
        dot = "*" if m.tag in installed else " "
        cur = "›" if m.tag == ctx.model else " "
        star = "!" if m.tag == rec else " "
        note = f"  [{m.license_note}]" if m.license_note else ""
        lines.append(f"{dot}{cur}{star} {m.tag}\t{m.size_gb:g}GB\t{m.blurb}{note}")
    for tag in sorted(installed):
        if all(tag != m.tag for m in CATALOG):
            cur = "›" if tag == ctx.model else " "
            lines.append(f"*{cur}  {tag}\t(installed)")
    lines.append("switch: /model <tag>   ·   pull: /pull <tag>")
    return _text("\n".join(lines))


def _pull(ctx: SlashContext, arg: str) -> SlashResult:
    tag = arg.strip()
    if not tag:
        return _text("usage: /pull <tag>")
    if ctx.ollama is None:
        return _text("(pull is a local-Ollama command)")
    ok, detail = ctx.ollama.pull(tag)
    return _text(f"pull {tag}: {'ok' if ok else 'failed'} ({detail})")


def _doctor(ctx: SlashContext, arg: str) -> SlashResult:
    if ctx.ollama is None:
        return _text("(doctor checks the local Ollama; not available in cloud-only mode)")
    from aether_agent import onboarding
    return _text(onboarding.doctor(ctx.ollama))


def _serve(ctx: SlashContext, arg: str) -> SlashResult:
    if ctx.ollama is None:
        return _text("(serve status is a local-Ollama command)")
    det = ctx.ollama.detect()
    owned = bool(getattr(ctx.ollama, "_owned", False))
    state = "up" if det.get("daemon_up") else "down"
    return _text(f"ollama daemon: {state}{' (owned by this session)' if owned else ''}")


def _setup(ctx: SlashContext, arg: str) -> SlashResult:
    return {"setup": True, "text": "re-running setup…"}


def _to_int(s: str, *, default: int) -> int:
    try:
        v = int(s)
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


# --- registry + dispatch ---------------------------------------------------
REGISTRY: dict[str, Handler] = {
    "help": _help,
    "": _help,        # bare "/" -> help
    "models": _models,
    "model": _model,
    "agents": _agents,             # B: list custom local agents
    "agent": _agent_cmd,           # B: /agent <name> <verb>
    "new-agent": _new_agent,       # B: create
    "orchestrators": _agents_orch, # cloud orchestrators (was /agents)
    "orchestrator": _agent_orch,   # cloud orchestrator switch (was /agent)
    "tier": _tier,
    "audit": _audit,
    "clear": _clear,
    "web": _web,
    "pull": _pull,
    "doctor": _doctor,
    "serve": _serve,
    "setup": _setup,
    "exit": _exit,
    "quit": _exit,
}


def dispatch(ctx: SlashContext, line: str) -> SlashResult:
    """Route one ``/command`` line to its handler. Pure: returns a result dict,
    never touches the terminal. An unknown command returns a helpful note (never
    raises) so a typo can't break the REPL."""
    body = (line or "").strip()
    if body.startswith("/"):
        body = body[1:]
    parts = body.split()
    cmd = parts[0].lower() if parts else ""
    arg = " ".join(parts[1:])
    handler = REGISTRY.get(cmd)
    if handler is not None:
        return handler(ctx, arg)
    custom = _resolve_custom_command(ctx, cmd, arg)
    if custom is not None:
        return custom
    return _text(f"(unknown command: /{cmd}) — try /help")


def _resolve_custom_command(ctx: SlashContext, cmd: str, arg: str) -> SlashResult | None:
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


__all__ = ["SlashContext", "SlashResult", "REGISTRY", "dispatch"]
