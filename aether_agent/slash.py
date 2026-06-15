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


# --- help text -------------------------------------------------------------
_HELP_LINES = (
    "/models            list chat models",
    "/model <tag>       switch model (clears context)",
    "/agents            list orchestrators (Neo/Kronus)",
    "/agent <id>        switch orchestrator",
    "/tier              plan tier + default",
    "/audit [n]         recent Aether audit trail",
    "/web <query>       search the web inline",
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


def _agents(ctx: SlashContext, arg: str) -> SlashResult:
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


def _agent(ctx: SlashContext, arg: str) -> SlashResult:
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
    "agents": _agents,
    "agent": _agent,
    "tier": _tier,
    "audit": _audit,
    "clear": _clear,
    "web": _web,
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
    if handler is None:
        return _text(f"(unknown command: /{cmd}) — try /help")
    return handler(ctx, arg)


__all__ = ["SlashContext", "SlashResult", "REGISTRY", "dispatch"]
