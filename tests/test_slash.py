# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Slash command tests (U4) — the in-REPL ``/command`` registry + dispatch.

Mirror of aether-code ``src/commands/slash.ts``. Pure: no network, no Ollama, no
real REPL. A ``FakeApiClient`` stands in for ``transport.ApiClient`` (records
which paths were fetched and returns scripted JSON); a stub ``web_search``
verifies ``/web`` runs the network tool inline. ``dispatch`` is a pure function
``dispatch(ctx, line) -> {exit?, restart?, text?}`` so the whole surface is
unit-testable without ever opening a socket or a TTY.
"""
from __future__ import annotations

from typing import Any

import pytest

from aether_agent import slash
from aether_agent.transport import MODELS_PATH


# --- stubs -----------------------------------------------------------------
class FakeApiClient:
    """Records GET paths and returns scripted JSON per path."""

    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        self._responses = responses or {}
        self.get_paths: list[str] = []

    def get_json(self, path: str) -> Any:
        self.get_paths.append(path)
        # match exact path or its prefix (audit trail carries a ?limit= query)
        if path in self._responses:
            return self._responses[path]
        for key, val in self._responses.items():
            if path.startswith(key):
                return val
        return {}


def _ctx(*, authed: bool = False, api: Any = None, model: str = "qwen3-coder:30b",
         web: Any = None) -> slash.SlashContext:
    """Build a SlashContext with sane defaults for tests."""
    return slash.SlashContext(
        api=api if api is not None else FakeApiClient(),
        authed=authed,
        model=model,
        web=web,
    )


# --- /help -----------------------------------------------------------------
def test_help_lists_the_commands() -> None:
    ctx = _ctx()
    res = slash.dispatch(ctx, "/help")
    assert res.get("exit") is not True
    text = res["text"]
    for cmd in ("/models", "/model", "/agents", "/tier", "/audit", "/clear", "/web", "/exit"):
        assert cmd in text


def test_bare_slash_is_treated_as_help() -> None:
    ctx = _ctx()
    res = slash.dispatch(ctx, "/")
    assert "/help" in res["text"] or "/models" in res["text"]


# --- /exit, /quit ----------------------------------------------------------
@pytest.mark.parametrize("line", ["/exit", "/quit", "/EXIT", "  /quit  "])
def test_exit_and_quit_set_exit_true(line: str) -> None:
    res = slash.dispatch(_ctx(), line)
    assert res.get("exit") is True


# --- /model <tag> ----------------------------------------------------------
def test_model_sets_ctx_model_and_returns_restart() -> None:
    ctx = _ctx(model="old-model")
    res = slash.dispatch(ctx, "/model qwen3-coder:30b")
    assert ctx.model == "qwen3-coder:30b"
    assert res.get("restart") == {"model": "qwen3-coder:30b"}
    assert "qwen3-coder:30b" in res["text"]


def test_model_with_no_arg_shows_usage_and_does_not_change_model() -> None:
    ctx = _ctx(model="keep-me")
    res = slash.dispatch(ctx, "/model")
    assert ctx.model == "keep-me"  # unchanged
    assert res.get("restart") is None
    assert "usage" in res["text"].lower()


# --- /models : local-vs-authed branch --------------------------------------
def test_models_when_authed_calls_api_models_path() -> None:
    api = FakeApiClient({MODELS_PATH: {"models": [{"id": "claude-x", "label": "Claude X"}], "tier": "pro"}})
    ctx = _ctx(authed=True, api=api)
    res = slash.dispatch(ctx, "/models")
    assert api.get_paths == [MODELS_PATH]
    assert "claude-x" in res["text"]


def test_models_when_local_shows_ollama_hint_and_does_not_call_api() -> None:
    api = FakeApiClient({MODELS_PATH: {"models": [{"id": "should-not-appear"}]}})
    ctx = _ctx(authed=False, api=api)
    res = slash.dispatch(ctx, "/models")
    assert api.get_paths == []  # never hit the network when local
    low = res["text"].lower()
    assert "ollama" in low
    assert "/model" in res["text"]


# --- /agents, /agent -------------------------------------------------------
def test_agents_when_authed_lists_orchestrators() -> None:
    api = FakeApiClient(
        {"/agents": {"agents": [{"id": "neo", "label": "Neo"}, {"id": "kronus", "label": "Kronus"}]}}
    )
    ctx = _ctx(authed=True, api=api)
    res = slash.dispatch(ctx, "/agents")
    assert any(p.startswith("/agents") for p in api.get_paths)
    assert "neo" in res["text"] and "kronus" in res["text"]


def test_agent_sets_id_and_returns_restart() -> None:
    ctx = _ctx(authed=True, model="m")
    res = slash.dispatch(ctx, "/agent kronus")
    assert res.get("restart") == {"agent": "kronus"}


# --- /tier -----------------------------------------------------------------
def test_tier_when_authed_reports_tier() -> None:
    api = FakeApiClient({MODELS_PATH: {"models": [], "tier": "pro", "default": "claude-x"}})
    ctx = _ctx(authed=True, api=api)
    res = slash.dispatch(ctx, "/tier")
    assert "pro" in res["text"]


def test_tier_when_local_says_local() -> None:
    res = slash.dispatch(_ctx(authed=False), "/tier")
    assert "local" in res["text"].lower()


# --- /audit ----------------------------------------------------------------
def test_audit_when_authed_fetches_trail() -> None:
    api = FakeApiClient(
        {"/audit/trail/live": {"entries": [{"event_type": "chat", "commitment_hash": "abc123"}], "count": 1}}
    )
    ctx = _ctx(authed=True, api=api)
    res = slash.dispatch(ctx, "/audit 5")
    assert any(p.startswith("/audit/trail/live") for p in api.get_paths)
    assert "abc123" in res["text"]


def test_audit_when_local_notes_unavailable() -> None:
    res = slash.dispatch(_ctx(authed=False), "/audit")
    assert res.get("text")  # a non-empty note, never a crash
    assert "sign" in res["text"].lower() or "log in" in res["text"].lower() or "local" in res["text"].lower()


# --- /clear ----------------------------------------------------------------
def test_clear_returns_screen_clear_sequence() -> None:
    res = slash.dispatch(_ctx(), "/clear")
    assert "\x1b[2J" in res["text"]  # ANSI clear-screen


# --- /web <query> ----------------------------------------------------------
def test_web_runs_web_search_inline() -> None:
    calls: list[tuple[str, int]] = []

    class WebStub:
        def web_search(self, query: str, limit: int = 5) -> str:
            calls.append((query, limit))
            return "1. Result Title\n   https://example.com"

    ctx = _ctx(web=WebStub())
    res = slash.dispatch(ctx, "/web how to write a python decorator")
    assert calls and calls[0][0] == "how to write a python decorator"
    assert "Result Title" in res["text"]


def test_web_with_no_query_shows_usage() -> None:
    res = slash.dispatch(_ctx(), "/web")
    assert "usage" in res["text"].lower()


# --- unknown ---------------------------------------------------------------
def test_unknown_command_is_helpful_not_a_crash() -> None:
    res = slash.dispatch(_ctx(), "/wat")
    assert res.get("exit") is not True
    low = res["text"].lower()
    assert "unknown" in low
    assert "/help" in res["text"]
