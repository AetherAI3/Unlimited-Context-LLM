# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""CLI dispatch tests (U4) — ``aether`` front-door routing.

Mirror of aether-code ``src/cli.ts`` / ``commands/chat.ts`` dispatch. Pure and
hermetic: every path that would touch a TTY, Ollama, or the network is
monkeypatched to a sentinel. We assert ROUTING (which handler a given argv runs),
never real I/O:

  - bare ``aether``                  -> repl.main()             (sentinel)
  - ``aether "<prompt>"``            -> one-shot via select_brain (sentinel)
  - ``aether code ...``              -> agent.run_agent          (unchanged)
  - ``aether auth status``           -> prints auth status
  - ``aether models`` / ``config``   -> their handlers

No subprocess: we call ``cli.main(argv)`` directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aether_agent import cli


# --- bare invocation -> REPL ----------------------------------------------
def test_bare_invocation_enters_repl(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel: dict[str, Any] = {"called": False}

    def fake_repl_main(*args: Any, **kw: Any) -> int:
        sentinel["called"] = True
        return 0

    # repl is imported lazily inside cli.main; patch the module attribute.
    import aether_agent.repl as repl_mod

    monkeypatch.setattr(repl_mod, "main", fake_repl_main)
    code = cli.main([])
    assert sentinel["called"] is True
    assert code == 0


# --- one-shot prompt -> a brain -------------------------------------------
def test_one_shot_prompt_routes_to_a_brain(monkeypatch: pytest.MonkeyPatch,
                                           capsys: pytest.CaptureFixture[str]) -> None:
    seen: dict[str, Any] = {}

    class FakeBrain:
        def run(self, task: str):
            seen["task"] = task
            yield {"type": "monologue", "text": "the answer is 42"}
            yield {"type": "done", "text": "the answer is 42"}

    def fake_select_brain(**kw: Any) -> FakeBrain:
        seen["select_kw"] = kw
        return FakeBrain()

    monkeypatch.setattr(cli, "select_brain", fake_select_brain)
    code = cli.main(["what is the answer"])
    out = capsys.readouterr().out
    assert seen.get("task") == "what is the answer"
    assert "42" in out
    assert code == 0


def test_one_shot_does_not_collide_with_subcommands(monkeypatch: pytest.MonkeyPatch) -> None:
    # 'code' must NOT be treated as a one-shot prompt — it is a subcommand.
    called: dict[str, Any] = {"run_agent": False, "brain": False}

    def fake_run_agent(*args: Any, **kw: Any):
        called["run_agent"] = True

        class R:
            ok = True
            summary = "done"
            steps = 1

        return R()

    def fake_select_brain(**kw: Any):
        called["brain"] = True
        raise AssertionError("'code' must route to run_agent, not a brain")

    monkeypatch.setattr(cli, "run_agent", fake_run_agent)
    monkeypatch.setattr(cli, "select_brain", fake_select_brain)
    cli.main(["code", "fix the tests"])
    assert called["run_agent"] is True
    assert called["brain"] is False


# --- existing `code` subcommand still routes to run_agent ------------------
def test_code_subcommand_routes_to_run_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run_agent(task: str, **kw: Any):
        captured["task"] = task
        captured["kw"] = kw

        class R:
            ok = True
            summary = "all green"
            steps = 3

        return R()

    monkeypatch.setattr(cli, "run_agent", fake_run_agent)
    code = cli.main(["code", "fix", "the", "failing", "tests"])
    assert captured["task"] == "fix the failing tests"
    assert code == 0


# --- `brain` subcommand still routes to headless --------------------------
def test_brain_subcommand_routes_to_headless(monkeypatch: pytest.MonkeyPatch) -> None:
    import aether_agent.headless as headless_mod

    called: dict[str, Any] = {"brain": False}

    def fake_brain_main(*a: Any, **k: Any) -> int:
        called["brain"] = True
        return 0

    monkeypatch.setattr(headless_mod, "main", fake_brain_main)
    code = cli.main(["brain"])
    assert called["brain"] is True
    assert code == 0


# --- auth status -----------------------------------------------------------
def test_auth_status_prints_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                   capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("AETHER_TOKEN", raising=False)
    code = cli.main(["auth", "status"])
    out = capsys.readouterr().out.lower()
    assert code == 0
    # logged out -> says so, and names the base url surface
    assert "logged" in out or "not" in out


def test_auth_login_with_token_stores_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("AETHER_TOKEN", raising=False)
    code = cli.main(["auth", "login", "--token", "aek_testtoken123"])
    assert code == 0
    from aether_agent.auth import FileTokenStore

    assert FileTokenStore().get() == "aek_testtoken123"


def test_auth_logout_clears_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("AETHER_TOKEN", raising=False)
    from aether_agent.auth import FileTokenStore

    FileTokenStore().set("sess-tok")
    code = cli.main(["auth", "logout"])
    assert code == 0
    assert FileTokenStore().get() is None


# --- config show/get/set ---------------------------------------------------
def test_config_set_then_get(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                             capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    code = cli.main(["config", "set", "defaultModel", "qwen3-coder:30b"])
    assert code == 0
    capsys.readouterr()  # drain
    code2 = cli.main(["config", "get", "defaultModel"])
    out = capsys.readouterr().out
    assert code2 == 0
    assert "qwen3-coder:30b" in out


def test_config_show_lists_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    code = cli.main(["config", "show"])
    out = capsys.readouterr().out
    assert code == 0
    assert "baseUrl" in out and "backend" in out


# --- models list -----------------------------------------------------------
def test_models_list_when_logged_out_does_not_crash(monkeypatch: pytest.MonkeyPatch,
                                                     tmp_path: Path,
                                                     capsys: pytest.CaptureFixture[str]) -> None:
    # Logged out: models must not hit the network; prints the local hint.
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("AETHER_TOKEN", raising=False)
    code = cli.main(["models"])
    out = capsys.readouterr().out.lower()
    assert code == 0
    assert "ollama" in out or "local" in out


# --- splash must survive a Windows cp1252 console (lockstep w/ protocol ASCII rule)
def test_splash_encodes_on_cp1252_console() -> None:
    """The startup banner must never crash a legacy Windows console. We render it
    and assert it round-trips through cp1252 (the default Win console codepage),
    catching box-drawing / middle-dot glyphs that would raise UnicodeEncodeError."""
    from aether_agent.splash import render_splash

    banner = render_splash("0.1.0", "qwen3-coder:30b", "local")
    # Must contain the required content line.
    assert "qwen3-coder:30b" in banner
    assert "/help for commands" in banner
    assert "local" in banner
    # Must encode on a cp1252 console without raising.
    banner.encode("cp1252")


def test_repl_main_non_tty_runs_help_then_exit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    """Non-TTY REPL: feed '/help' then '/exit' on stdin; it must print the banner
    + help and exit 0 without ever opening a socket or Ollama."""
    import io

    import aether_agent.repl as repl_mod

    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("AETHER_TOKEN", raising=False)
    # Force the non-TTY plain-line path with a scripted stdin.
    monkeypatch.setattr(repl_mod.sys, "stdin", io.StringIO("/help\n/exit\n"))
    code = repl_mod.main()
    out = capsys.readouterr().out
    assert code == 0
    assert "/help for commands" in out  # banner status line
    assert "/models" in out             # help body printed
