# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``aether-context setup`` — the guided first run.

Hermetic like the rest of the suite: the Ollama probe is monkeypatched (never a real socket),
pool state lives under ``tmp_path``, and every invocation is non-interactive so nothing can
block on ``input()``.

The behaviours worth pinning are the ones a broken first run would silently violate: the pool
gets written, the engine is really exercised, a missing daemon is a warning rather than a
failure, and the verification does not leave slices behind in the pool it just configured.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aether_context import cli


@pytest.fixture(autouse=True)
def _no_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default every test to 'no Ollama daemon' — no test in this file touches the network."""
    monkeypatch.setattr(cli, "_probe_ollama", lambda host: False)


def _setup(tmp_pool_dir: Path, *extra: str) -> int:
    """Run `setup` non-interactively against ``tmp_pool_dir``."""
    return cli.main(["setup", "--pool", "5", "--yes", "--dir", str(tmp_pool_dir), *extra])


# --- the happy path ----------------------------------------------------------
def test_setup_writes_the_pool_config_and_exits_zero(
    tmp_pool_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A clean setup configures the pool on disk and reports success."""
    # Act
    code = _setup(tmp_pool_dir)
    out = capsys.readouterr().out

    # Assert
    assert code == 0
    config = tmp_pool_dir / "config.json"
    assert config.exists()
    assert json.loads(config.read_text())["pool_gb"] == 5
    assert "pool configured" in out


def test_setup_runs_all_three_steps_and_verifies_the_engine(
    tmp_pool_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """All three numbered steps report, and the engine round trip actually passes."""
    # Act
    _setup(tmp_pool_dir)
    out = capsys.readouterr().out

    # Assert
    assert "[1/3]" in out and "[2/3]" in out and "[3/3]" in out
    assert "engine round trip passed" in out


def test_setup_ends_with_runnable_next_commands(
    tmp_pool_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The last thing a new user sees is commands they can paste, not a wall of prose."""
    # Act
    _setup(tmp_pool_dir)
    out = capsys.readouterr().out

    # Assert
    assert "aether-context chat" in out
    assert "aether-context status" in out
    assert "aether-context doctor" in out


# --- offline is a first-class path -------------------------------------------
def test_missing_daemon_warns_but_does_not_fail(
    tmp_pool_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No Ollama is a warning: the engine runs offline, so setup must not report failure."""
    # Act
    code = _setup(tmp_pool_dir)
    out = capsys.readouterr().out

    # Assert
    assert code == 0
    assert "[warn]" in out
    assert "no Ollama daemon" in out
    assert "mock model" in out  # the offline route is named, not just implied


def test_offline_setup_suggests_the_mock_model(
    tmp_pool_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no daemon, the suggested next command is one that works right now."""
    # Act
    _setup(tmp_pool_dir)
    out = capsys.readouterr().out

    # Assert
    assert "chat --model mock" in out


def test_reachable_daemon_with_pulled_model_reports_ok(
    tmp_pool_dir: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reachable daemon holding the model reports [ok] and points chat at that model."""
    # Arrange
    monkeypatch.setattr(cli, "_probe_ollama", lambda host: True)
    monkeypatch.setattr(cli, "_model_is_pulled", lambda host, model: True)

    # Act
    code = _setup(tmp_pool_dir, "--model", "qwen2.5")
    out = capsys.readouterr().out

    # Assert
    assert code == 0
    assert "'qwen2.5' is pulled" in out
    assert "chat --model ollama/qwen2.5" in out


def test_reachable_daemon_missing_model_prints_the_pull_command(
    tmp_pool_dir: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An un-pulled model is a warning carrying the exact `ollama pull` fix."""
    # Arrange
    monkeypatch.setattr(cli, "_probe_ollama", lambda host: True)
    monkeypatch.setattr(cli, "_model_is_pulled", lambda host, model: False)

    # Act
    code = _setup(tmp_pool_dir, "--model", "qwen2.5")
    out = capsys.readouterr().out

    # Assert
    assert code == 0
    assert "ollama pull qwen2.5" in out


# --- the verify must not pollute the pool it just sized ----------------------
def test_verification_leaves_the_configured_pool_empty(tmp_pool_dir: Path) -> None:
    """The engine check runs in a throwaway dir, so a fresh setup leaves 0 slices behind."""
    # Act
    _setup(tmp_pool_dir)

    # Assert
    from aether_context.config import PoolConfig

    slices, _capacity = cli._pool_counts(PoolConfig.load(tmp_pool_dir))
    assert slices == 0


# --- non-interactive discipline ----------------------------------------------
def test_setup_never_prompts_with_yes_even_on_a_tty(
    tmp_pool_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--yes` must skip the slider even when stdin looks interactive (scripts, Dockerfiles)."""
    # Arrange: a tty-looking stdin, and an input() that fails the test if it is ever reached.
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)

    def _explode(prompt: str = "") -> str:
        raise AssertionError("setup --yes must not prompt")

    monkeypatch.setattr("builtins.input", _explode)

    # Act
    code = cli.main(["setup", "--yes", "--dir", str(tmp_pool_dir)])

    # Assert
    assert code == 0


def test_setup_reports_failure_when_the_engine_check_breaks(
    tmp_pool_dir: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuinely broken engine exits non-zero and names the breakage instead of a traceback."""
    # Arrange
    def _broken(*args: object, **kwargs: object) -> object:
        raise RuntimeError("numpy is on fire")

    monkeypatch.setattr(cli, "Session", _broken)

    # Act
    code = _setup(tmp_pool_dir)
    out = capsys.readouterr().out

    # Assert
    assert code == 1
    assert "engine check failed" in out
    assert "numpy is on fire" in out
