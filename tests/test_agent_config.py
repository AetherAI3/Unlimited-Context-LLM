# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Agent config — distinct from aether_context/config.py (the engine pool config).

Mirrors aether-code src/core/config.ts: ~/.config/aether/config.json with an
AETHER_CONFIG_DIR override (tests) and an AETHER_BASE_URL override (one env var
re-points every API call). Corrupt JSON must NEVER crash — it falls back to the
defaults so a bad config can't brick the CLI. All tests write only under a tmp
AETHER_CONFIG_DIR.
"""
from __future__ import annotations

import json
from pathlib import Path

from aether_agent import config as agent_config


# --- defaults -------------------------------------------------------------
def test_defaults_when_no_file(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("AETHER_BASE_URL", raising=False)
    cfg = agent_config.load_config()
    assert cfg["baseUrl"] == "https://api.aethersystems.net/cloud"
    assert cfg["defaultModel"] == ""
    assert cfg["backend"] == "auto"
    assert cfg["permissionMode"] == "ask"
    assert cfg["autoApply"] is False
    assert cfg["telemetry"] is True


def test_config_dir_and_path_honor_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    assert agent_config.config_dir() == tmp_path
    assert agent_config.config_path() == tmp_path / "config.json"


def test_config_dir_default_is_under_home(monkeypatch):
    monkeypatch.delenv("AETHER_CONFIG_DIR", raising=False)
    d = agent_config.config_dir()
    assert d == Path.home() / ".config" / "aether"


# --- env override ---------------------------------------------------------
def test_base_url_env_overrides_default(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("AETHER_BASE_URL", "https://staging.example.net/cloud")
    cfg = agent_config.load_config()
    assert cfg["baseUrl"] == "https://staging.example.net/cloud"


def test_base_url_env_overrides_saved_file(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    # A persisted baseUrl is still overridden by the env var (env wins).
    (tmp_path / "config.json").write_text(
        json.dumps({"baseUrl": "https://saved.example.net/cloud"}), encoding="utf-8"
    )
    monkeypatch.setenv("AETHER_BASE_URL", "https://env.example.net/cloud")
    cfg = agent_config.load_config()
    assert cfg["baseUrl"] == "https://env.example.net/cloud"


# --- corrupt json fallback ------------------------------------------------
def test_corrupt_json_falls_back_to_defaults(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("AETHER_BASE_URL", raising=False)
    (tmp_path / "config.json").write_text("{ this is not json", encoding="utf-8")
    cfg = agent_config.load_config()  # must NOT raise
    assert cfg["baseUrl"] == "https://api.aethersystems.net/cloud"
    assert cfg["permissionMode"] == "ask"


def test_partial_json_merges_over_defaults(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("AETHER_BASE_URL", raising=False)
    (tmp_path / "config.json").write_text(
        json.dumps({"defaultModel": "sonnet", "autoApply": True}), encoding="utf-8"
    )
    cfg = agent_config.load_config()
    assert cfg["defaultModel"] == "sonnet"
    assert cfg["autoApply"] is True
    # untouched keys keep their defaults
    assert cfg["baseUrl"] == "https://api.aethersystems.net/cloud"
    assert cfg["telemetry"] is True


# --- save / load roundtrip ------------------------------------------------
def test_save_creates_dir_and_load_roundtrips(tmp_path: Path, monkeypatch):
    nested = tmp_path / "nested" / "aether"
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(nested))
    monkeypatch.delenv("AETHER_BASE_URL", raising=False)
    cfg = agent_config.load_config()
    cfg["defaultModel"] = "haiku"
    cfg["permissionMode"] = "auto"
    cfg["telemetry"] = False
    agent_config.save_config(cfg)
    assert (nested / "config.json").exists()

    again = agent_config.load_config()
    assert again["defaultModel"] == "haiku"
    assert again["permissionMode"] == "auto"
    assert again["telemetry"] is False


def test_save_writes_valid_json(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    cfg = agent_config.load_config()
    cfg["defaultModel"] = "opus"
    agent_config.save_config(cfg)
    raw = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert raw["defaultModel"] == "opus"
