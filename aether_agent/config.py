# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Agent config — load/save ~/.config/aether/config.json.

This is the AGENT (CLI host) config and is deliberately distinct from
``aether_context.config`` (the engine's on-disk *pool* config). Mirror of
aether-code ``src/core/config.ts``: a single env var (AETHER_BASE_URL) re-points
every API call at a staging/self-hosted backend, and AETHER_CONFIG_DIR relocates
the whole config directory (used by tests). A corrupt config.json must NEVER
crash the CLI — it falls back to the defaults.

On-disk contract (lockstep with the TS mirror):
  - dir:  ~/.config/aether            (env AETHER_CONFIG_DIR override)
  - file: <dir>/config.json
  - default baseUrl: https://api.aethersystems.net/cloud
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

#: The public API front door. The backend is exposed under /cloud; the apex
#: returns an info blob, so the /cloud suffix is required for every API call.
#: Override with AETHER_BASE_URL.
DEFAULT_BASE_URL = "https://api.aethersystems.net/cloud"

#: Config defaults — keys mirror src/core/config.ts DEFAULT_CONFIG (plus
#: ``backend``, the Python host's local/cloud routing knob).
DEFAULT_CONFIG: dict[str, Any] = {
    "baseUrl": DEFAULT_BASE_URL,
    "defaultModel": "",
    "backend": "auto",
    "permissionMode": "ask",
    "autoApply": False,
    "telemetry": True,
}

CONFIG_FILENAME = "config.json"


def config_dir() -> Path:
    """The agent config directory: ``~/.config/aether`` (env AETHER_CONFIG_DIR override)."""
    override = os.environ.get("AETHER_CONFIG_DIR")
    if override:
        return Path(override)
    return Path.home() / ".config" / "aether"


def config_path() -> Path:
    """Path to ``<config_dir>/config.json``."""
    return config_dir() / CONFIG_FILENAME


def load_config() -> dict[str, Any]:
    """Load config.json merged over the defaults; never raises.

    Missing file -> a copy of the defaults. Corrupt/partial JSON -> defaults with
    any readable keys merged in (a bad config can't brick the CLI). The
    AETHER_BASE_URL env var ALWAYS wins for ``baseUrl`` (env > file > default).
    """
    cfg = dict(DEFAULT_CONFIG)
    path = config_path()
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                cfg.update(raw)
        except (json.JSONDecodeError, OSError, ValueError):
            # Corrupt config must never crash — fall back to defaults.
            cfg = dict(DEFAULT_CONFIG)
    env_base = os.environ.get("AETHER_BASE_URL")
    if env_base:
        cfg["baseUrl"] = env_base
    return cfg


def save_config(cfg: dict[str, Any]) -> Path:
    """Write ``cfg`` to ``<config_dir>/config.json`` (creating the dir). Returns the path."""
    d = config_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = config_path()
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_CONFIG",
    "CONFIG_FILENAME",
    "config_dir",
    "config_path",
    "load_config",
    "save_config",
]
