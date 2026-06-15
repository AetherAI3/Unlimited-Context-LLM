# aether_agent/agent_store.py
# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""File-per-agent persistence.

Definitions live at ``<config>/agents/<name>.json``; each agent's memory pool is
separate local state at ``<config>/agents/<name>/pool/``. Sharing an agent =
copying its one ``<name>.json``. Honors ``AETHER_CONFIG_DIR`` (via ``config``).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from aether_agent.agent_profile import Agent
from aether_agent.config import config_dir


def agents_dir() -> Path:
    return Path(config_dir()) / "agents"


def agent_path(name: str) -> Path:
    d = agents_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.json"


def agent_pool_dir(name: str) -> Path:
    return agents_dir() / name / "pool"


def exists(name: str) -> bool:
    return agent_path(name).is_file()


def list_agents() -> list[str]:
    d = agents_dir()
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


def create(agent: Agent) -> None:
    if exists(agent.name):
        raise FileExistsError(f"agent already exists: {agent.name}")
    save(agent)


def save(agent: Agent) -> None:
    agent_path(agent.name).write_text(
        json.dumps(agent.to_dict(), indent=2) + "\n", encoding="utf-8"
    )


def load(name: str) -> Agent:
    path = agent_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"no such agent: {name}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        raise ValueError(f"agent file is corrupt: {name} ({e})") from e
    return Agent.from_dict(data)


def delete(name: str, *, purge: bool = False) -> None:
    path = agent_path(name)
    if path.is_file():
        path.unlink()
    if purge:
        pool_parent = agents_dir() / name
        if pool_parent.is_dir():
            shutil.rmtree(pool_parent, ignore_errors=True)


__all__ = [
    "agents_dir", "agent_path", "agent_pool_dir",
    "exists", "list_agents", "create", "save", "load", "delete",
]
