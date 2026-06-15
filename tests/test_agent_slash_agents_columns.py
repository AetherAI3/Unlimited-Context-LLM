# tests/test_agent_slash_agents_columns.py
import pytest

from aether_agent import agent_slash
from aether_agent.agent_profile import Agent


@pytest.fixture(autouse=True)
def _tmp_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))


def test_agents_renders_columns():
    from aether_agent import agent_store
    agent_store.create(Agent.from_dict({"name": "jane", "model": "qwen2.5-coder:7b", "pool_gb": 5}))
    agent_store.create(Agent.from_dict({"name": "neo", "model": "qwen3-coder:30b", "pool_gb": 10}))
    out = agent_slash.dispatch_agent("/agents", active="neo")["text"]
    assert "jane" in out and "neo" in out
    assert "qwen2.5-coder:7b" in out and "qwen3-coder:30b" in out
    assert "GB" in out  # the card stats line
    # column-dashboard format (not the old B simple list):
    assert "cmds" in out and " GB" in out
    assert "* neo" in out  # active marker on its own card


def test_agents_empty():
    out = agent_slash.dispatch_agent("/agents", active="")["text"]
    assert "no agents" in out.lower()
