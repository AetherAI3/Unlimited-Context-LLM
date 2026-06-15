# tests/test_agent_store.py
import pytest

from aether_agent import agent_store
from aether_agent.agent_profile import Agent


@pytest.fixture(autouse=True)
def _tmp_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))


def test_create_load_list_exists():
    a = Agent.from_dict({"name": "jane", "pool_gb": 7})
    agent_store.create(a)
    assert agent_store.exists("jane")
    assert agent_store.list_agents() == ["jane"]
    loaded = agent_store.load("jane")
    assert loaded.name == "jane" and loaded.pool_gb == 7


def test_create_duplicate_raises():
    agent_store.create(Agent.from_dict({"name": "jane"}))
    with pytest.raises(Exception):
        agent_store.create(Agent.from_dict({"name": "jane"}))


def test_load_missing_raises():
    with pytest.raises(Exception):
        agent_store.load("nope")


def test_pool_dir_path_layout():
    p = agent_store.agent_pool_dir("jane")
    assert p.name == "pool" and p.parent.name == "jane"
    assert "agents" in str(p)


def test_delete_keeps_pool_unless_purge():
    a = Agent.from_dict({"name": "jane"})
    agent_store.create(a)
    pool = agent_store.agent_pool_dir("jane")
    pool.mkdir(parents=True, exist_ok=True)
    agent_store.delete("jane", purge=False)
    assert not agent_store.exists("jane")
    assert pool.exists()  # pool kept
    agent_store.create(a)
    agent_store.delete("jane", purge=True)
    assert not pool.exists()


def test_corrupt_file_fails_soft():
    agent_store.agent_path("broken").write_text("{ not json", encoding="utf-8")
    with pytest.raises(Exception):
        agent_store.load("broken")
    assert "broken" in agent_store.list_agents()  # listing must not crash
