# tests/test_agent_slash.py
import pytest

from aether_agent import agent_slash
from aether_agent.agent_profile import Agent


@pytest.fixture(autouse=True)
def _tmp_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))


def _mk(name="jane"):
    from aether_agent import agent_store
    agent_store.create(Agent.from_dict({"name": name}))


def test_new_agent_creates():
    res = agent_slash.dispatch_agent("/new-agent jane", active="")
    assert "jane" in res["text"]
    from aether_agent import agent_store
    assert agent_store.exists("jane")


def test_new_agent_duplicate_reports():
    _mk()
    res = agent_slash.dispatch_agent("/new-agent jane", active="")
    assert "exist" in res["text"].lower()


def test_agents_lists_with_active_marker():
    _mk("jane")
    _mk("neo")
    res = agent_slash.dispatch_agent("/agents", active="neo")
    assert "jane" in res["text"] and "neo" in res["text"]


def test_agent_switch_returns_flag():
    _mk()
    res = agent_slash.dispatch_agent("/agent jane", active="")
    assert res.get("switch_agent") == "jane"


def test_agent_run_returns_flag():
    _mk()
    res = agent_slash.dispatch_agent('/agent jane fix the bug', active="")
    assert res.get("run_agent") == {"name": "jane", "task": "fix the bug"}


def test_agent_set_persists():
    _mk()
    res = agent_slash.dispatch_agent("/agent jane set memory 10", active="")
    assert "10" in res["text"]
    assert Agent.load("jane").pool_gb == 10


def test_agent_set_invalid_reports_not_crash():
    _mk()
    res = agent_slash.dispatch_agent("/agent jane set permission bogus", active="")
    assert "permission" in res["text"].lower()
    assert Agent.load("jane").permission == "ask"  # unchanged


def test_unknown_agent_is_friendly():
    res = agent_slash.dispatch_agent("/agent ghost", active="")
    assert "no" in res["text"].lower() or "unknown" in res["text"].lower()
