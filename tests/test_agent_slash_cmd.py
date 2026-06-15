# tests/test_agent_slash_cmd.py
import pytest

from aether_agent import agent_slash
from aether_agent.agent_profile import Agent


@pytest.fixture(autouse=True)
def _tmp_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))


def _mk(name="jane"):
    from aether_agent import agent_store
    agent_store.create(Agent.from_dict({"name": name}))


def test_cmd_add_list_remove():
    _mk()
    res = agent_slash.dispatch_agent("/agent jane cmd add review = Review $1 for bugs", active="")
    assert "review" in res["text"]
    assert Agent.load("jane").commands == {"review": "Review $1 for bugs"}
    res2 = agent_slash.dispatch_agent("/agent jane cmd list", active="")
    assert "review" in res2["text"]
    agent_slash.dispatch_agent("/agent jane cmd remove review", active="")
    assert Agent.load("jane").commands == {}


def test_cmd_add_requires_equals_and_rejects_reserved():
    _mk()
    res = agent_slash.dispatch_agent("/agent jane cmd add review Review $1", active="")  # no '='
    assert "=" in res["text"]
    res2 = agent_slash.dispatch_agent("/agent jane cmd add help = x", active="")  # reserved
    assert "command" in res2["text"].lower()
    assert Agent.load("jane").commands == {}  # nothing saved


def test_run_verb_expands_to_run_agent():
    _mk()
    agent_slash.dispatch_agent("/agent jane cmd add review = Review $1 for bugs", active="")
    res = agent_slash.dispatch_agent("/agent jane run review src/api.py", active="")
    assert res.get("run_agent") == {"name": "jane", "task": "Review src/api.py for bugs"}


def test_run_unknown_command_friendly():
    _mk()
    res = agent_slash.dispatch_agent("/agent jane run ghost x", active="")
    assert "no" in res["text"].lower() or "unknown" in res["text"].lower()
