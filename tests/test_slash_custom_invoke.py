# tests/test_slash_custom_invoke.py
import pytest

from aether_agent.slash import SlashContext, dispatch
from aether_agent.agent_profile import Agent


@pytest.fixture(autouse=True)
def _tmp_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))


def _mk_with_cmd():
    from aether_agent import agent_store, agent_commands
    a = agent_commands.add(Agent.from_dict({"name": "jane"}), "review", "Review $1 for bugs")
    agent_store.create(a)


def test_builtin_wins_over_custom_same_name():
    res = dispatch(SlashContext(api=None, active_agent="jane"), "/help")
    assert "/exit" in res["text"]  # the built-in help text resolves, not a custom


def test_active_agent_custom_resolves():
    _mk_with_cmd()
    res = dispatch(SlashContext(api=None, active_agent="jane"), "/review src/api.py")
    assert res.get("run_agent") == {"name": "jane", "task": "Review src/api.py for bugs"}


def test_no_active_agent_unknown():
    _mk_with_cmd()
    res = dispatch(SlashContext(api=None, active_agent=""), "/review x")
    assert "unknown" in res["text"].lower()


def test_unknown_name_with_active_agent_is_unknown():
    _mk_with_cmd()
    res = dispatch(SlashContext(api=None, active_agent="jane"), "/ghost x")
    assert "unknown" in res["text"].lower()
