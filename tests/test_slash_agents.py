# tests/test_slash_agents.py
import pytest

from aether_agent.slash import SlashContext, dispatch


@pytest.fixture(autouse=True)
def _tmp_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))


def test_slash_routes_agent_commands():
    res = dispatch(SlashContext(api=None), "/new-agent jane")
    assert "jane" in res["text"]
    res2 = dispatch(SlashContext(api=None, active_agent=""), "/agent jane")
    assert res2.get("switch_agent") == "jane"
    res3 = dispatch(SlashContext(api=None), "/agents")
    assert "jane" in res3["text"]


def test_help_lists_agent_commands():
    res = dispatch(SlashContext(api=None), "/help")
    assert "/agent" in res["text"] and "/new-agent" in res["text"]
