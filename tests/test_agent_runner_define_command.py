import pytest

from aether_agent import agent_runner
from aether_agent.agent_profile import Agent


@pytest.fixture(autouse=True)
def _tmp_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))


def test_define_command_schema_only_when_allowed():
    with_tool = Agent.from_dict({"name": "jane", "tools": ["read_file", "define_command"]})
    without = Agent.from_dict({"name": "jane", "tools": ["read_file"]})
    assert "define_command" in agent_runner._offered_tool_names(with_tool)
    assert "define_command" not in agent_runner._offered_tool_names(without)


def test_define_command_writes_and_reloads():
    from aether_agent import agent_store
    a = Agent.from_dict({"name": "jane", "tools": ["read_file", "define_command"]})
    agent_store.create(a)
    pt = agent_runner._PolicyTools(
        inner=None, allowed=set(a.tools), permission="skip", confirm=lambda n, x: True, agent_name="jane",
    )
    out = pt.execute("define_command", {"name": "review", "template": "Review $1 for bugs"})
    assert "defined" in out.lower()
    assert agent_store.load("jane").commands == {"review": "Review $1 for bugs"}


def test_define_command_refused_when_not_allowed():
    pt = agent_runner._PolicyTools(
        inner=None, allowed={"read_file"}, permission="skip", confirm=lambda n, x: True, agent_name="jane",
    )
    out = pt.execute("define_command", {"name": "review", "template": "x"})
    assert "not allowed" in out.lower()


def test_define_command_rejects_reserved_name():
    from aether_agent import agent_store
    agent_store.create(Agent.from_dict({"name": "jane", "tools": ["define_command"]}))
    pt = agent_runner._PolicyTools(
        inner=None, allowed={"define_command"}, permission="skip", confirm=lambda n, x: True, agent_name="jane",
    )
    out = pt.execute("define_command", {"name": "help", "template": "x"})
    assert "refused" in out.lower()
    assert agent_store.load("jane").commands == {}
