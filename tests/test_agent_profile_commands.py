from aether_agent import agent_profile
from aether_agent.agent_profile import Agent


def test_commands_map_validates():
    good = Agent.from_dict({"name": "jane", "commands": {"review": "Review $1 for bugs"}})
    assert agent_profile.validate(good) == []


def test_reserved_or_empty_command_rejected():
    bad = Agent.from_dict({"name": "jane", "commands": {"help": "x"}})  # reserved name
    assert any("command" in p.lower() for p in agent_profile.validate(bad))
    empty = Agent.from_dict({"name": "jane", "commands": {"ok": "   "}})  # empty template
    assert any("template" in p.lower() for p in agent_profile.validate(empty))


def test_define_command_allowed_in_tools():
    a = Agent.from_dict({"name": "jane", "tools": ["read_file", "define_command"]})
    assert agent_profile.validate(a) == []  # define_command is an allowed agent tool


def test_is_valid_command_name():
    assert agent_profile.is_valid_command_name("review")
    assert not agent_profile.is_valid_command_name("help")        # reserved
    assert not agent_profile.is_valid_command_name("Bad Name")    # not a slug
