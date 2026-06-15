# tests/test_agent_commands.py
import pytest

from aether_agent import agent_commands
from aether_agent.agent_profile import Agent


def test_valid_name():
    assert agent_commands.valid_name("review")
    assert not agent_commands.valid_name("help")       # reserved
    assert not agent_commands.valid_name("Bad Name")   # not a slug


def test_expand_positional_all_and_dollar():
    assert agent_commands.expand("Review $1 and $2", ["a.py", "b.py"]) == "Review a.py and b.py"
    assert agent_commands.expand("all: $*", ["x", "y", "z"]) == "all: x y z"
    assert agent_commands.expand("all: $@", ["x", "y"]) == "all: x y"
    assert agent_commands.expand("cost $$5 for $1", ["a"]) == "cost $5 for a"
    assert agent_commands.expand("missing $3 here", ["a"]) == "missing  here"  # absent -> ""
    assert agent_commands.expand("no placeholders", []) == "no placeholders"


def test_add_remove_list_roundtrip():
    a = Agent.from_dict({"name": "jane"})
    a = agent_commands.add(a, "review", "Review $1 for bugs")
    assert agent_commands.list_commands(a) == {"review": "Review $1 for bugs"}
    a = agent_commands.remove(a, "review")
    assert agent_commands.list_commands(a) == {}


def test_add_rejects_bad_name_and_empty_template():
    a = Agent.from_dict({"name": "jane"})
    with pytest.raises(ValueError):
        agent_commands.add(a, "help", "x")        # reserved
    with pytest.raises(ValueError):
        agent_commands.add(a, "ok", "   ")        # empty template
