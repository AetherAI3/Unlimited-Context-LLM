# tests/test_agent_profile.py
from aether_agent import agent_profile
from aether_agent.agent_profile import Agent


def test_from_dict_fills_defaults_and_ignores_effort():
    a = Agent.from_dict({"name": "jane", "effort": "codepro"})  # effort ignored
    assert a.name == "jane"
    assert a.model  # a default model
    assert a.pool_gb == 5
    assert a.permission == "ask"
    assert set(a.tools)  # defaults to the canonical 8
    assert not hasattr(a, "effort")


def test_validate_catches_bad_fields():
    bad = Agent.from_dict({
        "name": "Bad Name!", "pool_gb": 0, "permission": "nope",
        "tools": ["read_file", "made_up_tool"], "max_steps": 9999, "accent": "chartreuse",
    })
    problems = agent_profile.validate(bad)
    joined = " ".join(problems).lower()
    assert "name" in joined
    assert "pool_gb" in joined or "pool" in joined
    assert "permission" in joined
    assert "tool" in joined
    assert "max_steps" in joined or "step" in joined
    assert "accent" in joined


def test_valid_agent_has_no_problems():
    a = Agent.from_dict({"name": "jane"})
    assert agent_profile.validate(a) == []


def test_roundtrip_and_set_returns_new_validated_copy():
    a = Agent.from_dict({"name": "jane"})
    d = a.to_dict()
    assert d["name"] == "jane" and "commands" in d and "effort" not in d
    b = a.set(pool_gb=10, accent="green")
    assert b.pool_gb == 10 and b.accent == "green"
    assert a.pool_gb == 5  # original unchanged (immutable)


def test_set_with_invalid_value_raises():
    a = Agent.from_dict({"name": "jane"})
    try:
        a.set(permission="bogus")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "permission" in str(e).lower()
