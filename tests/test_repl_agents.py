import io

import pytest

from aether_agent import repl
from aether_agent.agent_profile import Agent


@pytest.fixture(autouse=True)
def _tmp_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))


def test_apply_agent_sets_prompt_accent_persona():
    a = Agent.from_dict({"name": "jane", "accent": "green", "prompt": "jane > ", "banner": "~J~"})
    state = repl._apply_agent(a)
    assert state["prompt"] == "jane > "
    assert "32" in state["accent_ansi"]  # green = 32
    assert state["banner"] == "~J~"


def test_run_agent_flag_streams_via_runner(monkeypatch):
    from aether_agent import agent_store
    agent_store.create(Agent.from_dict({"name": "jane"}))
    seen = {}

    def fake_run(agent, task, **kw):
        seen["name"] = agent.name
        seen["task"] = task
        yield {"type": "done", "text": "ran"}

    monkeypatch.setattr("aether_agent.agent_runner.run", fake_run)
    out = io.StringIO()
    repl._handle_agent_action({"run_agent": {"name": "jane", "task": "do x"}}, out)
    assert seen == {"name": "jane", "task": "do x"}
    assert "ran" in out.getvalue()
