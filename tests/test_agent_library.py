# tests/test_agent_library.py
import pytest

from aether_agent import Agent  # the public library API


@pytest.fixture(autouse=True)
def _tmp_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))


def test_create_load_list_set_save_delete_roundtrip():
    a = Agent.create("jane", model="qwen2.5-coder:7b", pool_gb=8)
    assert a.pool_gb == 8
    assert Agent.list() == ["jane"]
    b = Agent.load("jane").set(pool_gb=12)
    b.save()
    assert Agent.load("jane").pool_gb == 12
    b.delete()
    assert Agent.list() == []


def test_run_streams_via_runner(monkeypatch):
    Agent.create("jane")

    def fake_run(agent, task, **kw):
        yield {"type": "done", "text": f"{agent.name}:{task}"}

    monkeypatch.setattr("aether_agent.agent_runner.run", fake_run)
    out = list(Agent.load("jane").run("hi"))
    assert out[-1]["text"] == "jane:hi"
