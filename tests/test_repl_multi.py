# tests/test_repl_multi.py
import io

import pytest

from aether_agent import repl
from aether_agent.agent_profile import Agent


@pytest.fixture(autouse=True)
def _tmp_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))


def test_parse_multi_two_runs():
    from aether_agent import agent_store
    agent_store.create(Agent.from_dict({"name": "jane"}))
    agent_store.create(Agent.from_dict({"name": "neo"}))
    jobs = repl._parse_multi("/agent jane fix tests \\ /agent neo write docs")
    assert jobs is not None
    assert [(a.name, t) for a, t in jobs] == [("jane", "fix tests"), ("neo", "write docs")]


def test_parse_multi_single_returns_none():
    from aether_agent import agent_store
    agent_store.create(Agent.from_dict({"name": "jane"}))
    assert repl._parse_multi("/agent jane fix tests") is None  # no ' \\ ' -> not multi


def test_handle_multi_streams_labeled(monkeypatch):
    from aether_agent import agent_store
    agent_store.create(Agent.from_dict({"name": "jane"}))
    agent_store.create(Agent.from_dict({"name": "neo"}))

    def fake_run_many(jobs, *, emit, **kw):
        for a, _ in jobs:
            emit(a.name, {"type": "done", "text": f"{a.name} ok"})
        return [{"name": a.name, "ok": True, "summary": f"{a.name} ok", "tool_calls": 0} for a, _ in jobs]

    monkeypatch.setattr("aether_agent.multi_runner.run_many", fake_run_many)
    out = io.StringIO()
    jobs = repl._parse_multi("/agent jane t \\ /agent neo t")
    repl._handle_multi(jobs, out)
    text = out.getvalue()
    assert "[jane]" in text and "[neo]" in text
    assert "done" in text.lower()
