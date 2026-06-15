# tests/test_agent_runner.py
from aether_agent import agent_runner
from aether_agent.agent_profile import Agent
from aether_agent.tools import Tools


def test_policy_tools_blocks_disallowed_and_allows_allowed(tmp_path):
    inner = Tools(str(tmp_path))
    pt = agent_runner._PolicyTools(inner, allowed={"read_file"}, permission="skip", confirm=lambda n, a: True)
    assert "not allowed" in pt.execute("write_file", {"path": "x", "content": "y"}).lower()
    out = pt.execute("read_file", {"path": "missing"})
    assert "no such file" in out.lower()  # delegated to inner


def test_policy_tools_ask_denies_destructive_without_confirm(tmp_path):
    inner = Tools(str(tmp_path))
    pt = agent_runner._PolicyTools(inner, allowed={"write_file"}, permission="ask", confirm=lambda n, a: False)
    assert "denied" in pt.execute("write_file", {"path": "x", "content": "y"}).lower()
    pt2 = agent_runner._PolicyTools(inner, allowed={"write_file"}, permission="ask", confirm=lambda n, a: True)
    assert "wrote" in pt2.execute("write_file", {"path": "x.txt", "content": "y"}).lower()


def test_run_builds_session_with_agent_pool_and_streams(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    captured = {}

    class _FakeSession:
        def remember(self, *a, **k): pass
        def status_dict(self): return {}
        def close(self): captured["closed"] = True

    def fake_session_factory(agent):
        captured["pool_gb"] = agent.pool_gb
        captured["model"] = agent.model
        return _FakeSession()

    class _FakeLLM:
        def chat(self, messages, tools=None):
            captured["system"] = messages[0]["content"]
            captured["schema_names"] = [t["function"]["name"] for t in (tools or [])]
            return {"role": "assistant", "content": "done", "tool_calls": []}

    a = Agent.from_dict({"name": "jane", "pool_gb": 9, "persona": "You are Jane.",
                         "tools": ["read_file", "web_search"], "permission": "skip"})
    events = list(agent_runner.run(
        a, "hello", cwd=str(tmp_path), llm=_FakeLLM(), session_factory=fake_session_factory,
    ))
    assert captured["pool_gb"] == 9 and captured["model"] == a.model
    assert captured["system"] == "You are Jane."
    assert set(captured["schema_names"]) == {"read_file", "web_search"}
    assert any(e["type"] == "done" for e in events)
    assert captured.get("closed") is True
