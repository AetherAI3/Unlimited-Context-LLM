# tests/test_agent_runner.py
import json

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


def test_patch_preview_reaches_approval_before_mutation_and_stale_edits_conflict(tmp_path, capsys):
    target = tmp_path / "space é.txt"
    target.write_text("before\nafter\n", encoding="utf-8")
    inner = Tools(str(tmp_path))
    digest = json.loads(inner.read_file(target.name))["sha256"]
    patch = {"path": target.name, "expected_sha256": digest,
             "old_text": "before", "new_text": "updated"}
    seen = []

    def confirm(name, args):
        assert name == "patch_file"
        assert target.read_text(encoding="utf-8") == "before\nafter\n"
        seen.append(args["_patch_preview"])
        return True

    policy = agent_runner._PolicyTools(inner, allowed={"patch_file"}, permission="ask", confirm=confirm)
    assert "patched" in policy.execute("patch_file", patch)
    assert target.read_text(encoding="utf-8") == "updated\nafter\n"
    assert '- "before"\n+ "updated"' in seen[0]
    assert seen[0] in capsys.readouterr().err

    digest = json.loads(inner.read_file(target.name))["sha256"]
    stale = {"path": target.name, "expected_sha256": digest,
             "old_text": "updated", "new_text": "agent edit"}

    def user_edits_after_preview(_name, args):
        assert '- "updated"\n+ "agent edit"' in args["_patch_preview"]
        target.write_text("user edit\nafter\n", encoding="utf-8")
        return True

    policy = agent_runner._PolicyTools(inner, allowed={"patch_file"}, permission="ask",
                                       confirm=user_edits_after_preview)
    assert "conflict" in policy.execute("patch_file", stale)
    assert target.read_text(encoding="utf-8") == "user edit\nafter\n"


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
