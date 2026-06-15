# tests/test_multi_runner.py
from aether_agent import multi_runner
from aether_agent.agent_profile import Agent


def test_run_many_streams_labeled_events_and_summaries(monkeypatch):
    scripts = {
        "jane": [{"type": "monologue", "text": "j1"}, {"type": "tool_call", "name": "read_file", "args": {}},
                 {"type": "done", "text": "jane done", "ok": True}],
        "neo": [{"type": "monologue", "text": "n1"}, {"type": "done", "text": "neo done", "ok": True}],
    }

    def fake_run(agent, task, **kw):
        for ev in scripts[agent.name]:
            yield ev

    monkeypatch.setattr("aether_agent.agent_runner.run", fake_run)
    seen = []
    jobs = [(Agent.from_dict({"name": "jane"}), "t1"), (Agent.from_dict({"name": "neo"}), "t2")]
    summaries = multi_runner.run_many(jobs, emit=lambda label, ev: seen.append((label, ev.get("type"))))
    labels = {label for label, _ in seen}
    assert labels == {"jane", "neo"}
    assert ("jane", "monologue") in seen and ("neo", "done") in seen
    by_name = {s["name"]: s for s in summaries}
    assert by_name["jane"]["tool_calls"] == 1 and by_name["jane"]["ok"] is True
    assert by_name["neo"]["summary"] == "neo done"


def test_cap_at_8(monkeypatch):
    def fake_run(agent, task, **kw):
        yield {"type": "done", "text": "x", "ok": True}
    monkeypatch.setattr("aether_agent.agent_runner.run", fake_run)
    jobs = [(Agent.from_dict({"name": f"a{i}"}), "t") for i in range(12)]
    summaries = multi_runner.run_many(jobs, emit=lambda label, ev: None)
    assert len(summaries) == 8  # capped


def test_shared_write_lock_passed_to_each_runner(monkeypatch):
    locks_seen = []

    def fake_run(agent, task, *, write_lock=None, **kw):
        locks_seen.append(write_lock)
        yield {"type": "done", "text": "x", "ok": True}

    monkeypatch.setattr("aether_agent.agent_runner.run", fake_run)
    jobs = [(Agent.from_dict({"name": "a"}), "t"), (Agent.from_dict({"name": "b"}), "t")]
    multi_runner.run_many(jobs, emit=lambda label, ev: None)
    assert locks_seen[0] is locks_seen[1] is not None  # same shared lock for all jobs


def test_duplicate_agent_runs_once(monkeypatch):
    # The same agent twice in one batch must run ONCE (no concurrent Sessions on
    # its shared pool dir, no summary collision).
    runs = []

    def fake_run(agent, task, **kw):
        runs.append((agent.name, task))
        yield {"type": "done", "text": "x", "ok": True}

    monkeypatch.setattr("aether_agent.agent_runner.run", fake_run)
    jobs = [(Agent.from_dict({"name": "jane"}), "a"), (Agent.from_dict({"name": "jane"}), "b")]
    summaries = multi_runner.run_many(jobs, emit=lambda label, ev: None)
    assert len(runs) == 1 and runs[0] == ("jane", "a")   # first kept, dup dropped
    assert len(summaries) == 1 and summaries[0]["name"] == "jane"
