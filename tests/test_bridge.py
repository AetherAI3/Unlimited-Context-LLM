"""Bridge tests — protocol codec, skill matcher, and the headless brain loop.

All pure: no Ollama, no subprocess, no filesystem. A scripted FakeTransport
feeds host commands and captures emitted events; a fake chat_fn stands in for
the model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import pytest

from aether_agent import protocol, skills
from aether_agent.headless import run_brain
from aether_agent.tools import Tools

_FIXTURE = Path(__file__).parent / "fixtures" / "bridge_conformance.json"


# --- protocol codec --------------------------------------------------------
def test_encode_decode_round_trip():
    ev = protocol.tool_call("c1", "read_file", {"path": "a.py"})
    line = protocol.encode(ev)
    assert line.endswith("\n")
    assert protocol.decode(line) == ev


def test_decode_rejects_garbage_and_blanks():
    assert protocol.decode("") is None
    assert protocol.decode("   ") is None
    assert protocol.decode("not json") is None
    assert protocol.decode('{"no":"type"}') is None
    assert protocol.decode('{"type":"status","phase":"reasoning"}')["phase"] == "reasoning"


def test_status_denominator_is_pool_reach():
    s = protocol.status("reasoning", 100, 5 * 233_000_000)
    assert s["pool_cap"] == 1_165_000_000


def test_skill_event_in_vocab():
    assert protocol.EV_SKILL in protocol.BRAIN_EVENTS
    assert protocol.skill("fix-failing-tests", "x")["name"] == "fix-failing-tests"


# --- skill matcher ---------------------------------------------------------
def test_match_skills_picks_relevant_and_respects_budget():
    got = skills.match_skills("the pytest suite has failing tests, fix them")
    names = [s.name for s in got]
    assert "fix-failing-tests" in names
    assert len(got) <= skills.SKILL_BUDGET


def test_match_skills_excludes_loaded():
    got = skills.match_skills("debug the async race", exclude=frozenset({"debug-async-race"}))
    assert "debug-async-race" not in [s.name for s in got]


def test_skill_as_prompt_carries_acceptance():
    sk = skills.LIBRARY[0]
    p = sk.as_prompt()
    assert "DONE WHEN:" in p and sk.acceptance in p


# --- scripted transport ----------------------------------------------------
class FakeTransport:
    """Feeds queued host commands; captures emitted events. Auto-stamps a
    tool_result whose id is None with the id of the last tool_call emitted, so
    tests don't need to predict the brain's generated checkpoint id."""

    def __init__(self, inbox: list[dict[str, Any]]):
        self._inbox = list(inbox)
        self.sent: list[dict[str, Any]] = []

    def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)

    def recv(self) -> Optional[dict[str, Any]]:
        if not self._inbox:
            return None
        msg = self._inbox.pop(0)
        if msg.get("type") == protocol.CMD_TOOL_RESULT and msg.get("id") is None:
            last = next(
                (s for s in reversed(self.sent) if s["type"] == protocol.EV_TOOL_CALL), None
            )
            if last:
                msg["id"] = last["id"]
        return msg

    def types(self) -> list[str]:
        return [m["type"] for m in self.sent]


def _assistant_tool_call(name: str, args_json: str):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call-1", "function": {"name": name, "arguments": args_json}}],
    }


def test_brain_emits_tool_call_then_done():
    turns = iter([
        _assistant_tool_call("write_file", '{"path": "x.py", "content": "print(1)"}'),
        {"role": "assistant", "content": "all set", "tool_calls": []},
    ])

    def fake_chat(messages, tools):
        return next(turns)

    transport = FakeTransport([
        {"type": protocol.CMD_TASK, "text": "write x.py", "pool_gb": 5},
        {"type": protocol.CMD_TOOL_RESULT, "id": "call-1", "output": "[wrote x.py]", "exit_code": 0},
        # the no-call turn triggers a FINAL VERIFY run_tests (id auto-stamped) — green:
        {"type": protocol.CMD_TOOL_RESULT, "id": None, "output": "[exit 0]\nall passed", "exit_code": 0},
    ])

    assert run_brain(transport, chat_fn=fake_chat) == 0
    types = transport.types()
    assert protocol.EV_TOOL_CALL in types
    assert types[-1] == protocol.EV_DONE
    done = transport.sent[-1]
    assert done["ok"] is True  # ok derived from the green verify, not self-report
    tc = next(m for m in transport.sent if m["type"] == protocol.EV_TOOL_CALL)
    assert tc["name"] == "write_file" and tc["args"]["path"] == "x.py"


def test_brain_loads_skill_for_test_task():
    turns = iter([{"role": "assistant", "content": "nothing to do", "tool_calls": []}])

    def fake_chat(messages, tools):
        # the skill procedure must already be pinned into the system messages
        assert any("PROCEDURE (skill: fix-failing-tests)" in m.get("content", "") for m in messages)
        return next(turns)

    transport = FakeTransport([
        {"type": protocol.CMD_TASK, "text": "fix the failing pytest tests", "pool_gb": 5},
        {"type": protocol.CMD_TOOL_RESULT, "id": None, "output": "[exit 0]\nall passed", "exit_code": 0},
    ])
    assert run_brain(transport, chat_fn=fake_chat) == 0
    skill_evs = [m for m in transport.sent if m["type"] == protocol.EV_SKILL]
    assert any(m["name"] == "fix-failing-tests" for m in skill_evs)


def test_brain_checkpoints_on_green_tests():
    turns = iter([
        _assistant_tool_call("run_tests", "{}"),
        {"role": "assistant", "content": "green", "tool_calls": []},
    ])

    def fake_chat(messages, tools):
        return next(turns)

    transport = FakeTransport([
        {"type": protocol.CMD_TASK, "text": "make the build pass", "pool_gb": 10},
        {"type": protocol.CMD_TOOL_RESULT, "id": "call-1", "output": "[exit 0]\nall passed", "exit_code": 0},
        # reply to the brain's checkpoint git_commit (id auto-stamped):
        {"type": protocol.CMD_TOOL_RESULT, "id": None, "output": "[exit 0]\n a1b2c3d4e5 committed", "exit_code": 0},
        # reply to the FINAL VERIFY run_tests on the no-call turn (id auto-stamped):
        {"type": protocol.CMD_TOOL_RESULT, "id": None, "output": "[exit 0]\nall passed", "exit_code": 0},
    ])

    assert run_brain(transport, chat_fn=fake_chat) == 0
    assert protocol.EV_CHECKPOINT in transport.types()
    ckpt = next(m for m in transport.sent if m["type"] == protocol.EV_CHECKPOINT)
    assert ckpt["git_sha"] == "a1b2c3d4e5"


def test_brain_errors_without_task_first():
    transport = FakeTransport([{"type": protocol.CMD_CONTROL, "action": "resume"}])
    assert run_brain(transport, chat_fn=lambda m, t: {}) == 1
    assert transport.types()[-1] == protocol.EV_ERROR


# --- probe 1: tool-call correlation ---------------------------------------
def test_two_sequential_tool_calls_stay_correlated():
    """Two tool calls in one assistant turn: each result must pair to its own id,
    in order. The brain blocks on each call, so the host replies in order."""
    two = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "A", "function": {"name": "read_file", "arguments": '{"path": "a"}'}},
            {"id": "B", "function": {"name": "read_file", "arguments": '{"path": "b"}'}},
        ],
    }
    turns = iter([two, {"role": "assistant", "content": "done", "tool_calls": []}])
    transport = FakeTransport([
        {"type": protocol.CMD_TASK, "text": "read a and b", "pool_gb": 5},
        {"type": protocol.CMD_TOOL_RESULT, "id": "A", "output": "contents-of-a", "exit_code": 0},
        {"type": protocol.CMD_TOOL_RESULT, "id": "B", "output": "contents-of-b", "exit_code": 0},
        {"type": protocol.CMD_TOOL_RESULT, "id": None, "output": "[exit 0]\nall passed", "exit_code": 0},
    ])
    assert run_brain(transport, chat_fn=lambda m, t: next(turns)) == 0
    # the two model calls A, B emit in order (the final verify run_tests follows):
    calls = [m["id"] for m in transport.sent if m["type"] == protocol.EV_TOOL_CALL]
    assert calls[:2] == ["A", "B"]  # emitted in order, each id distinct


def test_malformed_tool_args_emit_a_countable_marker():
    """Un-parseable tool-call args are coerced to {} but surfaced as a marker
    (the small-model emission-failure signal the stress test buckets)."""
    bad = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "A", "function": {"name": "read_file", "arguments": "{not json"}}],
    }
    turns = iter([bad, {"role": "assistant", "content": "done", "tool_calls": []}])
    transport = FakeTransport([
        {"type": protocol.CMD_TASK, "text": "x", "pool_gb": 5},
        {"type": protocol.CMD_TOOL_RESULT, "id": "A", "output": "[no such file]", "exit_code": 1},
        {"type": protocol.CMD_TOOL_RESULT, "id": None, "output": "[exit 0]\nall passed", "exit_code": 0},
    ])
    assert run_brain(transport, chat_fn=lambda m, t: next(turns)) == 0
    markers = [m for m in transport.sent if m["type"] == protocol.EV_MONOLOGUE and "malformed-args" in m["text"]]
    assert len(markers) == 1 and "read_file" in markers[0]["text"]


def test_invented_tool_emits_a_countable_marker():
    """A call to a tool that doesn't exist is surfaced as a marker (emission fray
    signal, distinct from malformed args)."""
    bad = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "A", "function": {"name": "make_coffee", "arguments": "{}"}}],
    }
    turns = iter([bad, {"role": "assistant", "content": "done", "tool_calls": []}])
    transport = FakeTransport([
        {"type": protocol.CMD_TASK, "text": "x", "pool_gb": 5},
        {"type": protocol.CMD_TOOL_RESULT, "id": "A", "output": "[unknown tool: make_coffee]", "exit_code": 1},
        {"type": protocol.CMD_TOOL_RESULT, "id": None, "output": "[exit 0]\nall passed", "exit_code": 0},
    ])
    assert run_brain(transport, chat_fn=lambda m, t: next(turns)) == 0
    markers = [m for m in transport.sent if m["type"] == protocol.EV_MONOLOGUE and "invented-tool" in m["text"]]
    assert len(markers) == 1 and "make_coffee" in markers[0]["text"]


def test_tool_result_id_mismatch_fails_loud():
    """A tool_result with the wrong id is a protocol violation — error, not skip."""
    one = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "A", "function": {"name": "read_file", "arguments": "{}"}}],
    }
    transport = FakeTransport([
        {"type": protocol.CMD_TASK, "text": "x", "pool_gb": 5},
        {"type": protocol.CMD_TOOL_RESULT, "id": "WRONG", "output": "huh", "exit_code": 0},
    ])
    assert run_brain(transport, chat_fn=lambda m, t: one) == 1
    assert transport.types()[-1] == protocol.EV_ERROR
    assert "mismatch" in transport.sent[-1]["msg"]


# --- probe 2: path-guard canonicalization ---------------------------------
def test_path_guard_rejects_traversal_and_absolute(tmp_path):
    t = Tools(str(tmp_path))
    with pytest.raises(ValueError):
        t._safe("../../etc/passwd")
    outside = os.path.abspath(os.path.join(str(tmp_path), "..", "outside.txt"))
    with pytest.raises(ValueError):
        t._safe(outside)  # absolute path outside the workspace


def test_path_guard_rejects_symlink_escape(tmp_path):
    target = tmp_path.parent / "secret_outside"
    target.mkdir(exist_ok=True)
    link = tmp_path / "link"
    try:
        os.symlink(str(target), str(link), target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this platform")
    t = Tools(str(tmp_path))
    with pytest.raises(ValueError):
        t._safe("link/escaped.txt")  # follows the symlink, lands outside -> reject


# --- probe 3: shell hardening (exit code reaches the brain; stderr captured)
def test_shell_nonzero_exit_and_stderr_captured(tmp_path):
    t = Tools(str(tmp_path))
    out = t.run_shell('python -c "import sys; sys.stderr.write(\'boom\'); sys.exit(3)"')
    assert out.startswith("[exit 3]")  # non-zero exit surfaced
    assert "boom" in out  # stderr captured into the output


# --- probe 4: schema conformance (drift detector) -------------------------
def test_protocol_version_matches_fixture():
    fx = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert fx["protocol_version"] == protocol.PROTOCOL_VERSION


def test_every_fixture_message_is_in_the_vocab_and_round_trips():
    fx = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    for ev in fx["events"]:
        assert ev["type"] in protocol.BRAIN_EVENTS
        assert protocol.decode(protocol.encode(ev)) == ev  # lossless round-trip
    for cmd in fx["commands"]:
        assert cmd["type"] in protocol.HOST_COMMANDS
        assert protocol.decode(protocol.encode(cmd)) == cmd


def test_fixture_shapes_match_canonical_keysets_exactly():
    """STRONG drift detector: every fixture message's KEY-SET must equal the canonical
    constructor's, and the fixture must COVER every event/command type. The weak
    `decode(encode(ev)) == ev` round-trip above is a pass-through no-op for shape, so it
    missed the v1->v2 drift (a wire field added without updating the fixture). This catches it.
    """
    canon_ev = {
        m["type"]: frozenset(m.keys())
        for m in (
            protocol.stage("s", "f"),
            protocol.monologue("t", 1),
            protocol.skill("n", "r"),
            protocol.turn(1, 2, 0, 0, False, None),
            protocol.tool_call("c", "n", {}),
            protocol.telemetry(),
            protocol.status("p", 1, 2),
            protocol.checkpoint("sha"),
            protocol.done(True, "r", 0, ""),
            protocol.error("m"),
        )
    }
    canon_cmd = {
        protocol.CMD_TASK: frozenset({"type", "text", "cwd", "pool_gb", "effort", "model", "test_cmd"}),
        protocol.CMD_TOOL_RESULT: frozenset({"type", "id", "output", "exit_code"}),
        protocol.CMD_CONTROL: frozenset({"type", "action", "note"}),
    }
    fx = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    seen_ev = set()
    for ev in fx["events"]:
        t = ev["type"]
        assert frozenset(ev.keys()) == canon_ev[t], f"{t} fixture shape drift: {set(ev.keys()) ^ canon_ev[t]}"
        seen_ev.add(t)
    assert seen_ev == protocol.BRAIN_EVENTS, f"fixture missing event types: {protocol.BRAIN_EVENTS - seen_ev}"
    seen_cmd = set()
    for cmd in fx["commands"]:
        t = cmd["type"]
        assert frozenset(cmd.keys()) == canon_cmd[t], f"{t} command shape drift: {set(cmd.keys()) ^ canon_cmd[t]}"
        seen_cmd.add(t)
    assert seen_cmd == protocol.HOST_COMMANDS, f"fixture missing command types: {protocol.HOST_COMMANDS - seen_cmd}"


# --- probe 5: escaping is lossless (escape-not-strip), CJK + emoji ---------
def test_unicode_round_trips_losslessly():
    fx = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    for s in fx["unicode_round_trip"] + ["日本語テスト", "🤖🔥", "Ω≈ç√"]:
        wire = protocol.encode(protocol.monologue(s))
        # wire is ASCII-escaped (no raw non-ascii bytes) but decodes back exactly
        assert wire.isascii()
        assert protocol.decode(wire)["text"] == s


# --- staged lifecycle + steer control -------------------------------------
def test_stage_order_and_steer_injection():
    """Stages run recon -> execute -> ... -> reveal, and a /steer arriving mid
    tool-wait is injected into the thread as high-priority user guidance."""
    saw_steer: dict[str, bool] = {}

    def fake_chat(messages, tools):
        if any("User steer" in m.get("content", "") for m in messages):
            saw_steer["yes"] = True
            return {"role": "assistant", "content": "done", "tool_calls": []}
        return _assistant_tool_call("read_file", '{"path": "a"}')

    transport = FakeTransport([
        {"type": protocol.CMD_TASK, "text": "read a", "pool_gb": 5},
        {"type": protocol.CMD_CONTROL, "action": "steer", "note": "focus on the auth path"},
        {"type": protocol.CMD_TOOL_RESULT, "id": "call-1", "output": "contents", "exit_code": 0},
        {"type": protocol.CMD_TOOL_RESULT, "id": None, "output": "[exit 0]\nall passed", "exit_code": 0},
    ])
    assert run_brain(transport, chat_fn=fake_chat) == 0
    assert saw_steer.get("yes") is True  # steer reached the model's messages
    stages = [m["name"] for m in transport.sent if m["type"] == protocol.EV_STAGE]
    assert stages[0] == "recon"
    assert "execute" in stages
    assert stages[-1] == "reveal"


# --- loop fixes: verification gate + no-call != done + no-progress + turn ---
def test_no_call_never_claims_ok_while_tests_fail():
    """A no-call turn with RED verify must NOT finish ok; after the no-call limit
    it terminates 'stalled' — honest, not a false success."""
    transport = FakeTransport([
        {"type": protocol.CMD_TASK, "text": "fix the failing tests", "pool_gb": 5},
        {"type": protocol.CMD_TOOL_RESULT, "id": None, "output": "[exit 1]\n24 failed", "exit_code": 1},
        {"type": protocol.CMD_TOOL_RESULT, "id": None, "output": "[exit 1]\n24 failed", "exit_code": 1},
        {"type": protocol.CMD_TOOL_RESULT, "id": None, "output": "[exit 1]\n24 failed", "exit_code": 1},
    ])
    code = run_brain(transport, chat_fn=lambda m, t: {"role": "assistant", "content": "all done!", "tool_calls": []})
    assert code == 1  # not ok
    done = transport.sent[-1]
    assert done["type"] == protocol.EV_DONE
    assert done["ok"] is False and done["reason"] == "stalled" and done["remaining"] == 24


def test_verify_green_sets_ok_true():
    """A no-call turn whose verify is GREEN finishes ok — ground-truth, not self-report."""
    transport = FakeTransport([
        {"type": protocol.CMD_TASK, "text": "fix it", "pool_gb": 5},
        {"type": protocol.CMD_TOOL_RESULT, "id": None, "output": "[exit 0]\n42 passed", "exit_code": 0},
    ])
    code = run_brain(transport, chat_fn=lambda m, t: {"role": "assistant", "content": "", "tool_calls": []})
    assert code == 0
    done = transport.sent[-1]
    assert done["ok"] is True and done["remaining"] == 0


def test_unverified_when_no_test_cmd():
    """No test gate on the task -> never 'ok'; finishes 'unverified'."""
    transport = FakeTransport([{"type": protocol.CMD_TASK, "text": "do a thing", "pool_gb": 5, "test_cmd": ""}])
    code = run_brain(transport, chat_fn=lambda m, t: {"role": "assistant", "content": "done", "tool_calls": []})
    assert code == 1
    done = transport.sent[-1]
    assert done["ok"] is False and done["reason"] == "unverified"


def test_turn_instrumentation_emitted():
    """Each assistant turn appends a `turn` diag event (the §8 emission curve feed)."""
    transport = FakeTransport([
        {"type": protocol.CMD_TASK, "text": "x", "pool_gb": 5},
        {"type": protocol.CMD_TOOL_RESULT, "id": None, "output": "[exit 0]\nok", "exit_code": 0},
    ])
    run_brain(transport, chat_fn=lambda m, t: {"role": "assistant", "content": "", "tool_calls": []})
    turns = [m for m in transport.sent if m["type"] == protocol.EV_TURN]
    assert len(turns) >= 1
    assert turns[0]["no_call"] is True and "fail_count" in turns[0]
