"""
Headless brain — the event-emitting agent loop for the AetherCode bridge.

Same persona + Unlimited Context working memory + autonomy/checkpoint kernel as
`agent.run_agent`, but it does NOT execute tools and does NOT render. It emits
protocol events and round-trips every tool call through the host:

    brain  --tool_call-->  host  (executes, path-guarded)
    brain  <--tool_result--  host

The host owns all rendering + all file/test/git execution, so the local and
cloud paths are identical by construction (see specs/aethercode_bridge.md).

Run over stdio:  python -m aether_agent.headless   (or `aether brain`)
The loop is transport-injected so it is unit-testable with a fake LLM + a
scripted transport (no Ollama, no subprocess).
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable, Optional, Protocol

from aether_agent import kernel, protocol, skills
from aether_agent.adapter import DEFAULT_MODEL, OllamaChat
from aether_agent.persona import SYSTEM_PROMPT

# A chat function: (messages, tools) -> assistant message dict.
ChatFn = Callable[[list[dict], list[dict]], dict[str, Any]]

# Maps a tool name to the status phase the bar should show while it runs.
_PHASE_BY_TOOL = {
    "repo_search": "scanning",
    "read_file": "scanning",
    "write_file": "anchoring",
    "run_tests": "grounding",
    "run_shell": "grounding",
    "git_commit": "paging",
}
_TOKENS_PER_GB = 233_000_000  # pool reach; mirrors statusbar / aether_context.config

# Loop-control limits (Fix 2). A no-call turn verifies; if still red after this
# many consecutive no-call turns -> stalled. Failing-count not improving across
# this many turns -> no-progress. Both terminate as `incomplete`, never `ok`.
_NO_CALL_LIMIT = 3
_STALL_LIMIT = 3

# Sentinel: _await_result hit a protocol violation and already emitted an error.
_MISMATCH = object()


class Transport(Protocol):
    """The byte seam to the host. send = emit an event; recv = read a command."""

    def send(self, message: dict[str, Any]) -> None: ...

    def recv(self) -> Optional[dict[str, Any]]: ...


class StdioTransport:
    """NDJSON over stdio: events to stdout, commands from stdin."""

    def __init__(self, out=sys.stdout, inp=sys.stdin):
        # Force utf-8 on both streams so unicode in commands/results survives a
        # Windows cp1252 pipe (events are already ASCII-escaped by protocol.encode).
        for stream in (out, inp):
            reconfig = getattr(stream, "reconfigure", None)
            if callable(reconfig):
                try:
                    reconfig(encoding="utf-8")
                except (ValueError, OSError):
                    pass
        self._out = out
        self._in = inp

    def send(self, message: dict[str, Any]) -> None:
        self._out.write(protocol.encode(message))
        self._out.flush()

    def recv(self) -> Optional[dict[str, Any]]:
        while True:
            line = self._in.readline()
            if line == "":  # EOF — host hung up
                return None
            msg = protocol.decode(line)
            if msg is not None:
                return msg


def _call_args(call: dict[str, Any]) -> tuple[str, str, dict[str, Any], bool]:
    """Extract (id, name, args, malformed) from an OpenAI tool_call.

    `malformed` is True when the model emitted a non-empty arguments string that
    is not valid JSON — the small-model failure signature. We still coerce args
    to {} and proceed (so a single bad call doesn't kill the run), but the marker
    is surfaced so a long run's emission quality is measurable (the kill-gate /
    stress test buckets these by session position)."""
    fn = call.get("function", {})
    name = fn.get("name", "")
    raw = fn.get("arguments")
    malformed = False
    try:
        args = json.loads(raw or "{}")
        if not isinstance(args, dict):
            args, malformed = {}, True
    except (json.JSONDecodeError, TypeError):
        args, malformed = {}, bool(raw and str(raw).strip())
    return call.get("id", ""), name, args, malformed


def run_brain(
    transport: Transport,
    *,
    chat_fn: Optional[ChatFn] = None,
    max_steps: int = 80,  # MAX_TURNS — a sane cap for a multi-bug task (Fix 2)
) -> int:
    """Drive one task to completion over the transport. Returns a process code.

    chat_fn is injectable for tests; in production it is an OllamaChat bound to
    the model named in the `task` command.
    """
    first = transport.recv()
    if first is None or first.get("type") != protocol.CMD_TASK:
        transport.send(protocol.error("expected a 'task' command first"))
        return 1

    task = str(first.get("text", "")).strip()
    pool_gb = int(first.get("pool_gb", 5) or 5)
    model = str(first.get("model") or DEFAULT_MODEL)
    pool_cap = pool_gb * _TOKENS_PER_GB

    # The ground-truth gate command. "" (explicitly empty) => unverifiable: the
    # brain may NEVER claim ok without a green check, so it finishes "unverified".
    raw_tc = first.get("test_cmd", "pytest -q")
    test_cmd = "" if raw_tc is None else str(raw_tc)
    verifiable = test_cmd.strip() != ""

    chat = chat_fn or OllamaChat(model=model).chat
    schema = _tool_schema()
    stuck = kernel.StuckDetector()

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    # Skill layer (procedure): pin the matching how-to so the small model knows
    # the approach, not just remembers context. Priors, not truth — the grounding
    # gate below overrides any skill whose approach fails the tests.
    loaded_skills: set[str] = set()
    _load_skills(transport, task, messages, loaded_skills)

    # Staged lifecycle. These are the stages this loop ACTUALLY performs: RECON
    # (the model's first orientation), EXECUTE (the act/test loop), then
    # SELF-REVIEW + REVEAL bracketing a clean finish. The full per-stage context
    # profiles (parse/brainstorm/write-plans with their own retrieval tiers) are
    # an engine-side pass, deferred. Stage markers give the host its pause/steer
    # boundaries (specs/neo_lite_context_lifecycle_logs_killgate.md §3-4).
    transport.send(protocol.stage("recon", "( ⚆ _ ⚆ )"))
    pool_used = 0
    steers: list[str] = []
    in_execute = False

    no_call_streak = 0
    prev_fail = float("inf")
    stalls = 0
    last_fail: Optional[int] = None  # failing count from the latest run_tests
    ok = False
    reason = ""
    result = ""

    try:
        for turn_n in range(1, max_steps + 1):
            transport.send(protocol.status("reasoning", pool_used, pool_cap))
            msg = chat(messages, schema)
            calls = msg.get("tool_calls") or []

            # --- Fix 2: a no-tool-call turn means VERIFY, never "success" -------
            if not calls:
                transport.send(protocol.turn(turn_n, 0, 0, 0, True, last_fail))
                if not verifiable:
                    ok, reason = False, "unverified"  # can't claim ok with no gate
                    result = (msg.get("content") or "(done)").strip()
                    break
                v = _run_tests(transport, test_cmd, turn_n, steers)
                if v is None:
                    transport.send(protocol.error("host closed during final verify"))
                    return 1
                vexit, vfail, vout = v
                if vfail is not None:
                    last_fail = vfail
                if vexit == 0:
                    ok, reason, result = True, "", "(done)"  # Fix 1: ground-truth ok
                    break
                no_call_streak += 1
                if no_call_streak >= _NO_CALL_LIMIT:
                    ok, reason = False, "stalled"
                    result = f"(incomplete — {vfail if vfail is not None else '?'} failing)"
                    break
                # nudge with reality (Fix 3 reinforcement: read the SOURCE)
                messages.append({
                    "role": "user",
                    "content": (
                        f"{vfail if vfail is not None else 'Some'} tests still failing:\n{_tail(vout)}\n"
                        "Read the SOURCE modules the failing tests import (the implementation, "
                        "not the test files) and edit them. Do not stop until run_tests exits 0."
                    ),
                })
                continue

            no_call_streak = 0
            if not in_execute:
                transport.send(protocol.stage("execute", "(ง'̀-'́)ง"))
                in_execute = True

            messages.append(
                {"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls}
            )
            malformed_n = 0
            invented_n = 0
            for call in calls:
                call_id, name, args, malformed = _call_args(call)
                if malformed:
                    malformed_n += 1
                    transport.send(protocol.monologue(f"malformed-args: {name}", depth=1))
                elif name and name not in protocol.TOOLS:
                    invented_n += 1
                    transport.send(protocol.monologue(f"invented-tool: {name}", depth=1))
                phase = _PHASE_BY_TOOL.get(name, "reasoning")
                transport.send(protocol.status(phase, pool_used, pool_cap))
                transport.send(protocol.tool_call(call_id, name, args))

                reply = _await_result(transport, call_id, steers)
                if reply is _MISMATCH:
                    return 1  # error already emitted by _await_result
                if reply is None:
                    transport.send(protocol.error("host closed mid tool_call"))
                    return 1
                assert isinstance(reply, dict)
                output = str(reply.get("output", ""))
                exit_code = int(reply.get("exit_code", 0) or 0)

                pool_used += max(1, len(output) // 4)
                messages.append({"role": "tool", "tool_call_id": call_id, "content": output})
                _drain_steers(steers, messages)

                if name == "run_tests":
                    last_fail = 0 if kernel.tests_pass(output) else kernel.parse_fail_count(output)
                    _grounding_gate(transport, output, exit_code, turn_n, stuck, messages, loaded_skills)

            # per-turn diagnostics (the §8 emission curve feed)
            transport.send(protocol.turn(turn_n, len(calls), malformed_n, invented_n, False, last_fail))

            # --- no-progress breaker: failing count not improving -> stall ------
            if last_fail is not None:
                if last_fail >= prev_fail:
                    stalls += 1
                    if stalls >= _STALL_LIMIT:
                        ok, reason = False, "no-progress"
                        result = f"(incomplete — {last_fail} failing)"
                        break
                else:
                    stalls = 0
                prev_fail = last_fail
        else:
            # max turns hit — derive ok from a real final run, never assume.
            if verifiable:
                v = _run_tests(transport, test_cmd, max_steps + 1, steers)
                if v is not None:
                    vexit, vfail, _ = v
                    ok = vexit == 0
                    if vfail is not None:
                        last_fail = vfail
            reason = "" if ok else "max-turns"
            result = "(done)" if ok else f"(incomplete — {last_fail if last_fail is not None else '?'} failing)"

        transport.send(protocol.stage("self-review", "(¬_¬\")→[•‿•]"))
        transport.send(protocol.stage("reveal", "ᕙ(`▽`)ᕗ"))
        remaining = 0 if ok else (last_fail if last_fail is not None else 0)
        transport.send(protocol.done(ok, result or ("(done)" if ok else "(incomplete)"), remaining, reason))
        return 0 if ok else 1
    except RuntimeError as e:  # adapter raises these with a clear hint
        transport.send(protocol.error(str(e)))
        return 1


def _tail(text: str, limit: int = 1500) -> str:
    """Last `limit` chars of test output — enough signal for the model to act on."""
    return text[-limit:] if len(text) > limit else text


def _run_tests(
    transport: Transport, test_cmd: str, turn_n: int, steers: list[str]
) -> Optional[tuple[int, Optional[int], str]]:
    """Round-trip a run_tests through the host. Returns (exit_code, fail_count, output)
    or None if the host closed. This is the brain's ground-truth verify."""
    cid = f"verify-{turn_n}"
    transport.send(protocol.status("grounding", 0, 0))
    transport.send(protocol.tool_call(cid, "run_tests", {"command": test_cmd}))
    reply = _await_result(transport, cid, steers)
    if not isinstance(reply, dict):
        return None
    output = str(reply.get("output", ""))
    exit_code = int(reply.get("exit_code", 0) or 0)
    return exit_code, kernel.parse_fail_count(output), output


def _drain_steers(steers: list[str], messages: list[dict]) -> None:
    """Inject any pending /steer notes into the thread as user guidance."""
    while steers:
        note = steers.pop(0)
        messages.append({"role": "user", "content": f"User steer (high priority): {note}"})


def _await_result(transport: Transport, call_id: str, steers: list[str]):
    """Read commands until the matching tool_result arrives; apply controls.
    Returns the result dict, None on EOF, or `_MISMATCH` on a wrong-id reply.

    Tool calls are strictly correlated: the brain emits ONE tool_call and blocks
    here until the host replies, so the host replies in order. A tool_result that
    carries the WRONG id is therefore a protocol violation, not something to
    skip — skipping would mis-pair results to calls (a brutal intermittent bug).
    We fail loud instead: emit an error and abort.

    A `steer` control that arrives while waiting is collected into `steers` (the
    caller drains it into the message thread) and echoed as monologue. `pause` is
    realized by the host simply withholding its reply; `resume` is the reply
    arriving — so they need no explicit handling here.
    """
    while True:
        cmd = transport.recv()
        if cmd is None:
            return None
        kind = cmd.get("type")
        if kind == protocol.CMD_TOOL_RESULT:
            got = str(cmd.get("id", ""))
            if got == call_id:
                return cmd
            transport.send(protocol.error(f"tool_result id mismatch: expected {call_id!r} got {got!r}"))
            return _MISMATCH
        if kind == protocol.CMD_CONTROL:
            note = str(cmd.get("note", "")).strip()
            if cmd.get("action") == "steer" and note:
                steers.append(note)
                transport.send(protocol.monologue(f"steer: {note}", depth=1))
        # ignore anything else (unknown) and keep waiting for our result


def _load_skills(
    transport: Transport, focus: str, messages: list[dict], loaded: set[str]
) -> None:
    """Match procedure packets against the focus and pin the new ones (reserved
    budget). Emits a skill event per pin so the host renders it."""
    matched = skills.match_skills(focus, exclude=frozenset(loaded))
    for sk in matched:
        loaded.add(sk.name)
        messages.append({"role": "system", "content": sk.as_prompt()})
        transport.send(protocol.skill(sk.name, reason="trigger matched"))


def _grounding_gate(
    transport: Transport,
    output: str,
    exit_code: int,
    step: int,
    stuck: kernel.StuckDetector,
    messages: list[dict],
    loaded_skills: set[str],
) -> None:
    """Ground-truth gate: green -> ask the host to checkpoint; stalled -> nudge.

    On a stall, re-match skills against the FAILING output — the right moment to
    inject a debug procedure the model didn't get from the task text alone.
    """
    green = exit_code == 0 or kernel.tests_pass(output)
    if green:
        transport.send(protocol.monologue("tests green — checkpointing", depth=1))
        sha = _checkpoint(transport, f"aether: step {step} green")
        if sha:
            transport.send(protocol.checkpoint(sha))
        return
    if stuck.update(kernel.parse_fail_count(output)):
        transport.send(protocol.monologue("no test progress — changing strategy", depth=1))
        _load_skills(transport, output, messages, loaded_skills)  # procedure at the moment of need
        messages.append(
            {"role": "user", "content": "You are stuck — change strategy; do not repeat the failing approach."}
        )


def _checkpoint(transport: Transport, message: str) -> str:
    """Round-trip a git_commit through the host; return the short sha if any."""
    cid = f"ckpt-{abs(hash(message)) % 100000}"
    transport.send(protocol.tool_call(cid, "git_commit", {"message": message}))
    reply = _await_result(transport, cid, [])  # checkpoint round-trip ignores steers
    if not isinstance(reply, dict):
        return ""  # None (EOF) or _MISMATCH — no sha to report
    out = str(reply.get("output", ""))
    for tok in out.replace("\n", " ").split():
        if len(tok) >= 7 and all(c in "0123456789abcdef" for c in tok.lower()):
            return tok[:10]
    return ""


def _tool_schema() -> list[dict]:
    # Lazy import keeps protocol/headless free of the engine's numpy chain for
    # pure-codec tests; tools.tool_schema is dependency-light regardless.
    from aether_agent.tools import tool_schema

    return tool_schema()


def main(argv: Optional[list[str]] = None) -> int:
    return run_brain(StdioTransport())


if __name__ == "__main__":
    raise SystemExit(main())
