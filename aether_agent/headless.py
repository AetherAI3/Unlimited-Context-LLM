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
    max_steps: int = 40,
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

    try:
        for step in range(1, max_steps + 1):
            transport.send(protocol.status("reasoning", pool_used, pool_cap))
            msg = chat(messages, schema)
            calls = msg.get("tool_calls") or []
            if not calls:
                transport.send(protocol.stage("self-review", "(¬_¬\")→[•‿•]"))
                transport.send(protocol.stage("reveal", "ᕙ(`▽`)ᕗ"))
                transport.send(protocol.done(True, (msg.get("content") or "(done)").strip()))
                return 0

            if not in_execute:
                transport.send(protocol.stage("execute", "(ง'̀-'́)ง"))
                in_execute = True

            messages.append(
                {"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls}
            )
            for call in calls:
                call_id, name, args, malformed = _call_args(call)
                if malformed:
                    # countable signal: the model produced un-parseable tool args
                    transport.send(protocol.monologue(f"malformed-args: {name}", depth=1))
                elif name and name not in protocol.TOOLS:
                    # countable signal: the model invented a tool that doesn't exist
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

                # Tool output is encoded into the working-memory thread (the
                # host holds the real pool; the brain tracks the conversation).
                pool_used += max(1, len(output) // 4)
                messages.append(
                    {"role": "tool", "tool_call_id": call_id, "content": output}
                )
                # Any /steer notes that arrived while waiting are injected as
                # high-priority user guidance (never faded — the spec's user slice).
                _drain_steers(steers, messages)

                if name == "run_tests":
                    _grounding_gate(transport, output, exit_code, step, stuck, messages, loaded_skills)

        transport.send(protocol.done(False, "max steps reached without a clean finish"))
        return 1
    except RuntimeError as e:  # adapter raises these with a clear hint
        transport.send(protocol.error(str(e)))
        return 1


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
