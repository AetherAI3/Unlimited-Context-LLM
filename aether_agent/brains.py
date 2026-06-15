"""
Brains — the pluggable, event-yielding decision source behind one interface.
Mirror of aether-code ``src/core/brain*.ts``.

A Brain ``run(task)`` yields lightweight event dicts a REPL / bridge renders:

    {'type': 'monologue',    'text': ...}
    {'type': 'tool_call',    'name': ..., 'args': {...}}
    {'type': 'tool_result',  'name': ..., 'output': ...}
    {'type': 'done',         'text': ...}
    {'type': 'error',        'msg': ...}

Two implementations, one vocabulary, so local and cloud render identically:

- ``LocalBrain`` runs the agentic chat+tool loop on Ollama + the real Tools
  (the canonical 8-tool schema), reusing ``agent.run_agent_events`` so the loop
  lives in exactly one place. It executes tools in-process (this is the REPL
  surface — not the host-round-tripped bridge in ``headless.py``).
- ``CloudBrain`` surfaces the AetherCloud SSE stream
  (``transport.ApiClient.stream(CHAT_STREAM_PATH, ...)``) through the SAME event
  vocabulary, mapping reasoning/delta/done/error frames; it fails soft to the
  non-streaming ``CHAT_PATH`` when the server returns JSON instead of an SSE body.

``select_brain`` is the policy seam: ``local`` -> LocalBrain, ``cloud`` ->
CloudBrain, ``auto`` -> CloudBrain when authed else LocalBrain.
"""

from __future__ import annotations

from typing import Any, Iterator, Protocol, runtime_checkable

from aether_agent.adapter import DEFAULT_MODEL, OllamaChat
from aether_agent.agent import run_agent_events
from aether_agent.tools import Tools, tool_schema
from aether_agent.transport import CHAT_PATH, CHAT_STREAM_PATH, StreamUnavailable


@runtime_checkable
class Brain(Protocol):
    """The decision source. ``run`` yields render-ready event dicts until a
    terminal ``done`` (or ``error``) event."""

    def run(self, task: str) -> Iterator[dict[str, Any]]: ...


class LocalBrain:
    """Local brain — agentic chat+tool loop on Ollama + the real Tools.

    Reuses ``agent.run_agent_events`` (the one shared loop) so behaviour stays in
    lockstep with ``run_agent``. The chat loop does not git-checkpoint (that is
    the kernel's job inside a full ``run_agent`` run); a REPL chat turn just
    reads/writes/searches and answers.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        cwd: str = ".",
        pool_gb: int = 5,
        max_steps: int = 40,
        test_cmd: str = "pytest -q",
        llm: Any | None = None,
        tools: Tools | None = None,
    ) -> None:
        self.model = model
        self.cwd = cwd
        self.pool_gb = pool_gb
        self.max_steps = max_steps
        # Injectable for tests; OllamaChat / Tools in production.
        self._llm = llm if llm is not None else OllamaChat(model=model)
        self._tools = tools if tools is not None else Tools(cwd, test_cmd=test_cmd)

    def run(self, task: str) -> Iterator[dict[str, Any]]:
        # The canonical 8-tool schema is advertised on every chat call.
        yield from run_agent_events(
            task,
            llm=self._llm,
            tools=self._tools,
            cwd=self.cwd,
            pool_gb=self.pool_gb,
            max_steps=self.max_steps,
            sess=None,           # REPL chat loop: no Unlimited-Context Session memory
            schema=tool_schema(),
            git_checkpoint=False,  # chat loop need not checkpoint
            verify_finish=False,   # a chat turn is not a test-gated build run
        )


class CloudBrain:
    """Cloud brain — the AetherCloud SSE stream mapped to the bridge vocabulary.

    Honest boundary (today's server contract): the universal stream runs its
    tools server-side and is one-way, so it does NOT emit ``tool_call`` frames or
    accept an upstream ``tool_result``. CloudBrain maps the frames that exist
    (reasoning/delta/done/error) and fails soft to the non-streaming route.
    """

    def __init__(self, *, api: Any, model: str = DEFAULT_MODEL) -> None:
        self.api = api
        self.model = model

    def run(self, task: str) -> Iterator[dict[str, Any]]:
        body: dict[str, Any] = {"prompt": task}
        if self.model:
            body["model"] = self.model
        try:
            for frame in self.api.stream(CHAT_STREAM_PATH, body):
                ev = _map_frame(frame)
                if ev is not None:
                    yield ev
            yield {"type": "done", "text": ""}  # pump emits its own terminal done
            return
        except StreamUnavailable:
            # Fail soft: the server returned JSON (contract {"stream": false}).
            yield from self._fallback(body)
        except RuntimeError as e:  # transport/adapter raises these with a clear hint
            yield {"type": "error", "msg": str(e)}

    def _fallback(self, body: dict[str, Any]) -> Iterator[dict[str, Any]]:
        try:
            reply = self.api.post_json(CHAT_PATH, body) or {}
        except Exception as e:  # noqa: BLE001 — surface as an error event, never crash
            yield {"type": "error", "msg": str(e)}
            return
        text = str(reply.get("response", "") or "")
        if text:
            yield {"type": "monologue", "text": text}
        yield {"type": "done", "text": text}


def _map_frame(frame: dict[str, Any]) -> dict[str, Any] | None:
    """Map a universal SSE frame onto the brain event vocabulary (None = ignore).

    Mirrors aether-code ``brain_cloud.ts``: reasoning -> nested monologue, delta
    -> monologue, error -> error. ``done`` is swallowed here — the pump emits its
    own terminal ``done`` after the loop. Everything else (open/ping/usage/...) is
    not part of the agent view.
    """
    ftype = frame.get("type")
    if ftype == "reasoning":
        return {"type": "monologue", "text": str(frame.get("text", "")), "depth": 1}
    if ftype == "delta":
        return {"type": "monologue", "text": str(frame.get("text", "")), "depth": 0}
    if ftype == "error":
        return {"type": "error", "msg": str(frame.get("msg", "task failed"))}
    return None


def select_brain(*, authed: bool, backend: str, api: Any, model: str = DEFAULT_MODEL) -> Brain:
    """Pick a Brain by policy:

    - ``local`` -> LocalBrain (always)
    - ``cloud`` -> CloudBrain (always)
    - ``auto``  -> CloudBrain when authed, else LocalBrain

    Raises ``ValueError`` on an unknown backend so a typo fails loud.
    """
    b = (backend or "").strip().lower()
    if b == "local":
        return LocalBrain(model=model)
    if b == "cloud":
        return CloudBrain(api=api, model=model)
    if b == "auto":
        return CloudBrain(api=api, model=model) if authed else LocalBrain(model=model)
    raise ValueError(f"unknown backend {backend!r} (expected 'local', 'cloud', or 'auto')")


__all__ = ["Brain", "LocalBrain", "CloudBrain", "select_brain"]
