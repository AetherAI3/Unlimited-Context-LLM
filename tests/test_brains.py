"""Brains tests — the event-yielding brain interface (U3).

Pure: no Ollama, no real network, no subprocess. A stub chat object stands in
for OllamaChat (returns a scripted tool_call then a final answer); a fake
ApiClient stands in for the cloud transport (yields scripted SSE frames). The
LocalBrain runs the real Tools on a scratch file in a tmp cwd.
"""

from __future__ import annotations

from typing import Any, Iterator

import pytest

from aether_agent import brains
from aether_agent.brains import CloudBrain, LocalBrain, select_brain


# --- stubs -----------------------------------------------------------------
class StubChat:
    """A scripted OllamaChat: returns each queued assistant message in turn.

    Each message is a dict ({content, tool_calls?}) exactly like the adapter
    returns. The last queued message has no tool_calls -> the loop finishes.
    """

    def __init__(self, scripted: list[dict[str, Any]]):
        self._scripted = list(scripted)
        self.calls: list[list[dict]] = []
        self.last_tools: Any = None

    def chat(self, messages: list[dict], tools: Any = None, **kw: Any) -> dict[str, Any]:
        self.calls.append(list(messages))
        self.last_tools = tools
        if self._scripted:
            return self._scripted.pop(0)
        return {"content": "(done)", "tool_calls": []}


def _tool_call(call_id: str, name: str, arguments: str) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


class FakeApiClient:
    """Fake transport.ApiClient: stream() yields scripted SSE frames."""

    def __init__(self, frames: list[dict], *, raise_on_stream: Exception | None = None,
                 post_reply: dict | None = None):
        self._frames = list(frames)
        self._raise = raise_on_stream
        self._post_reply = post_reply or {}
        self.stream_paths: list[str] = []
        self.post_paths: list[str] = []

    def stream(self, path: str, body: dict) -> Iterator[dict]:
        self.stream_paths.append(path)
        if self._raise is not None:
            raise self._raise
        yield from self._frames

    def post_json(self, path: str, body: dict) -> dict:
        self.post_paths.append(path)
        return self._post_reply


# --- select_brain policy table ---------------------------------------------
@pytest.mark.parametrize(
    "authed,backend,expected",
    [
        (False, "local", LocalBrain),
        (True, "local", LocalBrain),
        (False, "cloud", CloudBrain),
        (True, "cloud", CloudBrain),
        (False, "auto", LocalBrain),   # not authed -> local
        (True, "auto", CloudBrain),    # authed -> cloud
    ],
)
def test_select_brain_policy(authed, backend, expected):
    api = FakeApiClient([])
    b = select_brain(authed=authed, backend=backend, api=api, model="qwen3-coder:30b")
    assert isinstance(b, expected)


def test_select_brain_rejects_unknown_backend():
    with pytest.raises(ValueError):
        select_brain(authed=True, backend="nonsense", api=FakeApiClient([]), model="m")


# --- LocalBrain emits the event sequence -----------------------------------
def test_local_brain_emits_tool_call_result_and_done(tmp_path):
    # Arrange: a chat that first asks to read a scratch file, then answers.
    scratch = tmp_path / "note.txt"
    scratch.write_text("hello world", encoding="utf-8")
    chat = StubChat([
        {"content": "let me read it", "tool_calls": [_tool_call("c1", "read_file", '{"path": "note.txt"}')]},
        {"content": "the file says hello", "tool_calls": []},
    ])
    brain = LocalBrain(model="qwen3-coder:30b", cwd=str(tmp_path), llm=chat)

    # Act
    events = list(brain.run("read the note"))
    kinds = [e["type"] for e in events]

    # Assert: tool_call -> tool_result -> done, in that order.
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    assert "done" in kinds
    assert kinds.index("tool_call") < kinds.index("tool_result") < kinds.index("done")

    tc = next(e for e in events if e["type"] == "tool_call")
    assert tc["name"] == "read_file"
    assert tc["args"] == {"path": "note.txt"}

    tr = next(e for e in events if e["type"] == "tool_result")
    assert tr["name"] == "read_file"
    assert "hello world" in tr["output"]  # the real Tools read the scratch file

    done = next(e for e in events if e["type"] == "done")
    assert "hello" in done["text"]


def test_local_brain_supports_full_8_tool_schema():
    # The schema the local brain advertises must be the canonical 8 tools.
    from aether_agent import protocol

    chat = StubChat([{"content": "done", "tool_calls": []}])
    brain = LocalBrain(model="m", cwd=".", llm=chat)
    list(brain.run("noop"))
    # the brain offers the schema on every chat call; the canonical tuple is 8.
    assert len(protocol.TOOLS) == 8
    offered = {t["function"]["name"] for t in (chat.last_tools or [])}
    assert set(protocol.TOOLS) == offered


# --- CloudBrain maps SSE frames to events ----------------------------------
def test_cloud_brain_maps_frames_to_events():
    frames = [
        {"type": "reasoning", "text": "thinking..."},
        {"type": "delta", "text": "answer chunk"},
        {"type": "done"},
    ]
    api = FakeApiClient(frames)
    brain = CloudBrain(api=api, model="cloud-model")

    events = list(brain.run("hi"))
    kinds = [e["type"] for e in events]

    assert "monologue" in kinds        # reasoning + delta -> monologue
    assert kinds[-1] == "done"         # terminal done emitted by the pump
    assert api.stream_paths == ["/agent/chat/stream"]

    monologues = [e for e in events if e["type"] == "monologue"]
    texts = [m["text"] for m in monologues]
    assert "thinking..." in texts
    assert "answer chunk" in texts


def test_cloud_brain_error_frame_maps_to_error():
    api = FakeApiClient([{"type": "error", "msg": "boom"}])
    brain = CloudBrain(api=api, model="m")
    events = list(brain.run("hi"))
    errs = [e for e in events if e["type"] == "error"]
    assert errs and errs[0]["msg"] == "boom"


def test_cloud_brain_fail_soft_falls_back_to_post_json():
    from aether_agent.transport import StreamUnavailable

    api = FakeApiClient(
        [],
        raise_on_stream=StreamUnavailable({"stream": False}),
        post_reply={"response": "non-streamed answer"},
    )
    brain = CloudBrain(api=api, model="m")
    events = list(brain.run("hi"))

    assert api.post_paths == ["/agent/chat"]  # fell back to the non-streaming route
    kinds = [e["type"] for e in events]
    assert kinds[-1] == "done"
    done = next(e for e in events if e["type"] == "done")
    assert "non-streamed answer" in done["text"]
