# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Transport — ApiClient over the Aether API.

Mirrors aether-code src/core/transport.ts. Route constants are pinned (lockstep
with the TS host). post_json/get_json/stream are exercised against a real
127.0.0.1 http.server stub (no urllib mocking): stream() decodes SSE 'data:'
JSON frames; when the server fails soft by returning application/json instead of
text/event-stream, stream() raises StreamUnavailable. Bearer comes from the
token store. All disk under tmp AETHER_CONFIG_DIR.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from aether_agent import auth as agent_auth
from aether_agent import transport as agent_transport


# --- route constants (lockstep with transport.ts) -------------------------
def test_route_constants_match_ts_mirror():
    assert agent_transport.CHAT_STREAM_PATH == "/agent/chat/stream"
    assert agent_transport.CHAT_PATH == "/agent/chat"
    assert agent_transport.LOGIN_PATH == "/auth/login"
    assert agent_transport.MODELS_PATH == "/models"
    assert agent_transport.AGENTS_PATH == "/agents"
    assert agent_transport.AUDIT_TRAIL_PATH == "/audit/trail/live"


# --- the stub server ------------------------------------------------------
class _ApiHandler(BaseHTTPRequestHandler):
    """A throwaway API stub covering post_json, get_json, and stream."""

    def log_message(self, *args):
        return

    def _write(self, code, content_type, body_bytes):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body_bytes)))
        self.end_headers()
        self.wfile.write(body_bytes)

    def do_GET(self):
        if self.path == "/models":
            self._write(
                200, "application/json", json.dumps({"models": ["a", "b"]}).encode("utf-8")
            )
        elif self.path == "/needauth":
            auth = self.headers.get("Authorization", "")
            self._write(200, "application/json", json.dumps({"auth": auth}).encode("utf-8"))
        else:
            self._write(404, "application/json", b"{}")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        payload = json.loads(raw.decode("utf-8")) if raw else {}

        if self.path == "/agent/chat":
            self._write(
                200, "application/json", json.dumps({"ok": True, "echo": payload}).encode("utf-8")
            )
        elif self.path == "/agent/chat/stream":
            # Real SSE: two data: frames then a terminal frame.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            frames = [
                {"type": "delta", "text": "hello"},
                {"type": "delta", "text": " world"},
                {"type": "done", "ok": True},
            ]
            for f in frames:
                self.wfile.write(b"data: " + json.dumps(f).encode("utf-8") + b"\n\n")
            self.wfile.flush()
        elif self.path == "/agent/chat/stream-failsoft":
            # Fail-soft: server returns plain JSON instead of an SSE body.
            self._write(
                200, "application/json", json.dumps({"stream": False}).encode("utf-8")
            )
        else:
            self._write(404, "application/json", b"{}")


@pytest.fixture
def api_server():
    server = HTTPServer(("127.0.0.1", 0), _ApiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()


def _client(base_url, tmp_path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("AETHER_TOKEN", raising=False)
    store = agent_auth.FileTokenStore()
    return agent_transport.ApiClient(base_url, store), store


# --- post_json / get_json -------------------------------------------------
def test_post_json_roundtrips(tmp_path, monkeypatch, api_server):
    client, _ = _client(api_server, tmp_path, monkeypatch)
    res = client.post_json("/agent/chat", {"text": "hi"})
    assert res["ok"] is True
    assert res["echo"] == {"text": "hi"}


def test_get_json_roundtrips(tmp_path, monkeypatch, api_server):
    client, _ = _client(api_server, tmp_path, monkeypatch)
    res = client.get_json("/models")
    assert res["models"] == ["a", "b"]


def test_bearer_header_sent_from_token_store(tmp_path, monkeypatch, api_server):
    client, store = _client(api_server, tmp_path, monkeypatch)
    store.set("tok-xyz")
    res = client.get_json("/needauth")
    assert res["auth"] == "Bearer tok-xyz"


def test_no_bearer_header_when_logged_out(tmp_path, monkeypatch, api_server):
    client, _ = _client(api_server, tmp_path, monkeypatch)
    res = client.get_json("/needauth")
    assert res["auth"] == ""  # no Authorization header


# --- stream: SSE decode ---------------------------------------------------
def test_stream_decodes_sse_data_frames(tmp_path, monkeypatch, api_server):
    client, _ = _client(api_server, tmp_path, monkeypatch)
    frames = list(client.stream("/agent/chat/stream", {"text": "hi"}))
    assert frames[0] == {"type": "delta", "text": "hello"}
    assert frames[1] == {"type": "delta", "text": " world"}
    assert frames[-1] == {"type": "done", "ok": True}


def test_stream_fail_soft_raises_stream_unavailable(tmp_path, monkeypatch, api_server):
    client, _ = _client(api_server, tmp_path, monkeypatch)
    with pytest.raises(agent_transport.StreamUnavailable):
        list(client.stream("/agent/chat/stream-failsoft", {"text": "hi"}))


def test_trailing_slash_base_url_does_not_double_slash(tmp_path, monkeypatch, api_server):
    client, _ = _client(api_server + "/", tmp_path, monkeypatch)
    res = client.get_json("/models")
    assert res["models"] == ["a", "b"]
