# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI - Brandon Barrante
# SPDX-License-Identifier: Apache-2.0
"""Integration test for the Ollama wrapper against a local stub HTTP server.

Exercises the REAL ``urllib`` HTTP path of :class:`aether_context.local_llm.OllamaLLM` — no
mocks, no real Ollama daemon. A tiny stub binds to ``127.0.0.1`` (loopback only, so it is
hermetic and needs no external network) and speaks the two endpoints the wrapper uses:

* ``POST /api/show``  → ``model_info`` carrying a ``context_length`` (probed for the window);
* ``POST /api/chat``  → newline-delimited JSON, each line a ``{"message": {"content": ...}}``
  chunk (the streaming path the pager overlaps).

Then it drives a full :class:`~aether_context.session.Session` through the wrapper end to end.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from aether_context.local_llm import OllamaLLM
from aether_context.session import Session

# The chunks the stub streams back (>=2 so generate() genuinely yields more than once).
_CHUNKS = ["Build ", "the ", "auth ", "module ", "carefully."]
_REPLY = "".join(_CHUNKS)
_STUB_WINDOW = 4096


class _StubOllamaHandler(BaseHTTPRequestHandler):
    """Minimal faithful stub of the Ollama HTTP API the wrapper calls."""

    def log_message(self, *_args: object) -> None:  # silence test-server logging
        pass

    def _drain_body(self) -> None:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length:
            self.rfile.read(length)

    def _send_json(self, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._drain_body()
        if self.path == "/api/show":
            self._send_json({"model_info": {"general.context_length": _STUB_WINDOW}})
        elif self.path == "/api/chat":
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()  # HTTP/1.0 close-delimited stream
            for chunk in _CHUNKS:
                line = json.dumps({"message": {"role": "assistant", "content": chunk}, "done": False})
                self.wfile.write((line + "\n").encode("utf-8"))
                self.wfile.flush()
            done = json.dumps({"message": {"content": ""}, "done": True})
            self.wfile.write((done + "\n").encode("utf-8"))
            self.wfile.flush()
        elif self.path == "/api/pull":
            self._send_json({"status": "success"})
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/api/tags":
            self._send_json({"models": [{"name": "stub"}]})
        else:
            self.send_response(404)
            self.end_headers()


@pytest.fixture
def stub_ollama():
    """Run the stub on a free loopback port; yield its base URL; tear down after."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubOllamaHandler)
    host = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield host
    finally:
        server.shutdown()
        server.server_close()


def test_ollama_context_window_probe(stub_ollama):
    # The wrapper POSTs /api/show and parses model_info.*context_length.
    m = OllamaLLM("stub", host=stub_ollama)
    assert m.context_window == _STUB_WINDOW


def test_ollama_generate_streams_chunks(stub_ollama):
    # The wrapper POSTs /api/chat (stream=true) and yields message.content per NDJSON line.
    m = OllamaLLM("stub", host=stub_ollama)
    chunks = list(m.generate("write the auth module", system="be terse"))
    assert len(chunks) >= 2  # genuinely streamed, not one blob
    assert "".join(chunks) == _REPLY


def test_ollama_count_tokens_positive(stub_ollama):
    m = OllamaLLM("stub", host=stub_ollama)
    assert m.count_tokens("hello world this is a test") > 0


def test_ollama_unreachable_host_is_fail_soft():
    # A down daemon must degrade the window to the fallback, never raise from the probe.
    m = OllamaLLM("stub", host="http://127.0.0.1:1")  # nothing listens on port 1
    assert m.context_window > 0  # fell back, did not crash


def test_session_runs_end_to_end_through_ollama_wrapper(stub_ollama, tmp_pool_dir):
    # The headline: a real Session, driven over the wrapper's HTTP path, returns the reply.
    m = OllamaLLM("stub", host=stub_ollama)
    s = Session(model=m, pool_gb=5, pool_dir=str(tmp_pool_dir))
    try:
        result = s.run("build a long thing with durable facts to remember")
        assert result.text  # got text back through the engine
        assert _REPLY in result.text  # it is the model's streamed reply
    finally:
        s.close()
