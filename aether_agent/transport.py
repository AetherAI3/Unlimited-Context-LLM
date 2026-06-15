# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Transport — ApiClient, the ONLY network surface to the Aether API.

Mirror of aether-code ``src/core/transport.ts``. Talks to the public Aether API
front door; all access control and usage enforcement happen server-side. Route
constants live here so a path changes in exactly one place and stays in lockstep
with the TS host.

``stream`` decodes Server-Sent-Events ``data:`` lines into JSON frames. The
backend may *fail soft* by returning ``application/json`` (e.g. ``{"stream":
false}``) instead of an SSE body when it can't/shouldn't stream — that is raised
as ``StreamUnavailable`` so the caller can fall back to the non-streaming
``/agent/chat`` route. urllib only — no third-party HTTP deps.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Iterator

# --- Aether API routes (lockstep with transport.ts) -----------------------
CHAT_STREAM_PATH = "/agent/chat/stream"   # standard chat SSE
CHAT_PATH = "/agent/chat"                 # non-streaming fail-soft fallback
LOGIN_PATH = "/auth/login"               # session_token via username/password
MODELS_PATH = "/models"
AGENTS_PATH = "/agents"
AUDIT_TRAIL_PATH = "/audit/trail/live"   # entries carry commitment_hash

_TIMEOUT = 120  # seconds — streams can run long


class StreamUnavailable(RuntimeError):
    """Raised when the server fails soft (returns JSON, not an SSE body).

    Carries the parsed fail-soft body (if any) so the caller can decide how to
    fall back to the non-streaming chat route.
    """

    def __init__(self, body: Any = None) -> None:
        super().__init__("stream unavailable (server returned JSON, not an event stream)")
        self.body = body


class ApiClient:
    """Thin HTTP client over the Aether API. Bearer token is pulled from a token
    store (any object exposing ``get() -> str | None``) on every request."""

    def __init__(self, base_url: str, store: Any) -> None:
        self._base_url = base_url.rstrip("/")
        self._store = store

    def _url(self, path: str) -> str:
        return self._base_url + path

    def _auth_headers(self) -> dict[str, str]:
        token = self._store.get() if self._store is not None else None
        return {"Authorization": f"Bearer {token}"} if token else {}

    def post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST a JSON body and return the parsed JSON response."""
        data = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **self._auth_headers(),
        }
        req = urllib.request.Request(self._url(path), data=data, method="POST", headers=headers)
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:  # noqa: S310 — http/https only
            raw = resp.read()
        return self._parse_json(raw)

    def get_json(self, path: str) -> dict[str, Any]:
        """GET and return the parsed JSON response."""
        headers = {"Accept": "application/json", **self._auth_headers()}
        req = urllib.request.Request(self._url(path), method="GET", headers=headers)
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:  # noqa: S310 — http/https only
            raw = resp.read()
        return self._parse_json(raw)

    def stream(self, path: str, body: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """POST a body and yield decoded SSE ``data:`` JSON frames.

        Fail-soft: if the response content-type is ``application/json`` the
        server declined to stream — raise ``StreamUnavailable`` carrying the
        parsed body so the caller can fall back to ``/agent/chat``.
        """
        data = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            **self._auth_headers(),
        }
        req = urllib.request.Request(self._url(path), data=data, method="POST", headers=headers)
        resp = urllib.request.urlopen(req, timeout=_TIMEOUT)  # noqa: S310 — http/https only
        try:
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if ctype.startswith("application/json"):
                # The server failed soft — surface the body, do not stream.
                raw = resp.read()
                raise StreamUnavailable(self._parse_json(raw, default=None))
            yield from _decode_sse(resp)
        finally:
            resp.close()

    @staticmethod
    def _parse_json(raw: bytes, default: Any = None) -> Any:
        if not raw:
            return {} if default is None else default
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {} if default is None else default


def _decode_sse(resp) -> Iterator[dict[str, Any]]:
    """Yield JSON objects parsed from each ``data:`` line of an SSE response.

    Blank lines (event separators) and non-``data:`` lines are skipped; a
    ``data: [DONE]`` sentinel ends the stream. Malformed JSON on a single frame
    is skipped rather than aborting the whole stream.
    """
    for raw_line in resp:
        try:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        except AttributeError:
            line = str(raw_line).rstrip("\r\n")
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload:
            continue
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except ValueError:
            continue
        if isinstance(obj, dict):
            yield obj


__all__ = [
    "CHAT_STREAM_PATH",
    "CHAT_PATH",
    "LOGIN_PATH",
    "MODELS_PATH",
    "AGENTS_PATH",
    "AUDIT_TRAIL_PATH",
    "StreamUnavailable",
    "ApiClient",
]
