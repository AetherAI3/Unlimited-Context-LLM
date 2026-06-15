# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Auth — file token store + login_with_password.

Mirrors aether-code src/core/auth.ts. The token lives at
<AETHER_CONFIG_DIR>/.token (0600 best-effort), but AETHER_TOKEN in the env
ALWAYS wins on get() so a parent process (desktop/web) can inject a session
token without writing the disk store. login_with_password is exercised against
a real 127.0.0.1 http.server stub (no mocks of urllib) returning a success body
and a failure body. All disk under tmp AETHER_CONFIG_DIR.
"""
from __future__ import annotations

import json
import os
import stat
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from aether_agent import auth as agent_auth


# --- FileTokenStore: get/set/clear ----------------------------------------
def test_token_store_roundtrip_set_get_clear(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("AETHER_TOKEN", raising=False)
    store = agent_auth.FileTokenStore()
    assert store.get() is None  # nothing written yet
    store.set("sess-123")
    assert store.get() == "sess-123"
    store.clear()
    assert store.get() is None


def test_token_store_trims_and_treats_blank_as_none(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("AETHER_TOKEN", raising=False)
    (tmp_path / ".token").write_text("  spaced-token  \n", encoding="utf-8")
    assert agent_auth.FileTokenStore().get() == "spaced-token"
    (tmp_path / ".token").write_text("   \n", encoding="utf-8")
    assert agent_auth.FileTokenStore().get() is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX file mode bits not enforced on Windows")
def test_token_file_is_chmod_0600(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    store = agent_auth.FileTokenStore()
    store.set("secret")
    mode = stat.S_IMODE(os.stat(tmp_path / ".token").st_mode)
    assert mode == 0o600


def test_chmod_0600_is_best_effort_on_windows(tmp_path: Path, monkeypatch):
    # On Windows chmod is a no-op; set() must still succeed and round-trip.
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("AETHER_TOKEN", raising=False)
    store = agent_auth.FileTokenStore()
    store.set("secret")
    assert store.get() == "secret"


# --- AETHER_TOKEN env wins ------------------------------------------------
def test_env_token_overrides_disk(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    store = agent_auth.FileTokenStore()
    store.set("disk-token")
    monkeypatch.setenv("AETHER_TOKEN", "env-token")
    assert store.get() == "env-token"  # env wins
    monkeypatch.delenv("AETHER_TOKEN", raising=False)
    assert store.get() == "disk-token"  # falls back to disk


def test_env_token_works_with_no_disk_file(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("AETHER_TOKEN", "env-only")
    assert agent_auth.FileTokenStore().get() == "env-only"


# --- auth_status ----------------------------------------------------------
def test_auth_status_logged_out(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("AETHER_TOKEN", raising=False)
    store = agent_auth.FileTokenStore()
    st = agent_auth.auth_status("https://api.example.net/cloud", store)
    assert st["logged_in"] is False
    assert st["token_type"] is None
    assert st["base_url"] == "https://api.example.net/cloud"


def test_auth_status_session_and_aek_token_types(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("AETHER_TOKEN", raising=False)
    store = agent_auth.FileTokenStore()
    store.set("sess-abcdefghijklmnop")
    st = agent_auth.auth_status("https://api.example.net/cloud", store)
    assert st["logged_in"] is True
    assert st["token_type"] == "session"
    assert "sess-abcdefghijklmnop" not in st["masked"]  # masked, not raw

    store.set("aek_abcdefghijklmnop")
    st2 = agent_auth.auth_status("https://api.example.net/cloud", store)
    assert st2["token_type"] == "aek_"


# --- login_with_password against a real local stub ------------------------
class _LoginHandler(BaseHTTPRequestHandler):
    """A throwaway /auth/login stub. Echoes the request body decision:
    username 'good' -> authenticated, else a failure body."""

    def log_message(self, *args):  # silence test noise
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except ValueError:
            payload = {}
        if self.path == "/auth/login" and payload.get("username") == "good":
            body = {
                "authenticated": True,
                "session_token": "x",
                "plan": "pro",
                "commitment_hash": "abc123",
            }
            code = 200
        else:
            body = {"authenticated": False, "reason": "bad creds"}
            code = 401
        data = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def login_server():
    server = HTTPServer(("127.0.0.1", 0), _LoginHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()


def test_login_with_password_success_stores_token(tmp_path, monkeypatch, login_server):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("AETHER_TOKEN", raising=False)
    store = agent_auth.FileTokenStore()
    result = agent_auth.login_with_password(login_server, store, "good", "pw")
    assert store.get() == "x"
    assert result.get("plan") == "pro"
    assert result.get("commitment_hash") == "abc123"


def test_login_with_password_failure_raises_and_no_token(tmp_path, monkeypatch, login_server):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("AETHER_TOKEN", raising=False)
    store = agent_auth.FileTokenStore()
    with pytest.raises(RuntimeError) as exc:
        agent_auth.login_with_password(login_server, store, "bad", "pw")
    assert "bad creds" in str(exc.value)
    assert store.get() is None  # failure must not write a token


def test_login_with_password_trailing_slash_base_url(tmp_path, monkeypatch, login_server):
    # A baseUrl with a trailing slash must not produce a //auth/login path.
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("AETHER_TOKEN", raising=False)
    store = agent_auth.FileTokenStore()
    agent_auth.login_with_password(login_server + "/", store, "good", "pw")
    assert store.get() == "x"
