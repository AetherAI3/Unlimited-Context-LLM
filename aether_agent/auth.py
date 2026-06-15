# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Auth — file-backed token store + username/password login.

Mirror of aether-code ``src/core/auth.ts``. The Aether API authenticates
username/password at POST ``/auth/login`` and returns a ``session_token`` the
CLI sends as ``Authorization: Bearer <token>`` on every authed call.

The token lives at ``<config_dir>/.token`` (chmod 0600 best-effort — a no-op on
some Windows filesystems). ``AETHER_TOKEN`` in the env ALWAYS wins on ``get()``
so a parent process (desktop, web server) can inject a session token directly
without touching the on-disk store. urllib only — no third-party HTTP deps.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from aether_agent.config import config_dir

TOKEN_FILENAME = ".token"
LOGIN_PATH = "/auth/login"
_TIMEOUT = 30  # seconds


class FileTokenStore:
    """File-backed token store (0600). ``AETHER_TOKEN`` env overrides on get()."""

    def _path(self) -> Path:
        # Resolve lazily so an AETHER_CONFIG_DIR set after construction is honored.
        return config_dir() / TOKEN_FILENAME

    def get(self) -> str | None:
        """Return the active token, or ``None`` if logged out.

        Env wins: ``AETHER_TOKEN`` (if set and non-empty) overrides the disk
        store, letting an embedding host inject a session token.
        """
        env = os.environ.get("AETHER_TOKEN")
        if env and env.strip():
            return env.strip()
        path = self._path()
        if not path.exists():
            return None
        try:
            t = path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return t or None

    def set(self, token: str) -> None:
        """Persist ``token`` to ``<config_dir>/.token`` with mode 0600 (best-effort)."""
        d = config_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = self._path()
        path.write_text(token, encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except OSError:
            # chmod is a no-op on some Windows filesystems — non-fatal.
            pass

    def clear(self) -> None:
        """Remove the on-disk token (logout). No-op if absent."""
        path = self._path()
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass


def _post_json(url: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """POST a JSON body and return (status, parsed_json). Parses error bodies too
    (urllib raises HTTPError on >=400, which still carries the response body)."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:  # noqa: S310 — http/https only
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:  # 4xx/5xx still carry a JSON body
        raw = e.read()
        status = e.code
    try:
        parsed = json.loads(raw.decode("utf-8")) if raw else {}
    except (ValueError, UnicodeDecodeError):
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    return status, parsed


def login_with_password(
    base_url: str,
    store: FileTokenStore,
    username: str,
    password: str,
    license_key: str | None = None,
) -> dict[str, Any]:
    """Authenticate against ``{base_url}/auth/login`` and store the session token.

    On success (``authenticated`` truthy AND a ``session_token`` present) the
    token is written to ``store`` and a dict is returned with any ``plan`` /
    ``commitment_hash`` the server provided. On failure raises ``RuntimeError``
    with the server's ``reason`` (or an HTTP-status fallback) and writes nothing.
    """
    url = base_url.rstrip("/") + LOGIN_PATH
    payload = {"username": username, "password": password, "license_key": license_key}
    try:
        status, body = _post_json(url, payload)
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(f"login request failed: {e}") from e

    if not body.get("authenticated") or not body.get("session_token"):
        reason = body.get("reason") or f"HTTP {status}"
        raise RuntimeError(f"login failed: {reason}")

    store.set(str(body["session_token"]))
    result: dict[str, Any] = {}
    if body.get("plan") is not None:
        result["plan"] = body["plan"]
    if body.get("commitment_hash") is not None:
        result["commitment_hash"] = body["commitment_hash"]
    return result


def _mask(token: str) -> str:
    """Render a token as a short non-reversible hint (never the raw value)."""
    n = len(token)
    if n <= 4:
        return "*" * n
    return token[:4] + "…" + "*" * 4


def auth_status(base_url: str, store: FileTokenStore) -> dict[str, Any]:
    """Summarize the current auth state without exposing the raw token.

    ``token_type`` is ``'aek_'`` for API-key tokens (prefix ``aek_``),
    ``'session'`` for anything else present, or ``None`` when logged out.
    """
    token = store.get()
    if not token:
        return {
            "logged_in": False,
            "token_type": None,
            "masked": "",
            "base_url": base_url,
        }
    token_type = "aek_" if token.startswith("aek_") else "session"
    return {
        "logged_in": True,
        "token_type": token_type,
        "masked": _mask(token),
        "base_url": base_url,
    }


__all__ = [
    "FileTokenStore",
    "login_with_password",
    "auth_status",
    "TOKEN_FILENAME",
    "LOGIN_PATH",
]
