# aether_agent/ollama_ctl.py
# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Ollama lifecycle control — the terminal's managed view of the local daemon.

Talks to the Ollama HTTP API (``/api/tags``, ``/api/pull`` stream, ``/api/ps``,
``/api/delete``) and supervises a ``serve`` subprocess we may own. Every IO seam
(HTTP GET, HTTP streaming POST, ``Popen``, ``which``, ``sleep``, download, run)
is injectable so the whole surface is unit-testable offline. ``install`` (Task 4)
is the only path that downloads; it is consent-gated + https/official-host-pinned.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Iterator, Optional

from aether_agent.adapter import DEFAULT_HOST

_PROBE_TIMEOUT = 4.0
_SERVE_READY_TIMEOUT = 30.0
_SERVE_POLL_INTERVAL = 0.4

OFFICIAL_HOSTS = ("ollama.com", "www.ollama.com", "github.com")
_WIN_INSTALLER = "https://ollama.com/download/OllamaSetup.exe"
_UNIX_INSTALL_SH = "https://ollama.com/install.sh"
_MANUAL_LINK = "https://ollama.com/download"

ProgressFn = Callable[[str, float, float], None]


def is_official_url(url: str) -> bool:
    """True only for an https URL whose host is an official Ollama host."""
    try:
        p = urllib.parse.urlsplit(url)
    except ValueError:
        return False
    if p.scheme != "https":
        return False
    return (p.hostname or "").lower() in OFFICIAL_HOSTS


def _default_installer_url() -> str:
    return _WIN_INSTALLER if platform.system() == "Windows" else _UNIX_INSTALL_SH


class OllamaCtl:
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        *,
        http_get: Optional[Callable[[str], Any]] = None,
        http_post_stream: Optional[Callable[[str, dict], Iterator[dict]]] = None,
        popen: Optional[Callable[[list[str]], Any]] = None,
        which: Optional[Callable[[str], Optional[str]]] = None,
        sleep: Optional[Callable[[float], None]] = None,
        installer_url: Optional[Callable[[], str]] = None,
        downloader: Optional[Callable[[str], bytes]] = None,
        runner: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.host = host.rstrip("/")
        self._get = http_get or self._default_get
        self._post_stream = http_post_stream or self._default_post_stream
        self._popen = popen or (lambda args: subprocess.Popen(args))
        self._which = which or shutil.which
        self._sleep = sleep or time.sleep
        self._installer_url = installer_url or _default_installer_url
        self._download = downloader or self._default_download
        self._run_installer = runner or self._default_run_installer
        self._owned = False
        self._proc: Any = None

    # --- detection ---------------------------------------------------------
    def _daemon_up(self) -> bool:
        try:
            self._get("/api/tags")
            return True
        except Exception:  # noqa: BLE001 — any failure means "not up"
            return False

    def detect(self) -> dict:
        return {"installed": bool(self._which("ollama")), "daemon_up": self._daemon_up()}

    # --- serve lifecycle ---------------------------------------------------
    def ensure_serve(self) -> bool:
        """Ensure the daemon is up. Start it (and mark OWNED) only if it is down.
        Returns True when reachable, False on a start timeout."""
        if self._daemon_up():
            self._owned = False
            return True
        self._proc = self._popen(["ollama", "serve"])
        self._owned = True
        waited = 0.0
        while waited < _SERVE_READY_TIMEOUT:
            if self._daemon_up():
                return True
            self._sleep(_SERVE_POLL_INTERVAL)
            waited += _SERVE_POLL_INTERVAL
        return False

    def stop_owned(self) -> None:
        """Terminate the serve subprocess only if we started it."""
        if self._owned and self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:  # noqa: BLE001 — teardown is best-effort
                pass
            self._owned = False
            self._proc = None

    # --- model ops ---------------------------------------------------------
    def list_models(self) -> list[dict]:
        try:
            payload = self._get("/api/tags") or {}
        except Exception:  # noqa: BLE001
            return []
        out = []
        for m in payload.get("models", []) if isinstance(payload, dict) else []:
            if isinstance(m, dict) and m.get("name"):
                out.append({"tag": str(m["name"]), "size_bytes": int(m.get("size", 0) or 0)})
        return out

    def ps(self) -> list[dict]:
        try:
            payload = self._get("/api/ps") or {}
        except Exception:  # noqa: BLE001
            return []
        return payload.get("models", []) if isinstance(payload, dict) else []

    def pull(self, tag: str, on_progress: Optional[ProgressFn] = None) -> tuple[bool, str]:
        """Pull/update a model, streaming NDJSON progress to ``on_progress``."""
        last = ""
        try:
            for obj in self._post_stream("/api/pull", {"name": tag, "stream": True}):
                if not isinstance(obj, dict):
                    continue
                if obj.get("error"):
                    return False, str(obj["error"])
                last = str(obj.get("status", ""))
                if on_progress is not None:
                    on_progress(last, float(obj.get("completed", 0) or 0), float(obj.get("total", 0) or 0))
        except Exception as e:  # noqa: BLE001 — surface as a clean failure
            return False, str(e)
        return True, last or "done"

    def remove(self, tag: str) -> tuple[bool, str]:
        try:
            self._default_delete("/api/delete", {"name": tag})
            return True, "removed"
        except Exception as e:  # noqa: BLE001
            return False, str(e)

    # --- install (consent-gated, host-pinned) ------------------------------
    def install(self, *, consent: bool) -> tuple[bool, str]:
        if not consent:
            return False, f"install needs consent; get it yourself: {_MANUAL_LINK}"
        url = self._installer_url()
        if not is_official_url(url):
            return False, f"refusing non-official installer url ({url}); use {_MANUAL_LINK}"
        try:
            blob = self._download(url)
        except Exception as e:  # noqa: BLE001
            return False, f"download failed ({e}); install manually: {_MANUAL_LINK}"
        if not blob:
            return False, f"empty download; install manually: {_MANUAL_LINK}"
        suffix = ".exe" if url.endswith(".exe") else ".sh"
        fd, path = tempfile.mkstemp(prefix="ollama-install-", suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(blob)
            self._run_installer(path)
        except Exception as e:  # noqa: BLE001
            return False, f"installer failed ({e}); install manually: {_MANUAL_LINK}"
        if not self._which("ollama"):
            return False, f"install ran but ollama still not found; see {_MANUAL_LINK}"
        return True, "ollama installed"

    # --- default IO seams (real urllib / subprocess) -----------------------
    def _default_get(self, path: str) -> Any:
        req = urllib.request.Request(self.host + path, method="GET")
        with urllib.request.urlopen(req, timeout=_PROBE_TIMEOUT) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))

    def _default_post_stream(self, path: str, body: dict) -> Iterator[dict]:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.host + path, data=data, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=_SERVE_READY_TIMEOUT * 20) as resp:  # noqa: S310
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    continue

    def _default_delete(self, path: str, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.host + path, data=data, method="DELETE",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=_PROBE_TIMEOUT) as resp:  # noqa: S310
            resp.read()

    def _default_download(self, url: str) -> bytes:
        if not is_official_url(url):  # defense in depth — never fetch a non-official url
            raise ValueError("non-official url")
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 — pinned official host
            return resp.read()

    def _default_run_installer(self, path: str) -> None:
        if path.endswith(".exe"):
            subprocess.run([path, "/VERYSILENT"], check=False, timeout=600)
        else:
            subprocess.run(["sh", path], check=False, timeout=600)


__all__ = ["OllamaCtl", "ProgressFn", "OFFICIAL_HOSTS", "is_official_url"]
