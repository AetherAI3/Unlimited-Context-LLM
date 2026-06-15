# Ollama-wrapped Terminal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `aether` terminal own the full Ollama lifecycle (detect/install/serve/pull/switch) with a clean, hardware-aware first-run, so a local user never touches raw Ollama.

**Architecture:** Four new stdlib-only modules in `aether_agent/` — `progress.py` (pure render), `hardware.py` (resource detect + curated catalog + best-fit), `ollama_ctl.py` (lifecycle over the Ollama HTTP API + a managed `serve` subprocess), `onboarding.py` (preflight state machine) — wired into the existing `repl.py` (preflight on launch + tidy `serve` teardown), `slash.py` (`/pull /setup /doctor /serve` + local-aware `/models`), and `cli.py` (`aether doctor` / `aether setup`). All IO is behind injectable seams so tests run offline.

**Tech Stack:** Python 3.10+, stdlib only (`urllib`, `subprocess`, `ctypes`, `shutil`, `json`, `dataclasses`), `pytest`. No new runtime deps.

**Spec:** `docs/superpowers/specs/2026-06-15-ollama-wrapped-terminal-design.md`
**Branch:** `feat/ollama-wrapped-terminal`
**Gate:** `.venv/Scripts/python.exe -m pytest -q` (Windows venv; fall back to `python -m pytest -q`).

---

## File Structure

| File | New/Mod | Responsibility |
|---|---|---|
| `aether_agent/progress.py` | New | Pure render: progress bar, spinner, step lines, byte/pull formatting. cp1252-safe. |
| `aether_agent/hardware.py` | New | `detect_resources()` (injectable probes, fail-soft) + curated `CATALOG` + `best_fit()`. |
| `aether_agent/ollama_ctl.py` | New | `OllamaCtl`: `detect`, `ensure_serve`/`stop_owned`, `list_models`, `pull`, `ps`, `remove`, `install`. HTTP + `Popen` seams. |
| `aether_agent/onboarding.py` | New | `preflight()` state machine -> typed `Preflight` result; `doctor()` report. |
| `aether_agent/slash.py` | Mod | Add `ollama` to `SlashContext`; add `/pull /setup /doctor /serve`; local-aware `/models`. |
| `aether_agent/repl.py` | Mod | Run `preflight()` on launch; pass `ollama` into `SlashContext`; `stop_owned()` in a `finally`. |
| `aether_agent/cli.py` | Mod | `aether doctor` + `aether setup` subcommands. |
| `tests/test_progress.py` … `tests/test_cli_doctor.py` | New | One test module per unit (offline). |

Catalog note (refines spec): include a permissive low-end floor `qwen2.5-coder:3b` so a 6 GB machine gets a model that actually fits (the Gemma option is license-flagged and never auto-default).

---

## Task 1: `progress.py` — pure render helpers

**Files:**
- Create: `aether_agent/progress.py`
- Test: `tests/test_progress.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_progress.py
from aether_agent import progress


def test_bar_empty_full_and_clamp():
    assert progress.progress_bar(0.0, width=10) == "[----------] 0%"
    assert progress.progress_bar(1.0, width=10) == "[##########] 100%"
    assert progress.progress_bar(-5, width=10) == "[----------] 0%"
    assert progress.progress_bar(2, width=10) == "[##########] 100%"
    assert progress.progress_bar(0.5, width=10) == "[#####-----] 50%"


def test_fmt_bytes():
    assert progress.fmt_bytes(0) == "0 B"
    assert progress.fmt_bytes(1536) == "1.5 KB"
    assert progress.fmt_bytes(5 * 1024**3) == "5.0 GB"


def test_pull_line_and_step_and_spinner():
    line = progress.pull_line("qwen2.5-coder:7b", 2 * 1024**3, 4 * 1024**3)
    assert "qwen2.5-coder:7b" in line and "50%" in line
    assert progress.step("serve", ok=True) == "  [ok] serve"
    assert progress.step("serve", ok=False) == "  [x] serve"
    assert progress.step("serve") == "  ... serve"
    assert progress.spinner_frame(0) in "|/-\\"


def test_output_is_cp1252_safe():
    s = progress.progress_bar(0.5) + progress.step("x", True) + progress.pull_line("m", 1, 2)
    s.encode("cp1252")  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_progress.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aether_agent.progress'`

- [ ] **Step 3: Write minimal implementation**

```python
# aether_agent/progress.py
# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Pure, cp1252-safe render helpers for the Ollama-managed terminal UI.

No IO, no ANSI cursor control here (callers own the terminal) — just strings, so
every function is trivially unit-testable and safe on a Windows cp1252 console.
"""
from __future__ import annotations

_BLOCK = "#"
_EMPTY = "-"
_SPINNER = ("|", "/", "-", "\\")


def progress_bar(frac: float, width: int = 24) -> str:
    """Render ``[####----] 51%``. ``frac`` is clamped to [0, 1]."""
    try:
        f = float(frac)
    except (TypeError, ValueError):
        f = 0.0
    f = 0.0 if f < 0 else 1.0 if f > 1 else f
    filled = round(f * width)
    bar = _BLOCK * filled + _EMPTY * (width - filled)
    return f"[{bar}] {int(round(f * 100))}%"


def fmt_bytes(n: float) -> str:
    """Human bytes: ``4.7 GB``. Whole-byte values render without a decimal."""
    try:
        val = float(n or 0)
    except (TypeError, ValueError):
        val = 0.0
    if val < 1024:
        return f"{int(val)} B"
    for unit in ("KB", "MB", "GB", "TB"):
        val /= 1024
        if val < 1024 or unit == "TB":
            return f"{val:.1f} {unit}"
    return f"{val:.1f} TB"


def spinner_frame(i: int) -> str:
    return _SPINNER[i % len(_SPINNER)]


def step(label: str, ok: bool | None = None) -> str:
    """A step line: ``... label`` (pending) · ``[ok] label`` · ``[x] label``."""
    mark = "..." if ok is None else ("[ok]" if ok else "[x]")
    return f"  {mark} {label}"


def pull_line(tag: str, completed: float, total: float) -> str:
    """Single-line live pull status: ``tag  2.0 GB / 4.0 GB  [####----] 50%``."""
    frac = (float(completed) / float(total)) if total else 0.0
    return f"{tag}   {fmt_bytes(completed)} / {fmt_bytes(total)}   {progress_bar(frac)}"


__all__ = ["progress_bar", "fmt_bytes", "spinner_frame", "step", "pull_line"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_progress.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add aether_agent/progress.py tests/test_progress.py
git commit -m "feat(ollama): progress.py — cp1252-safe progress/step/pull render helpers"
```

---

## Task 2: `hardware.py` — resources + catalog + best-fit

**Files:**
- Create: `aether_agent/hardware.py`
- Test: `tests/test_hardware.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hardware.py
from aether_agent import hardware
from aether_agent.hardware import Resources


def _fit(ram, vram=None, gpu=None):
    res = hardware.detect_resources(
        ram_probe=lambda: ram,
        vram_probe=lambda: (vram, gpu),
    )
    return hardware.best_fit(res)[0]


def test_best_fit_matrix():
    assert _fit(16) == "qwen2.5-coder:7b"          # 16*0.7=11.2 -> 7b (min 8)
    assert _fit(6) == "qwen2.5-coder:3b"           # 6*0.7=4.2 -> 3b (min 4)
    assert _fit(48) == "qwen3-coder:30b"           # 48*0.7=33.6 -> 30b (min 24)
    assert _fit(8, vram=24, gpu="NVIDIA") == "qwen3-coder:30b"  # VRAM wins


def test_low_resources_falls_back_to_permissive_floor():
    # 2 GB fits nothing -> the permissive floor, never the license-flagged Gemma.
    assert _fit(2) == hardware.DEFAULT_FLOOR == "qwen2.5-coder:3b"


def test_flagged_model_never_auto_selected():
    # Even where Gemma (min 6) would "fit", best_fit must skip license-flagged.
    tag = _fit(7)  # 7*0.7=4.9 -> 3b (min4); gemma(min6) excluded anyway
    assert tag != "gemma3n:e4b"


def test_detect_resources_failsoft_default():
    def boom():
        raise OSError("no probe")
    res = hardware.detect_resources(ram_probe=boom, vram_probe=boom)
    assert res == Resources(ram_gb=8.0, vram_gb=None, gpu=None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_hardware.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aether_agent.hardware'`

- [ ] **Step 3: Write minimal implementation**

```python
# aether_agent/hardware.py
# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Hardware detection + a curated local-model catalog + a best-fit picker.

``detect_resources`` is fail-soft (any probe error -> a conservative 8 GB / no-GPU
default) and takes injectable probes so tests never touch the real machine.
``best_fit`` never returns an oversized or license-flagged model as the auto-pick.
"""
from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    tag: str
    size_gb: float
    min_ram_gb: float
    gpu_pref: bool
    license_note: str | None
    blurb: str


# Ordered small -> large. license_note != None => never an auto-default.
CATALOG: tuple[ModelInfo, ...] = (
    ModelInfo("qwen2.5-coder:3b", 1.9, 4, False, None, "low-end, permissive (Apache-2.0)"),
    ModelInfo("gemma3n:e4b", 3.3, 6, False, "Gemma3 custom license", "light machines"),
    ModelInfo("qwen2.5-coder:7b", 4.7, 8, False, None, "universal, fits an 8 GB GPU"),
    ModelInfo("qwen3-coder:30b", 24.0, 24, True, None, "depth build, 256K ctx (Apache-2.0)"),
)
DEFAULT_FLOOR = "qwen2.5-coder:3b"


@dataclass(frozen=True)
class Resources:
    ram_gb: float
    vram_gb: float | None
    gpu: str | None


def detect_resources(*, ram_probe=None, vram_probe=None) -> Resources:
    """Detect RAM/VRAM. Fail-soft to (8 GB, no GPU). Probes injectable for tests."""
    try:
        ram = float((ram_probe or _ram_gb)())
    except Exception:  # noqa: BLE001 — detection must never crash startup
        ram = 8.0
    try:
        vram, gpu = (vram_probe or _vram)()
    except Exception:  # noqa: BLE001
        vram, gpu = None, None
    return Resources(ram_gb=ram, vram_gb=vram, gpu=gpu)


def best_fit(res: Resources) -> tuple[str, str]:
    """Largest permissive model that fits. Returns (tag, reason)."""
    if res.vram_gb and res.vram_gb > 0:
        eff, where = res.vram_gb, (res.gpu or "VRAM")
    else:
        eff, where = res.ram_gb * 0.7, "RAM"
    eligible = [m for m in CATALOG if m.license_note is None and m.min_ram_gb <= eff]
    if not eligible:
        return DEFAULT_FLOOR, f"~{eff:.0f} GB usable -> safe default {DEFAULT_FLOOR}"
    best = max(eligible, key=lambda m: m.size_gb)
    return best.tag, f"{where} ~{eff:.0f} GB -> {best.tag}"


# --- real probes (best-effort, platform-specific) --------------------------
def _ram_gb() -> float:
    sysname = platform.system()
    if sysname == "Windows":
        import ctypes

        class _Mem(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        m = _Mem()
        m.dwLength = ctypes.sizeof(_Mem)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullTotalPhys / 1024**3
    if sysname == "Darwin":
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=5)
        return int(out.strip()) / 1024**3
    # Linux / other: /proc/meminfo
    with open("/proc/meminfo", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024**2  # kB -> GB
    raise OSError("cannot read MemTotal")


def _vram() -> tuple[float | None, str | None]:
    if not shutil.which("nvidia-smi"):
        return None, None
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        timeout=5,
    ).decode("utf-8", "replace").strip()
    first = out.splitlines()[0]
    name, mib = (p.strip() for p in first.split(","))
    return float(mib) / 1024, name  # MiB -> GB


__all__ = ["ModelInfo", "CATALOG", "DEFAULT_FLOOR", "Resources", "detect_resources", "best_fit"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_hardware.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add aether_agent/hardware.py tests/test_hardware.py
git commit -m "feat(ollama): hardware.py — resource detect + curated catalog + best_fit"
```

---

## Task 3: `ollama_ctl.py` — detect / serve / list / pull / ps / remove

**Files:**
- Create: `aether_agent/ollama_ctl.py`
- Test: `tests/test_ollama_ctl.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ollama_ctl.py
from aether_agent.ollama_ctl import OllamaCtl


class _FakeProc:
    def __init__(self):
        self.terminated = False
    def terminate(self):
        self.terminated = True
    def poll(self):
        return None


def _ctl(*, tags_up=True, which="ollama", pull_lines=None, proc=None):
    state = {"serves": 0}

    def http_get(path):
        if path == "/api/tags":
            if tags_up:
                return {"models": [{"name": "qwen2.5-coder:7b", "size": 5_000_000_000}]}
            raise OSError("connection refused")
        if path == "/api/ps":
            return {"models": []}
        raise AssertionError(path)

    def http_post_stream(path, body):
        assert path == "/api/pull"
        for ln in (pull_lines or []):
            yield ln

    def popen(args):
        state["serves"] += 1
        return proc or _FakeProc()

    ctl = OllamaCtl(
        http_get=http_get,
        http_post_stream=http_post_stream,
        popen=popen,
        which=lambda name: which,
        sleep=lambda s: None,
    )
    ctl._serve_calls = state
    return ctl


def test_detect_reports_install_and_daemon():
    d = _ctl(tags_up=True).detect()
    assert d["installed"] is True and d["daemon_up"] is True
    d2 = _ctl(tags_up=False, which=None).detect()
    assert d2["installed"] is False and d2["daemon_up"] is False


def test_ensure_serve_starts_only_when_down_and_owns_it():
    up = _ctl(tags_up=True)
    assert up.ensure_serve() is True
    assert up._owned is False and up._serve_calls["serves"] == 0  # already up -> not owned

    proc = _FakeProc()
    down = _ctl(tags_up=False, proc=proc)
    # daemon is down at first probe; mark it up after the spawn so the poll succeeds
    down._daemon_up = lambda: down._serve_calls["serves"] > 0
    assert down.ensure_serve() is True
    assert down._owned is True and down._serve_calls["serves"] == 1


def test_stop_owned_only_kills_what_we_started():
    proc = _FakeProc()
    down = _ctl(tags_up=False, proc=proc)
    down._daemon_up = lambda: down._serve_calls["serves"] > 0
    down.ensure_serve()
    down.stop_owned()
    assert proc.terminated is True

    up = _ctl(tags_up=True)
    up.ensure_serve()
    up.stop_owned()  # nothing owned -> no-op, must not raise


def test_pull_streams_progress_callbacks():
    lines = [
        {"status": "pulling", "completed": 1, "total": 4},
        {"status": "pulling", "completed": 4, "total": 4},
        {"status": "success"},
    ]
    ctl = _ctl(pull_lines=lines)
    seen = []
    ok, detail = ctl.pull("qwen2.5-coder:7b", on_progress=lambda s, c, t: seen.append((s, c, t)))
    assert ok is True
    assert seen[0] == ("pulling", 1, 4) and seen[-1][0] == "success"


def test_list_models_returns_installed_tags():
    ctl = _ctl(tags_up=True)
    models = ctl.list_models()
    assert {"tag": "qwen2.5-coder:7b", "size_bytes": 5_000_000_000} in models
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ollama_ctl.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aether_agent.ollama_ctl'`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ollama_ctl.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add aether_agent/ollama_ctl.py tests/test_ollama_ctl.py
git commit -m "feat(ollama): ollama_ctl.py — detect/serve(owned)/list/pull/ps/remove over the HTTP API"
```

---

## Task 4: install security tests (the installer shipped in Task 3)

**Files:**
- Test: `tests/test_install_security.py` (exercises `OllamaCtl.install` + `is_official_url` from Task 3)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_install_security.py
from aether_agent.ollama_ctl import OllamaCtl, is_official_url


def test_official_url_pinning():
    assert is_official_url("https://ollama.com/download/OllamaSetup.exe") is True
    assert is_official_url("https://github.com/ollama/ollama/releases/x") is True
    assert is_official_url("http://ollama.com/x") is False          # https only
    assert is_official_url("https://evil.example.com/OllamaSetup.exe") is False


def test_install_refuses_without_consent():
    calls = []
    ctl = OllamaCtl(downloader=lambda url: calls.append(url) or b"",
                    runner=lambda path: calls.append(("run", path)))
    ok, detail = ctl.install(consent=False)
    assert ok is False and "consent" in detail.lower()
    assert calls == []  # nothing downloaded, nothing run


def test_install_refuses_non_official_url():
    ctl = OllamaCtl(installer_url=lambda: "https://evil.example.com/x.exe",
                    downloader=lambda url: b"X", runner=lambda p: None)
    ok, detail = ctl.install(consent=True)
    assert ok is False and "official" in detail.lower()


def test_install_runs_official_with_consent():
    ran = {}
    ctl = OllamaCtl(
        installer_url=lambda: "https://ollama.com/download/OllamaSetup.exe",
        downloader=lambda url: b"INSTALLER-BYTES",
        runner=lambda path: ran.setdefault("path", path),
        which=lambda n: "ollama",  # re-detect succeeds after install
    )
    ok, detail = ctl.install(consent=True)
    assert ok is True and ran["path"].endswith(".exe")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_install_security.py -q`
Expected: PASS already IF Task 3 shipped `install`/`is_official_url`. If you split Task 3 without `install`, this FAILS with `ImportError: cannot import name 'is_official_url'` — go back and add the install block from Task 3.

- [ ] **Step 3: (No new impl — install shipped in Task 3.)** If any assertion fails, fix `install`/`is_official_url` minimally in `ollama_ctl.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_install_security.py tests/test_ollama_ctl.py -q`
Expected: PASS (9 passed total)

- [ ] **Step 5: Commit**

```bash
git add tests/test_install_security.py
git commit -m "test(ollama): install security — consent gate, https+official-host pin, manual fallback"
```

---

## Task 5: `onboarding.py` — preflight state machine + doctor

**Files:**
- Create: `aether_agent/onboarding.py`
- Test: `tests/test_onboarding.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_onboarding.py
from aether_agent import onboarding
from aether_agent.onboarding import Preflight


class _Ctl:
    def __init__(self, installed=True, daemon=True, models=None, pull_ok=True):
        self._installed, self._daemon = installed, daemon
        self._models = models if models is not None else ["qwen2.5-coder:7b"]
        self._pull_ok = pull_ok
        self.installed_called = self.served = self.pulled = None
    def detect(self):
        return {"installed": self._installed, "daemon_up": self._daemon}
    def install(self, *, consent):
        self.installed_called = consent
        self._installed = consent
        return (consent, "ok" if consent else "declined")
    def ensure_serve(self):
        self.served = True
        self._daemon = True
        return True
    def list_models(self):
        return [{"tag": t, "size_bytes": 1} for t in self._models]
    def pull(self, tag, on_progress=None):
        self.pulled = tag
        if self._pull_ok:
            self._models.append(tag)
        return (self._pull_ok, "success" if self._pull_ok else "boom")
    def stop_owned(self):
        pass


def test_preflight_healthy_is_silent_noop():
    ctl = _Ctl(installed=True, daemon=True, models=["qwen2.5-coder:7b"])
    pf = onboarding.preflight(ctl, resources=lambda: ("qwen2.5-coder:7b", "ok"),
                              prompt=lambda q: "")  # never prompted
    assert pf.ok is True and pf.chosen_model == "qwen2.5-coder:7b"
    assert ctl.served is None and ctl.pulled is None  # nothing to do


def test_preflight_full_path_installs_serves_picks_pulls():
    ctl = _Ctl(installed=False, daemon=False, models=[])
    answers = iter(["y", ""])  # y to install, Enter to accept the pick
    pf = onboarding.preflight(ctl, resources=lambda: ("qwen2.5-coder:7b", "16 GB -> 7b"),
                              prompt=lambda q: next(answers))
    assert ctl.installed_called is True
    assert ctl.served is True
    assert ctl.pulled == "qwen2.5-coder:7b"
    assert pf.ok is True and pf.chosen_model == "qwen2.5-coder:7b"


def test_preflight_declined_install_exits_clean():
    ctl = _Ctl(installed=False, daemon=False, models=[])
    pf = onboarding.preflight(ctl, resources=lambda: ("qwen2.5-coder:7b", "x"),
                              prompt=lambda q: "n")
    assert pf.ok is False and ctl.served is None
    assert "manual" in pf.message.lower() or "install" in pf.message.lower()


def test_doctor_reports_each_check():
    ctl = _Ctl(installed=True, daemon=True, models=["qwen2.5-coder:7b"])
    report = onboarding.doctor(ctl, resources=lambda: ("qwen2.5-coder:7b", "ok"))
    assert "ollama" in report.lower() and "model" in report.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_onboarding.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aether_agent.onboarding'`

- [ ] **Step 3: Write minimal implementation**

```python
# aether_agent/onboarding.py
# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""First-run / preflight state machine for the Ollama-managed terminal.

``preflight(ctl, ...)`` runs detect -> (install?) -> serve -> pick -> pull and
returns a typed :class:`Preflight`. Silent and instant when everything is already
healthy. ``prompt``/``resources``/``emit`` are injected so the flow is testable
with no TTY, no network, no real Ollama. ``emit`` defaults to a no-op (quiet
tests); ``repl.py`` passes a real writer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from aether_agent import hardware, progress

Prompt = Callable[[str], str]
Emit = Callable[[str], None]
ResourcePick = Callable[[], "tuple[str, str]"]


@dataclass
class Preflight:
    ok: bool
    chosen_model: str | None = None
    message: str = ""
    steps: list[str] = field(default_factory=list)


def _default_pick() -> "tuple[str, str]":
    return hardware.best_fit(hardware.detect_resources())


def preflight(
    ctl: Any,
    *,
    resources: ResourcePick | None = None,
    prompt: Prompt = lambda q: "",
    emit: Emit = lambda s: None,
) -> Preflight:
    pick = resources or _default_pick
    steps: list[str] = []

    if not ctl.detect().get("installed"):
        ans = prompt("Ollama not found. Install it now? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            return Preflight(False, None, "Install Ollama manually: https://ollama.com/download", steps)
        ok, detail = ctl.install(consent=True)
        steps.append(progress.step("install ollama", ok))
        emit(steps[-1] + "\n")
        if not ok:
            return Preflight(False, None, detail, steps)

    if not ctl.detect().get("daemon_up"):
        ok = ctl.ensure_serve()
        steps.append(progress.step("start ollama serve", ok))
        emit(steps[-1] + "\n")
        if not ok:
            return Preflight(False, None, "could not start ollama serve", steps)

    installed = {m["tag"] for m in ctl.list_models()}
    if installed:
        return Preflight(True, sorted(installed)[0], "", steps)

    tag, reason = pick()
    emit(f"  recommended: {tag}  ({reason})\n")
    ans = prompt(f"Pull {tag}? [Enter=yes / tag <x>] ").strip()
    if ans.lower().startswith("tag "):
        tag = ans[4:].strip() or tag
    ok, detail = ctl.pull(tag, on_progress=lambda s, c, t: emit("\r" + progress.pull_line(tag, c, t)))
    emit("\n" + progress.step(f"pull {tag}", ok) + "\n")
    if not ok:
        return Preflight(False, None, f"pull failed: {detail}", steps)
    return Preflight(True, tag, "", steps)


def doctor(ctl: Any, *, resources: ResourcePick | None = None) -> str:
    """Report-only health check with exact fixes. Never starts/installs anything."""
    pick = resources or _default_pick
    det = ctl.detect()
    models = [m["tag"] for m in ctl.list_models()] if det.get("daemon_up") else []
    tag, reason = pick()
    lines = [
        "aether doctor",
        progress.step("ollama installed", det.get("installed", False)),
        progress.step("ollama daemon up", det.get("daemon_up", False)),
        progress.step(f"a model is pulled ({len(models)})", bool(models)),
        f"  recommended for this machine: {tag}  ({reason})",
    ]
    if not det.get("installed"):
        lines.append("  fix: install -> https://ollama.com/download (or run /setup)")
    elif not det.get("daemon_up"):
        lines.append("  fix: start it -> run /setup (or `ollama serve`)")
    elif not models:
        lines.append(f"  fix: pull a model -> /pull {tag}")
    return "\n".join(lines)


__all__ = ["Preflight", "preflight", "doctor"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_onboarding.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add aether_agent/onboarding.py tests/test_onboarding.py
git commit -m "feat(ollama): onboarding.py — preflight state machine + doctor report"
```

---

## Task 6: `slash.py` — `/pull /setup /doctor /serve` + local-aware `/models`

**Files:**
- Modify: `aether_agent/slash.py` (add `ollama` to `SlashContext`; new handlers; registry; help)
- Test: `tests/test_slash_ollama.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_slash_ollama.py
from aether_agent.slash import SlashContext, dispatch


class _Ctl:
    def detect(self):
        return {"installed": True, "daemon_up": True}
    def list_models(self):
        return [{"tag": "qwen2.5-coder:7b", "size_bytes": 5_000_000_000}]
    def ps(self):
        return []
    def pull(self, tag, on_progress=None):
        return (True, "success")


def _ctx():
    return SlashContext(api=None, authed=False, model="qwen2.5-coder:7b", ollama=_Ctl())


def test_models_local_lists_installed_with_marker():
    out = dispatch(_ctx(), "/models")["text"]
    assert "qwen2.5-coder:7b" in out and "›" in out


def test_doctor_dispatch():
    assert "ollama" in dispatch(_ctx(), "/doctor")["text"].lower()


def test_pull_dispatch_reports_result():
    assert "qwen2.5-coder:7b" in dispatch(_ctx(), "/pull qwen2.5-coder:7b")["text"]


def test_serve_dispatch_shows_daemon_state():
    out = dispatch(_ctx(), "/serve")["text"].lower()
    assert "up" in out or "running" in out


def test_setup_dispatch_recognized_not_unknown():
    res = dispatch(_ctx(), "/setup")
    assert "text" in res and "unknown" not in res["text"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_slash_ollama.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'ollama'`

- [ ] **Step 3: Write minimal implementation** (edit `aether_agent/slash.py`)

Add `ollama` field to `SlashContext` (immediately after the `web: Any = None` line):

```python
    ollama: Any = None  # an ollama_ctl.OllamaCtl (local lifecycle); None when cloud-only
```

Replace the existing `_models` function body with a local-first version, and add the new handlers (place after `_web`):

```python
def _models(ctx: SlashContext, arg: str) -> SlashResult:
    if not ctx.authed and ctx.ollama is not None:
        return _ollama_models(ctx)
    if not ctx.authed:
        return _text("(local Ollama — set a model with /model <tag>)")
    payload = ctx.api.get_json(MODELS_PATH)
    models = _items(payload, "models")
    tier = (payload or {}).get("tier", "") if isinstance(payload, dict) else ""
    lines: list[str] = []
    if tier:
        lines.append(f"tier: {tier}")
    for i, m in enumerate(models, 1):
        mid = str(m.get("id", ""))
        label = str(m.get("label", "") or "")
        mark = "›" if mid and mid == ctx.model else " "
        lines.append(f"{mark} {i:>2}. {mid}\t{label}".rstrip())
    lines.append("switch: /model <tag>")
    return _text("\n".join(lines))


def _ollama_models(ctx: SlashContext) -> SlashResult:
    from aether_agent.hardware import CATALOG, best_fit, detect_resources

    ctl = ctx.ollama
    installed = {m["tag"] for m in ctl.list_models()} if ctl else set()
    rec, _ = best_fit(detect_resources())
    lines = ["local models (* installed, › current, ! recommended):"]
    for m in CATALOG:
        dot = "*" if m.tag in installed else " "
        cur = "›" if m.tag == ctx.model else " "
        star = "!" if m.tag == rec else " "
        note = f"  [{m.license_note}]" if m.license_note else ""
        lines.append(f"{dot}{cur}{star} {m.tag}\t{m.size_gb:g}GB\t{m.blurb}{note}")
    for tag in sorted(installed):
        if all(tag != m.tag for m in CATALOG):
            cur = "›" if tag == ctx.model else " "
            lines.append(f"*{cur}  {tag}\t(installed)")
    lines.append("switch: /model <tag>   ·   pull: /pull <tag>")
    return _text("\n".join(lines))


def _pull(ctx: SlashContext, arg: str) -> SlashResult:
    tag = arg.strip()
    if not tag:
        return _text("usage: /pull <tag>")
    if ctx.ollama is None:
        return _text("(pull is a local-Ollama command)")
    ok, detail = ctx.ollama.pull(tag)
    return _text(f"pull {tag}: {'ok' if ok else 'failed'} ({detail})")


def _doctor(ctx: SlashContext, arg: str) -> SlashResult:
    if ctx.ollama is None:
        return _text("(doctor checks the local Ollama; not available in cloud-only mode)")
    from aether_agent import onboarding
    return _text(onboarding.doctor(ctx.ollama))


def _serve(ctx: SlashContext, arg: str) -> SlashResult:
    if ctx.ollama is None:
        return _text("(serve status is a local-Ollama command)")
    det = ctx.ollama.detect()
    owned = bool(getattr(ctx.ollama, "_owned", False))
    state = "up" if det.get("daemon_up") else "down"
    return _text(f"ollama daemon: {state}{' (owned by this session)' if owned else ''}")


def _setup(ctx: SlashContext, arg: str) -> SlashResult:
    return {"setup": True, "text": "re-running setup…"}
```

Delete the OLD `_models` definition (there must be exactly one). Register the new commands in `REGISTRY` (after `"web": _web,`):

```python
    "pull": _pull,
    "doctor": _doctor,
    "serve": _serve,
    "setup": _setup,
```

Add to `_HELP_LINES` (after the `/web` line):

```python
    "/pull <tag>        download/update a local model",
    "/doctor            check the local Ollama setup",
    "/serve             show the Ollama daemon state",
    "/setup             re-run first-run setup",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_slash_ollama.py tests/test_slash.py -q`
Expected: PASS (existing `test_slash.py` still green; new file green)

- [ ] **Step 5: Commit**

```bash
git add aether_agent/slash.py tests/test_slash_ollama.py
git commit -m "feat(ollama): slash /pull /doctor /serve /setup + local-aware /models"
```

---

## Task 7: `repl.py` — preflight on launch + tidy teardown + ctx.ollama

**Files:**
- Modify: `aether_agent/repl.py` (`main()`, `_make_ctx`, add `_build_ollama`/`_preflight` seams)
- Test: `tests/test_repl_preflight.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_repl_preflight.py
import io

from aether_agent import repl


def test_make_ctx_carries_ollama():
    ctx = repl._make_ctx(authed=False, api=None, model="m", ollama="OLLAMA_SENTINEL")
    assert ctx.ollama == "OLLAMA_SENTINEL"


def test_main_runs_preflight_then_stops_owned(monkeypatch):
    calls = {"preflight": 0, "stopped": 0}

    class _Ctl:
        def stop_owned(self):
            calls["stopped"] += 1

    monkeypatch.setattr(repl, "_build_ollama", lambda backend, authed: _Ctl())

    def fake_preflight(c, **kw):
        calls["preflight"] += 1
        from aether_agent.onboarding import Preflight
        return Preflight(True, "qwen2.5-coder:7b", "", [])

    monkeypatch.setattr(repl, "_preflight", fake_preflight)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))  # immediate EOF -> loop exits

    rc = repl.main([])
    assert rc == 0
    assert calls["preflight"] == 1
    assert calls["stopped"] == 1  # teardown ran in finally
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_repl_preflight.py -q`
Expected: FAIL — `AttributeError: module 'aether_agent.repl' has no attribute '_build_ollama'`

- [ ] **Step 3: Write minimal implementation** (edit `aether_agent/repl.py`)

Add imports near the other `aether_agent` imports:

```python
from aether_agent.ollama_ctl import OllamaCtl
from aether_agent.onboarding import preflight as _preflight_impl
```

Add seams (module level, e.g. just above `_make_ctx`):

```python
def _build_ollama(backend: str, authed: bool) -> Any:
    """The local Ollama controller, only when this session will serve locally."""
    b = (backend or "auto").strip().lower()
    if b == "cloud" or (b == "auto" and authed):
        return None  # cloud session — no local daemon to manage
    return OllamaCtl()


def _preflight(ctl: Any, **kw: Any) -> Any:
    return _preflight_impl(ctl, **kw)
```

Change `_make_ctx` to carry `ollama`:

```python
def _make_ctx(authed: bool, api: Any, model: str, ollama: Any = None) -> SlashContext:
    return SlashContext(api=api, authed=authed, model=model, ollama=ollama)
```

In `main()`, replace everything from the splash write (`label = _backend_label(...)`) through the end of the `while True:` loop with the block below (adds preflight before the splash, threads `chosen_model`/`ollama`, wraps the loop in try/finally, and handles the `/setup` flag):

```python
    ollama = _build_ollama(backend, authed)
    chosen_model = model
    if ollama is not None:
        pf = _preflight(ollama, emit=lambda s: _safe_write(out, s))
        if not pf.ok:
            _safe_write(out, f"\n{pf.message}\n")
            ollama.stop_owned()
            return 1
        chosen_model = pf.chosen_model or model

    label = _backend_label(backend, authed)
    short_backend = "cloud" if "cloud" in label else "local"
    _safe_write(out, render_splash(VERSION, chosen_model or "auto", short_backend) + "\n\n")
    _safe_write(out, "Type a prompt, or /help for commands. /exit to quit.\n\n")

    ctx = _make_ctx(authed, api, chosen_model, ollama)
    is_tty = bool(getattr(sys.stdin, "isatty", lambda: False)())
    if is_tty:
        _enable_readline_history()

    try:
        while True:
            try:
                line = input(_PROMPT) if is_tty else _read_line()
            except KeyboardInterrupt:
                out.write("\n")
                return 0
            except EOFError:
                out.write("\n")
                return 0
            if line is None:
                return 0
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                res = dispatch(ctx, line)
                if res.get("exit"):
                    return 0
                text = res.get("text")
                if text:
                    _safe_write(out, text + "\n")
                if res.get("setup") and ollama is not None:
                    pf = _preflight(ollama, emit=lambda s: _safe_write(out, s))
                    ctx.model = pf.chosen_model or ctx.model
                if res.get("restart"):
                    ctx.authed = store.get() is not None
                    _safe_write(out, "(session restarted — context cleared)\n")
                continue
            brain = select_brain(
                authed=store.get() is not None,
                backend=backend,
                api=api,
                model=ctx.model or chosen_model or "",
            )
            _run_turn(brain, line, out)
            out.write("\n")
    finally:
        if ollama is not None:
            ollama.stop_owned()
```

(`return` statements inside the `try` still run the `finally`, so teardown always fires.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_repl_preflight.py tests/test_cli_dispatch.py -q`
Expected: PASS (preflight test green; existing REPL/cli tests still green)

- [ ] **Step 5: Commit**

```bash
git add aether_agent/repl.py tests/test_repl_preflight.py
git commit -m "feat(ollama): REPL preflight on launch + tidy serve teardown + ctx.ollama"
```

---

## Task 8: `cli.py` — `aether doctor` / `aether setup`

**Files:**
- Modify: `aether_agent/cli.py` (`_SUBCOMMANDS`, dispatch, handlers, help)
- Test: `tests/test_cli_doctor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_doctor.py
from aether_agent import cli


def test_doctor_subcommand_prints_report(capsys, monkeypatch):
    class _Ctl:
        def detect(self):
            return {"installed": True, "daemon_up": True}
        def list_models(self):
            return [{"tag": "qwen2.5-coder:7b", "size_bytes": 1}]
    monkeypatch.setattr(cli, "_ollama_ctl", lambda: _Ctl())
    rc = cli.main(["doctor"])
    out = capsys.readouterr().out.lower()
    assert rc == 0 and "ollama" in out and "model" in out


def test_setup_subcommand_runs_preflight(capsys, monkeypatch):
    seen = {"preflight": 0}

    class _Ctl:
        def stop_owned(self):
            pass
    monkeypatch.setattr(cli, "_ollama_ctl", lambda: _Ctl())

    def fake_preflight(ctl, **kw):
        seen["preflight"] += 1
        from aether_agent.onboarding import Preflight
        return Preflight(True, "qwen2.5-coder:7b", "", [])
    monkeypatch.setattr(cli, "_preflight", fake_preflight)
    rc = cli.main(["setup"])
    assert rc == 0 and seen["preflight"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli_doctor.py -q`
Expected: FAIL — `AttributeError: module 'aether_agent.cli' has no attribute '_ollama_ctl'`

- [ ] **Step 3: Write minimal implementation** (edit `aether_agent/cli.py`)

Change `_SUBCOMMANDS` to include the two new commands:

```python
_SUBCOMMANDS = frozenset({"code", "brain", "auth", "models", "config", "doctor", "setup"})
```

Add seams + handlers (near the other `_cmd_*` functions):

```python
def _ollama_ctl():
    from aether_agent.ollama_ctl import OllamaCtl
    return OllamaCtl()


def _preflight(ctl, **kw):
    from aether_agent.onboarding import preflight
    return preflight(ctl, **kw)


def _cmd_doctor(rest: list[str]) -> int:
    from aether_agent.onboarding import doctor
    print(doctor(_ollama_ctl()))
    return 0


def _cmd_setup(rest: list[str]) -> int:
    ctl = _ollama_ctl()
    pf = _preflight(ctl, emit=lambda s: sys.stdout.write(s))
    if not pf.ok:
        print(f"\n{pf.message}", file=sys.stderr)
        return 1
    print(f"\nready — model: {pf.chosen_model}")
    return 0
```

Wire into `main()` dispatch (after the `config` branch, before the `head.startswith("-")` check):

```python
    if head == "doctor":
        return _cmd_doctor(args[1:])
    if head == "setup":
        return _cmd_setup(args[1:])
```

Add two lines to `_print_help()` (after the `config` line in the list):

```python
                "  aether doctor                check the local Ollama setup",
                "  aether setup                 run first-run setup (install/serve/pull)",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli_doctor.py tests/test_cli_dispatch.py -q`
Expected: PASS (new green; existing cli dispatch tests still green)

- [ ] **Step 5: Commit**

```bash
git add aether_agent/cli.py tests/test_cli_doctor.py
git commit -m "feat(ollama): aether doctor + aether setup subcommands"
```

---

## Task 9: Full-suite green + boot smoke + push

**Files:** full suite

- [ ] **Step 1: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS — all prior tests (445+) plus the ~31 new tests from Tasks 1–8. Zero regressions.

- [ ] **Step 2: Boot smoke (no Ollama needed — report path)**

Run (bash): `AETHER_CONFIG_DIR="$(mktemp -d)" .venv/Scripts/python.exe -m aether_agent.cli doctor`
Expected: an `aether doctor` report — `ollama installed`, `ollama daemon up`, `a model is pulled (N)`, a recommended model — no crash, exit 0.

- [ ] **Step 3: Fix any cross-module wiring that is red, re-run**

Common pitfalls: `SlashContext` field order (keep `ollama` last so existing positional callers are unaffected); `repl.main` early-returns must still hit the `finally` (they do — `return` inside `try`); the old `_models` left duplicated (there must be exactly one).

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "test(ollama): full-suite green after Ollama-wrapped terminal (sub-project A)"
```

- [ ] **Step 5: Push the branch**

```bash
git push -u origin feat/ollama-wrapped-terminal
```

---

## Self-review (author checklist — completed)

- **Spec coverage:** full-manage ownership (Tasks 3,7) · consent-gated install (Tasks 3,4) · auto hardware-aware pick (Tasks 2,5) · tidy serve lifecycle (Task 3 `ensure_serve`/`stop_owned` + Task 7 `finally`) · onboarding flow (Task 5) · control surface (Task 3) · catalog/best-fit (Task 2) · TUI progress (Task 1) · slash `/models /model /pull /setup /doctor /serve` (Task 6) · install security (Tasks 3,4) · `/doctor` folds in the old doctor (Task 5) · per-unit offline tests (every task). All spec sections map to a task.
- **Placeholder scan:** no TBD/TODO; every code step shows complete code. The full picker UI is an explicit scoped cut (Enter / `tag <x>` ships now), not a placeholder.
- **Type consistency:** `OllamaCtl` methods (`detect/ensure_serve/stop_owned/list_models/pull/ps/remove/install`) identical across Tasks 3–8; `Preflight(ok, chosen_model, message, steps)` consistent in Tasks 5,7,8; `progress.step/pull_line/progress_bar` signatures match Tasks 1,5; `SlashContext(..., ollama=None)` matches Tasks 6,7; `_build_ollama`/`_preflight` seam names match Task 7 test + impl; `_ollama_ctl`/`_preflight` match Task 8 test + impl.
- **Scope:** sub-project A only; B/C/D explicitly out.
```
