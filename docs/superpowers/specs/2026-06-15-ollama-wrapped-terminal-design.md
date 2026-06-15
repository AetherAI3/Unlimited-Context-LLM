# Design Spec — Ollama-wrapped Terminal ("Claude Code for local")

**Date:** 2026-06-15
**Repo:** `unlimited-context-llm` (Python) · branch `feat/ollama-wrapped-terminal`
**Base:** stacked on `feat/native-terminal-mirror` (uses its `aether_agent/repl.py`, `slash.py`,
`statusbar.py`, `splash.py`, `adapter.py`). Land/rebase that branch first, or keep stacked.

## Sub-project A of 4

This is the **foundation** of a larger vision ("Claude Code for local"): a beautiful, clean local
agent terminal. The full vision decomposes into:

- **A — Ollama-wrapped terminal (THIS SPEC):** the terminal owns the whole Ollama lifecycle so the
  user never touches raw Ollama.
- B — Custom local agents (saved/shareable profile + own memory pool + effort + persona).
- C — Custom local commands (user/agent-defined slash commands).
- D — Multi-agent concurrency + `/agents` column dashboard (run several agents at once).

B/C/D are **out of scope here** and get their own spec → plan → build. A ships standalone.

## Goal

Make `aether` (the Python terminal in `aether_agent/`) **own the full Ollama lifecycle** and feel like
Claude Code: detect/install Ollama, supervise the daemon, pick a model that fits the machine, pull it
with a live progress bar, and drop the user into a clean REPL — **zero second terminal, zero raw
`ollama` commands required**.

## Locked decisions (from brainstorm)

1. **Ownership = full manage.** Terminal detects Ollama; if missing, offers a **consent-gated** install;
   supervises `ollama serve` (starts if down); pulls/switches/removes models. Install + serve are always
   shown before they run.
2. **Model onboarding = auto, hardware-aware.** Detect RAM/VRAM → pick best-fitting curated model →
   one keystroke to accept + pull → change later with `/model`.
3. **serve lifecycle = tidy.** If the daemon was already running, leave it. If WE started it, stop it on
   exit (track ownership). No orphan daemons, no surprise.
4. **Scope = UCL (Python) only.** The TS `aether-code` already connects to a user-run Ollama; managing
   the daemon from Node is a separate later mirror if wanted.
5. stdlib-only, no new runtime deps.

## Architecture — new `aether_agent/` modules

| Module | Purpose | Talks to |
|---|---|---|
| `ollama_ctl.py` | Lifecycle surface: `detect()`, `install(consent)`, `ensure_serve()` / `stop_owned()`, `list_models()`, `pull(tag, on_progress)`, `ps()`, `remove(tag)` | HTTP `:11434` (`/api/tags`, `/api/pull` stream, `/api/ps`, `/api/delete`) + `subprocess` (install, `serve`) |
| `hardware.py` | RAM/VRAM detection (fail-soft) + curated model catalog + `best_fit(ram, vram) -> (tag, reason)` | OS only |
| `onboarding.py` | First-run / preflight state machine: detect → (install?) → serve → pick → pull → ready; idempotent (`/setup`, `aether doctor`) | the two above |
| `progress.py` | cp1252-safe progress bar + spinner + step lines; single-line live `\r` updates | — |

Wires into existing: `repl.py` (preflight on launch, silent when healthy; status bar; exit hook for
`stop_owned`), `slash.py` (new commands), `statusbar.py` (daemon+model fields), `adapter.py` (reuses
`OllamaChat`, `DEFAULT_HOST`). New CLI entry behavior in `cli.py`: `aether doctor` / `aether setup`.

### Reuse (do not duplicate)
- `adapter.DEFAULT_HOST` (`http://localhost:11434`), `OllamaChat`.
- `smoke._ollama_up` probe pattern (lift the health-probe helper into `ollama_ctl`).
- README tier math + `adapter.py` model notes (Gemma3 license caveat) for the catalog.
- The existing `aether-context doctor` logic (Ollama/model/disk/RAM) — fold its checks into `/doctor`.

## Preflight / onboarding flow (`onboarding.py`)

Runs on every launch; **silent and instant when everything is green**.

```
detect ollama -- installed? --no--> [Install Ollama? y/N]
     | yes                                |  y                   |  n
     |                                    install(consent)       print manual link, exit clean (rc 0)
     |                                    (official src, see Install Security)
     v
ensure serve -- up? --no--> start `serve` (mark OWNED) --> health-poll until ready (timeout->clean error)
     | yes (not owned)
     v
model present? --no--> hardware.best_fit() --> [Enter pull . m pick . tag <x>] --> pull(stream %)
     | yes
     v
ready --> REPL (status bar: model . daemon . pool)
exit  --> if OWNED: stop_owned() . else: leave running
```

- Re-runnable as `/setup`.
- `/doctor` = the same checks, **report-only**, prints exact fixes (e.g. `ollama pull <tag>`, disk free,
  RAM/VRAM, pool reach) — supersedes/folds in `aether-context doctor`.
- State machine returns a typed result `(ok: bool, steps: list[Step], chosen_model: str|None)` so it is
  testable without a TTY and so `repl.py` can render it.

## Ollama control surface (`ollama_ctl.py`)

- `detect()` -> `{installed: bool, version: str|None, daemon_up: bool}`. `installed` = `ollama` on PATH
  (or known install dir); `daemon_up` = `/api/tags` answers.
- `ensure_serve()` -> probe `/api/tags`; if down, `subprocess.Popen(["ollama","serve"])`, set
  `self._owned = True`, poll `/api/tags` until healthy (bounded; clean error on timeout). If already up,
  `self._owned = False`. `stop_owned()` terminates the child **only if** `_owned`.
- `pull(tag, on_progress)` -> POST `/api/pull` `{name:tag, stream:true}`; read NDJSON lines
  `{status, completed?, total?}`; call `on_progress(status, completed, total)` per line -> drives the bar.
  Returns ok/err; never raises into the UI loop.
- `list_models()` -> `/api/tags` -> installed `[{tag, size_bytes}]`. `ps()` -> `/api/ps` (loaded models).
  `remove(tag)` -> `/api/delete`.
- `install(consent: bool)` -> see Install Security. Returns `(ok, detail)`; refuses without consent.

All HTTP via `urllib` (stdlib); all subprocess via `subprocess`. HTTP + `Popen` are **injectable seams**
(constructor args / module-level overridable) so tests run offline with no real Ollama.

## Hardware-aware pick + catalog (`hardware.py`)

- `detect_resources()` -> `{ram_gb: float, vram_gb: float|None, gpu: str|None}`, **fail-soft** ->
  conservative default `{8.0, None, None}` on any error. Stdlib only:
  - Windows: RAM via `ctypes.windll.kernel32.GlobalMemoryStatusEx`; VRAM via `nvidia-smi` if present, else
    `wmic path win32_VideoController get AdapterRAM` (best-effort).
  - Linux: RAM `/proc/meminfo`; VRAM `nvidia-smi` / `/sys/class/drm/*/device/mem_info_vram_total`.
  - macOS: RAM `sysctl hw.memsize`; VRAM `system_profiler SPDisplaysDataType` (best-effort).
- **Catalog** (curated, ordered small->large), each entry:
  `{tag, size_gb, min_ram_gb, gpu_pref: bool, license_note: str|None, blurb}`:
  - `qwen2.5-coder:7b` — 4.7 GB — min 8 — universal, fits 8 GB GPU — **default floor**
  - `gemma3n:e4b` — 3.3 GB — min 6 — light machines — *Gemma3 = custom license -> never auto-default*
  - `qwen3-coder:30b` — ~24 GB — min 24 — depth build (256K ctx, Apache-2.0)
  - (catalog is data — extendable; pin Apache/permissive as the auto-default-eligible set)
- `best_fit(ram, vram)` -> largest catalog model whose `min_ram_gb <= effective_available`, where
  `effective_available = vram if gpu_pref and vram else ram*0.7` (leave ~30 % headroom); skip
  license-flagged models for the auto-pick; return `(tag, reason_str)`. Never returns a flagged or
  oversized model as the auto-default.
- Drives the onboarding pick **and** `/models` (catalog + installed + current + fit yes/no).

## TUI + slash (`progress.py`, `slash.py`, `statusbar.py`)

- `progress_bar(frac, width=24)` -> `[######------] 51%`; ASCII-safe (`#`/`-`), cp1252-safe.
- `spinner()` frames + `step(label, ok)` -> `[ok] <label>` / `... <label>` (ASCII under cp1252).
- Pull renderer: single-line live `\r` update — `qwen2.5-coder:7b   2.4 / 4.7 GB   51%` — no scroll spam.
- Status bar (extend `statusbar.py`): `model . daemon(up|owned) . pool reach . hit%`.
- New slash commands (registered in `slash.py`, dispatch-pure where possible):
  - `/models` — catalog + installed + current + fit marks.
  - `/model <tag>` — switch active model; pull-if-missing (progress); clears context.
  - `/pull <tag>` — pull/update a model (progress bar).
  - `/setup` — re-run onboarding.
  - `/doctor` — health report + exact fixes.
  - `/serve` — daemon status (up? owned? pid).
- Aesthetic: muted palette (cyan accent, dim secondary), aligned columns, splash on entry showing the
  active model + daemon state. Minimal color, ASCII-safe (consistent with the cp1252 rule already in
  `splash.py`/`smoke.py`).

## Install security (CRITICAL)

- **Consent-gated, never silent.** The only auto-download is the **official** Ollama installer from a
  **pinned official host** (`ollama.com` / official `github.com/ollama/ollama` releases), **https-only**.
- Flow: detect missing -> show URL + version that will run -> require explicit `y` -> download to a temp
  file -> verify (HTTPS + official host; checksum where the project publishes one) -> run installer ->
  re-detect. No consent -> print the manual link and exit cleanly.
- Windows: official `OllamaSetup.exe`. macOS/Linux: fetch the official `install.sh`, show its host, run on
  consent — **no silent pipe-to-shell**. Always offer "I'll install it myself" with the link.
- Reject any non-official URL/host/scheme. No elevation/sudo without telling the user. If verification is
  unavailable on a platform -> fall back to the manual link (never run unverified).

## Testing (`pytest -q`, offline — no real Ollama/network)

- `tests/test_ollama_ctl.py` — inject HTTP + `Popen` seams: `detect`, `ensure_serve` (owned-tracking:
  owned when we start it, not-owned when already up, `stop_owned` only kills owned), `pull` (NDJSON
  stream -> progress callback sequence), `list_models`/`ps`/`remove`.
- `tests/test_hardware.py` — inject the RAM/VRAM probes: `best_fit()` matrix (8 GB no-GPU -> 7b; 32 GB ->
  30b; 6 GB -> e4b but never the license-flagged one as auto-default; GPU-preferred path) + fail-soft
  default on probe error.
- `tests/test_onboarding.py` — drive the state machine with fakes: full path (missing -> install-consent ->
  serve -> pick -> pull -> ready), idempotent (healthy -> silent no-op), consent-decline -> clean exit (rc 0).
- `tests/test_progress.py` — pure render (frac -> bar string; cp1252 encodability).
- `tests/test_slash_ollama.py` — `/models`, `/model`, `/pull`, `/doctor`, `/serve` dispatch with a fake
  `ollama_ctl` (no network).
- `tests/test_install_security.py` — no consent -> no download; non-official host -> refused; verify-or-
  fallback path.
- Gate: `.venv/Scripts/python.exe -m pytest -q`. Follow-up: `aether-smoke` gains an `ollama-managed`
  check (or `/doctor` covers it).

## Out of scope (A)

Sub-projects B/C/D (agents, custom commands, multi-agent columns); the TS `aether-code` daemon-manage
mirror; GPU-accelerated model auto-tuning; non-Ollama local backends (llama.cpp/HF already exist in
`aether_context.local_llm` but are not wrapped by this onboarding in v1).
