# Plan — UCL native terminal (mirror of aether-code), local-Ollama-first + cloud-on-auth + web tools

**Date:** 2026-06-14
**Branch:** `feat/native-terminal-mirror`
**Sibling repo:** `aether-code` (TS) branch `feat/local-ollama-web-tools`

## Goal

Make `unlimited-context-llm` a **self-sufficient** open-source agentic coding terminal that is
**feature-identical** to `aether-code` (the TS terminal), without requiring Node:

- **Default backend = local Ollama** (`aether_agent.adapter.OllamaChat`, `localhost:11434`).
- **`aether auth login` → Aether cloud API** (`/cloud` base, Bearer token). Backend policy
  `auto` = *authed ? cloud : local*.
- **Native interactive REPL** mirroring aether-code's command surface + UX (splash, slash
  registry, pinned input, streaming, status bar, Ctrl-C).
- **Web tools** for the agent: `web_search` (DuckDuckGo lite, no key) + `web_fetch`
  (stdlib GET → readable text). Universal capability, works offline-of-cloud.

## Hard invariants (LOCKSTEP with aether-code)

These MUST stay identical in meaning across both repos or the bridge breaks:

1. **Protocol** — `aether_agent/protocol.py` `PROTOCOL_VERSION` == `brain_protocol.ts`
   `PROTOCOL_VERSION`. Bump **both** to `3` when adding web tools.
2. **Tool names** — the canonical `TOOLS` tuple is identical on both sides:
   `read_file, write_file, run_shell, run_tests, repo_search, git_commit, web_search, web_fetch`.
3. **Tool I/O shape** — shell-class tools return `"[exit N]\n<output>"`; output capped at 8000
   (head+tail). The grounding gate (`kernel.tests_pass`/`parse_fail_count`) depends on it.
4. **Status bar** — `aether_agent/statusbar.py` renders identically to `src/ui/statusbar.ts`
   (`TOKENS_PER_GB = 233_000_000`, same phases/kaomoji).
5. **Auth/config on-disk layout** — token at `~/.config/aether/.token` (0600), config at
   `~/.config/aether/config.json`; env overrides `AETHER_BASE_URL`, `AETHER_TOKEN`,
   `AETHER_CONFIG_DIR`, `AETHER_LOGIN_URL`. Same as aether-code so one credential serves both.

## New modules (`aether_agent/`)

| File | Mirror of (aether-code) | Responsibility |
|---|---|---|
| `web.py` | `core/web.ts` | `web_search(query)` DuckDuckGo lite HTML scrape (no key); `web_fetch(url)` GET → tag-stripped readable text, size-capped, SSRF-guarded (no localhost/RFC1918/file). |
| `config.py` *(agent cfg)* | `core/config.ts` | load/save `~/.config/aether/config.json` (`baseUrl, defaultModel, backend=auto\|local\|cloud, permissionMode, autoApply`). NOTE: distinct from `aether_context/config.py`. |
| `auth.py` | `core/auth.ts` | file token store (0600) + `login_with_password` (`POST /auth/login` → `session_token`) + status. |
| `transport.py` | `core/transport.ts` | urllib `ApiClient`: `POST /agent/chat/stream` (SSE) + `/agent/chat` fail-soft; Bearer; route constants. |
| `brains.py` | `core/brain*.ts` | `LocalBrain` (agentic chat+tool loop on `OllamaChat` + `Tools`) and `CloudBrain` (SSE via transport); `select_brain(ctx)` = auto policy. |
| `splash.py` | `ui/splash.ts` | banner/logo + version/model/backend line. |
| `slash.py` | `commands/slash.ts` | slash registry: `/help /models /model /agents /agent /tier /audit /clear /web /exit /quit`. |
| `repl.py` | `commands/chat.ts` | interactive REPL: splash → readline input (history) → slash or turn → stream render → status bar; Ctrl-C aborts turn, double Ctrl-C exits. |

## Extended modules

- `tools.py` — add `web_search`/`web_fetch` to `tool_schema()` + `Tools.execute` (delegate to `web.py`).
- `protocol.py` — add web tools to `TOOLS`; bump `PROTOCOL_VERSION` → 3; keep encode/decode mirror.
- `headless.py` — advertise web tools to the brain's tool schema (so the TS host path gets them too).
- `cli.py` — dispatch: bare `aether` → `repl`; `aether "<prompt>"` → one-shot chat; add `auth`,
  `models`, `config`; keep `code`, `brain`.
- `agent.py` — final-turn verifying test run + loop-breakers parity with `headless` (lift the
  stronger brain behaviors into the human path).

## TDD

- `pytest -q` is the gate. Write tests first (RED → GREEN) under `tests/`.
- New test files: `test_web_tools.py`, `test_agent_config.py`, `test_auth.py`, `test_transport.py`,
  `test_brains.py`, `test_slash.py`, `test_protocol_lockstep.py` (asserts version + TOOLS match the
  TS mirror), `test_cli_dispatch.py`.
- Network tools tested against a local stub server (`http.server`) — no live egress in tests.
- Keep stdlib-only (no new runtime deps); `urllib` for HTTP, `html.parser` for fetch extraction.

## Out of scope (v1)

Raw-mode pinned-input pixel parity (use readline; Windows uses basic loop), MCP client, swarm,
receipt/audit export beyond a read-only `/audit` slash, desktop packaging.
