"""End-to-end smoke for the ``aether`` terminal.

    python -m aether_agent.smoke        # or: aether-smoke

Runs the real surfaces and prints a PASS / SKIP / FAIL line per check:

  - ssrf guard   the web_fetch host guard refuses internal/metadata targets
  - ollama       local Ollama is reachable
  - local turn   a real local turn answers (LocalBrain -> Ollama)
  - web_search   DuckDuckGo search returns results
  - web_fetch    a public URL fetches to readable text
  - auth         current credential state
  - cloud turn   a real cloud turn answers (only when signed in)

Designed to be safe offline: anything that needs Ollama or the network that
isn't up degrades to SKIP (never a false FAIL). Exit code is 0 unless a hard
FAIL - so it's CI-usable as a gate once Ollama is provisioned.
"""

from __future__ import annotations

import os
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from aether_agent import adapter, web
from aether_agent.auth import FileTokenStore, auth_status
from aether_agent.brains import CloudBrain, LocalBrain
from aether_agent.config import load_config
from aether_agent.transport import ApiClient

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

# Keep a smoke turn short: the prompt should answer in one step with no tools.
_SMOKE_PROMPT = "Reply with exactly one word: pong"
_MAX_EVENTS = 200


@dataclass
class Check:
    name: str
    status: str
    detail: str


# --------------------------------------------------------------------------- #
# probes / helpers (monkeypatchable in tests)
# --------------------------------------------------------------------------- #
def _ollama_up(host: str, timeout: float = 4.0) -> bool:
    """True iff Ollama answers on ``{host}/api/tags`` within ``timeout``."""
    url = host.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (local host)
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _build_local_brain(host: str, model: str) -> LocalBrain:
    """A LocalBrain wired to a short-timeout Ollama in a throwaway cwd, capped to
    a couple of steps so a smoke turn can't run away."""
    llm = adapter.OllamaChat(model=model, host=host, timeout=120.0)
    cwd = tempfile.mkdtemp(prefix="aether-smoke-")
    return LocalBrain(model=model, cwd=cwd, max_steps=2, llm=llm)


def _drain_turn(brain: Any, max_events: int = _MAX_EVENTS) -> tuple[bool, str, str]:
    """Run a brain turn, return ``(ok, text, err)``. Stops on the terminal
    ``done`` / ``error`` event (or after ``max_events`` as a runaway guard)."""
    text = ""
    try:
        for i, ev in enumerate(brain.run(_SMOKE_PROMPT)):
            etype = ev.get("type")
            if etype == "monologue" and ev.get("text"):
                text = ev["text"]
            elif etype == "done":
                final = str(ev.get("text") or text or "")
                return (bool(final.strip()), final, "")
            elif etype == "error":
                return (False, "", str(ev.get("msg", "error")))
            if i + 1 >= max_events:
                return (bool(text.strip()), text, "no terminal event (capped)")
    except Exception as e:  # noqa: BLE001 - a smoke must never raise
        return (False, "", f"{type(e).__name__}: {e}")
    return (bool(text.strip()), text, "" if text.strip() else "no output")


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def check_ssrf() -> Check:
    """Security gate (offline-safe): the host guard must refuse internal and
    cloud-metadata targets. A FAIL here means the SSRF protection regressed."""
    targets = ["http://169.254.169.254/", "http://127.0.0.1/", "http://10.0.0.1/"]
    leaked = [u for u in targets if web.is_safe_url(u) is True]
    if leaked:
        return Check("ssrf guard", FAIL, f"guard allowed internal target(s): {leaked}")
    return Check("ssrf guard", PASS, "refused 169.254.169.254 / 127.0.0.1 / 10.0.0.1")


def check_ollama(host: str) -> Check:
    if _ollama_up(host):
        return Check("ollama", PASS, f"reachable at {host}")
    return Check("ollama", SKIP, f"not reachable at {host} - start it: `ollama serve`")


def check_local_turn(host: str, model: str) -> Check:
    if not _ollama_up(host):
        return Check("local turn", SKIP, "ollama not reachable - `ollama serve`")
    try:
        brain = _build_local_brain(host, model)
    except Exception as e:  # noqa: BLE001
        return Check("local turn", FAIL, f"could not build local brain: {e}")
    ok, text, err = _drain_turn(brain)
    if ok:
        return Check("local turn", PASS, f"{model} answered: {text.strip()[:60]!r}")
    hint = err or "empty answer"
    if "pull" in hint.lower() or "not" in hint.lower():
        hint += f"  (try: ollama pull {model})"
    return Check("local turn", FAIL, hint)


def check_web_search() -> Check:
    out = web.web_search("aether systems", limit=2)
    if out.startswith("[web_search error"):
        return Check("web_search", SKIP, f"no network? {out}")
    if not out.strip():
        return Check("web_search", FAIL, "empty result set")
    first = out.strip().splitlines()[0][:60]
    return Check("web_search", PASS, f"results returned: {first!r}")


def check_web_fetch() -> Check:
    out = web.web_fetch("https://example.com")
    if out.startswith("[web_fetch error"):
        return Check("web_fetch", SKIP, f"no network? {out}")
    if out.startswith("[web_fetch refused"):
        return Check("web_fetch", FAIL, f"public URL wrongly refused: {out}")
    if "example" not in out.lower():
        return Check("web_fetch", FAIL, "fetched text missing expected content")
    return Check("web_fetch", PASS, "fetched example.com -> readable text")


def check_auth(base_url: str) -> Check:
    st = auth_status(base_url, FileTokenStore())
    if st.get("logged_in"):
        return Check("auth", PASS, f"signed in ({st.get('token_type')}) @ {base_url}")
    return Check("auth", PASS, f"not signed in @ {base_url} (local-first)")


def check_cloud_turn(base_url: str, model: str) -> Check:
    st = auth_status(base_url, FileTokenStore())
    if not st.get("logged_in"):
        return Check("cloud turn", SKIP, "not signed in - `aether auth login`")
    api = ApiClient(base_url, FileTokenStore())
    brain = CloudBrain(api=api, model=model)
    ok, text, err = _drain_turn(brain)
    if ok:
        return Check("cloud turn", PASS, f"cloud answered: {text.strip()[:60]!r}")
    return Check("cloud turn", FAIL, err or "empty answer")


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def run_checks(results: Iterable[Check]) -> int:
    """Print the result table; return an exit code (1 iff any FAIL)."""
    results = list(results)
    sys.stdout.write("\naether terminal smoke\n")
    fails = 0
    for c in results:
        if c.status == FAIL:
            fails += 1
        sys.stdout.write(f"  {c.status:<4}  {c.name:<12}  {c.detail}\n")
    npass = sum(1 for c in results if c.status == PASS)
    nskip = sum(1 for c in results if c.status == SKIP)
    sys.stdout.write(f"\n{npass} pass | {nskip} skip | {fails} fail\n")
    if fails:
        sys.stdout.write("FAIL - a real surface is broken (see above).\n")
    elif nskip:
        sys.stdout.write("OK - no failures (skips need Ollama / network / sign-in).\n")
    else:
        sys.stdout.write("OK - every surface live and green.\n")
    return 1 if fails else 0


def main(argv: Optional[list[str]] = None) -> int:
    _ = argv  # no flags yet; signature kept for the console-script entry
    cfg = load_config()
    base_url = str(cfg.get("baseUrl") or "https://api.aethersystems.net/cloud")
    model = str(cfg.get("defaultModel") or "") or adapter.DEFAULT_MODEL
    host = os.environ.get("OLLAMA_HOST", adapter.DEFAULT_HOST)

    # Each check is wrapped so one exception can never abort the whole smoke.
    def _safe(fn, *a) -> Check:
        try:
            return fn(*a)
        except Exception as e:  # noqa: BLE001
            return Check(getattr(fn, "__name__", "check"), FAIL, f"crashed: {e}")

    results = [
        _safe(check_ssrf),
        _safe(check_ollama, host),
        _safe(check_local_turn, host, model),
        _safe(check_web_search),
        _safe(check_web_fetch),
        _safe(check_auth, base_url),
        _safe(check_cloud_turn, base_url, model),
    ]
    return run_checks(results)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
