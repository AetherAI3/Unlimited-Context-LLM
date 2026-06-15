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
