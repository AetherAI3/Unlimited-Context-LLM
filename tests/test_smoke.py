"""Tests for the aether terminal smoke runner (aether_agent.smoke).

Pure logic only — no real Ollama, no network. The IO checks are exercised via
injected fakes / monkeypatched probes so the suite stays offline and fast.
"""

from __future__ import annotations

from aether_agent import smoke
from aether_agent.smoke import Check, FAIL, PASS, SKIP


def test_run_checks_exit_zero_when_no_fail(capsys):
    rc = smoke.run_checks([Check("a", PASS, "ok"), Check("b", SKIP, "skipped")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "PASS" in out and "SKIP" in out


def test_run_checks_exit_one_on_any_fail():
    rc = smoke.run_checks([Check("a", PASS, "ok"), Check("b", FAIL, "boom")])
    assert rc == 1


def test_ssrf_check_passes_by_refusing_internal_targets():
    # Offline-safe: literal internal IPs need no DNS. The guard MUST refuse them.
    c = smoke.check_ssrf()
    assert c.status == PASS, c.detail


def test_drain_turn_collects_done_text():
    class FakeBrain:
        def run(self, task):
            yield {"type": "monologue", "text": "thinking"}
            yield {"type": "done", "text": "pong"}

    ok, text, err = smoke._drain_turn(FakeBrain())
    assert ok is True
    assert "pong" in text
    assert err == ""


def test_drain_turn_reports_error_event():
    class FakeBrain:
        def run(self, task):
            yield {"type": "error", "msg": "model exploded"}

    ok, text, err = smoke._drain_turn(FakeBrain())
    assert ok is False
    assert "model exploded" in err


def test_local_turn_skips_when_ollama_down(monkeypatch):
    monkeypatch.setattr(smoke, "_ollama_up", lambda host, timeout=4.0: False)
    c = smoke.check_local_turn("http://localhost:11434", "any-model")
    assert c.status == SKIP
    assert "ollama" in c.detail.lower()


def test_local_turn_passes_with_injected_brain(monkeypatch):
    monkeypatch.setattr(smoke, "_ollama_up", lambda host, timeout=4.0: True)

    class FakeBrain:
        def run(self, task):
            yield {"type": "done", "text": "pong"}

    monkeypatch.setattr(smoke, "_build_local_brain", lambda host, model: FakeBrain())
    c = smoke.check_local_turn("http://localhost:11434", "any-model")
    assert c.status == PASS


def test_auth_check_reports_logged_out(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    c = smoke.check_auth("https://api.aethersystems.net/cloud")
    assert c.status == PASS
    assert "not" in c.detail.lower() or "out" in c.detail.lower()


def test_cloud_turn_skips_when_logged_out(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    c = smoke.check_cloud_turn("https://api.aethersystems.net/cloud", "any-model")
    assert c.status == SKIP
    assert "sign in" in c.detail.lower() or "auth login" in c.detail.lower()


def test_main_returns_int_and_never_raises(monkeypatch, tmp_path):
    # Force everything offline: Ollama down, logged out. Web/SSRF run for real
    # but must not crash the runner. main() returns an int exit code.
    monkeypatch.setenv("AETHER_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(smoke, "_ollama_up", lambda host, timeout=4.0: False)
    rc = smoke.main([])
    assert isinstance(rc, int)
