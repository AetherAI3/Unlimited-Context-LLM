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
