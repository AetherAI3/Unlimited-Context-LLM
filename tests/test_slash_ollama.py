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
