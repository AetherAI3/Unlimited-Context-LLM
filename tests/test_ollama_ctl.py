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
