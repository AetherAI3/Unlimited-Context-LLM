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
