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
