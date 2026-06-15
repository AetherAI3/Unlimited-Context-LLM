# tests/test_progress.py
from aether_agent import progress


def test_bar_empty_full_and_clamp():
    assert progress.progress_bar(0.0, width=10) == "[----------] 0%"
    assert progress.progress_bar(1.0, width=10) == "[##########] 100%"
    assert progress.progress_bar(-5, width=10) == "[----------] 0%"
    assert progress.progress_bar(2, width=10) == "[##########] 100%"
    assert progress.progress_bar(0.5, width=10) == "[#####-----] 50%"


def test_fmt_bytes():
    assert progress.fmt_bytes(0) == "0 B"
    assert progress.fmt_bytes(1536) == "1.5 KB"
    assert progress.fmt_bytes(5 * 1024**3) == "5.0 GB"


def test_pull_line_and_step_and_spinner():
    line = progress.pull_line("qwen2.5-coder:7b", 2 * 1024**3, 4 * 1024**3)
    assert "qwen2.5-coder:7b" in line and "50%" in line
    assert progress.step("serve", ok=True) == "  [ok] serve"
    assert progress.step("serve", ok=False) == "  [x] serve"
    assert progress.step("serve") == "  ... serve"
    assert progress.spinner_frame(0) in "|/-\\"


def test_output_is_cp1252_safe():
    s = progress.progress_bar(0.5) + progress.step("x", True) + progress.pull_line("m", 1, 2)
    s.encode("cp1252")  # must not raise
