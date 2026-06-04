"""Unit tests for the Aether coding agent — pure parts (no model, no network)."""

from __future__ import annotations

import pytest

from aether_agent.statusbar import render
from aether_agent import kernel
from aether_agent.tools import Tools, tool_schema


def test_statusbar_denominator_follows_pool():
    line5 = render(412_600_000, 5)
    line20 = render(1_800_000_000, 20)
    assert "local/cache" in line5
    assert "1.17B tokens" in line5  # 5 GB pool reach = 5 x 233M
    assert "4.66B tokens" in line20  # 20 GB pool reach
    assert "35.4%" in line5  # 412.6M / 1.165B


def test_statusbar_clamps_at_100():
    assert "100.0%" in render(10**12, 5)


def test_tests_pass_and_fail_count():
    assert kernel.tests_pass("[exit 0]\nall good")
    assert not kernel.tests_pass("[exit 1]\n2 failed")
    assert kernel.parse_fail_count("[exit 1]\n2 failed, 3 passed") == 2
    assert kernel.parse_fail_count("[exit 0]\nall passed") is None


def test_stuck_detector():
    s = kernel.StuckDetector(patience=2)
    assert not s.update(5)
    assert not s.update(5)  # stall 1
    assert s.update(5)  # stall 2 -> stuck
    s2 = kernel.StuckDetector(patience=2)
    assert not s2.update(5)
    assert not s2.update(3)  # improved -> reset, not stuck


def test_tool_schema_and_path_guard(tmp_path):
    names = {t["function"]["name"] for t in tool_schema()}
    assert {"read_file", "write_file", "run_shell", "run_tests", "repo_search", "git_commit"} <= names
    t = Tools(str(tmp_path))
    assert "[wrote" in t.write_file("a.txt", "hi")
    assert t.read_file("a.txt") == "hi"
    with pytest.raises(ValueError):
        t._safe("../escape")
