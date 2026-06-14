# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Tests for bench/api_eval.py (dry-run only — no key, no network)."""
import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "api_eval", Path(__file__).resolve().parent.parent / "bench" / "api_eval.py"
)
api_eval = importlib.util.module_from_spec(_SPEC)
sys.modules["api_eval"] = api_eval  # so dataclasses can resolve the module's annotations
_SPEC.loader.exec_module(api_eval)


def _issues(n=12):
    return [
        {"number": i, "title": f"bug in module {i}", "body": f"detail {i}",
         "labels": ["bug" if i % 2 else "docs"], "state": "open"}
        for i in range(1, n + 1)
    ]


def test_tools_lookup_and_search():
    tb = api_eval.IssueTools(_issues())
    assert tb.lookup_issue(3)["labels"] == ["bug"]
    hits = tb.search_issues("module 4")
    assert any(h["number"] == 4 for h in hits)


def test_redundant_lookup_counted():
    tb = api_eval.IssueTools(_issues())
    tb.lookup_issue(1)
    tb.lookup_issue(1)
    assert tb.redundant == 1


def test_cost_math():
    c = api_eval.cost_usd({"prompt_tokens": 1_000_000, "completion_tokens": 0},
                          price_in=0.5, price_out=1.5)
    assert abs(c - 0.5) < 1e-9


def test_dry_run_produces_results(tmp_path):
    res = api_eval.run_eval(api_eval.Config(
        dry_run=True, issues_n=12, turns=6, arms=("off", "on_chain"),
        out_dir=tmp_path,
    ), corpus=_issues())
    assert set(res["arms"]) == {"off", "on_chain"}
    for arm in res["arms"].values():
        assert "cost_usd" in arm and "coherence" in arm and "tool_calls" in arm
        assert len(arm["series"]) > 0  # per-turn series for the graph
    assert (tmp_path / "api_eval_results.json").exists()
    assert (tmp_path / "api_eval_series.csv").exists()


def test_dry_run_thread_task(tmp_path):
    res = api_eval.run_eval(api_eval.Config(
        dry_run=True, issues_n=16, turns=8, arms=("off", "on_chain"),
        task="thread", out_dir=tmp_path,
    ), corpus=_issues(16))
    for arm in res["arms"].values():
        assert "coherence" in arm and "series" in arm and len(arm["series"]) > 0


def test_thread_score():
    assert api_eval._thread_score("issues 1, 2, 3", {1, 2, 3, 4}) == 0.75
    assert api_eval._thread_score("none", set()) == 0.0
