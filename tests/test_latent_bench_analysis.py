# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Latent-transfer bench analysis: per-doc accuracy, paired bootstrap, verdict."""
from __future__ import annotations

import pytest

from bench.latent_transfer import analysis


def _records():
    return [
        {"doc_id": "d1", "arm": "capsule", "family": "yes_no", "correct": True},
        {"doc_id": "d1", "arm": "capsule", "family": "yes_no", "correct": False},
        {"doc_id": "d2", "arm": "capsule", "family": "yes_no", "correct": 1},
        {"doc_id": "d2", "arm": "blank", "family": "yes_no", "correct": 0},
        {"doc_id": "d2", "arm": "capsule", "family": "unseen_template", "correct": 0},
    ]


def test_per_doc_averages_within_arm_and_family():
    assert analysis.per_doc(_records(), "capsule", "yes_no") == {"d1": 0.5, "d2": 1.0}
    assert analysis.per_doc(_records(), "blank", "yes_no") == {"d2": 0.0}
    assert analysis.per_doc(_records(), "capsule", "unseen_template") == {"d2": 0.0}
    assert analysis.per_doc(_records(), "nope", "yes_no") == {}


def test_per_doc_custom_key():
    recs = [{"doc_id": "a", "arm": "x", "family": "f", "margin": 2}]
    assert analysis.per_doc(recs, "x", "f", key="margin") == {"a": 2.0}


def test_identical_arms_give_zero_difference():
    a = {"d1": 1.0, "d2": 0.0, "d3": 0.5}
    assert analysis.bootstrap_diff(a, dict(a)) == (0.0, 0.0, 0.0, 3)


def test_constant_gain_gives_exact_interval():
    a = {f"d{i}": 1.0 for i in range(5)}
    b = {f"d{i}": 0.0 for i in range(5)}
    assert analysis.bootstrap_diff(a, b) == (1.0, 1.0, 1.0, 5)


def test_same_seed_same_ci_and_interval_brackets_mean():
    a = {f"d{i}": (i * 7 % 11) / 11 for i in range(30)}
    b = {f"d{i}": (i * 5 % 13) / 13 for i in range(30)}
    first = analysis.bootstrap_diff(a, b, seed=3)
    assert analysis.bootstrap_diff(a, b, seed=3) == first
    mean, lo, hi, n = first
    assert n == 30 and lo <= mean <= hi
    assert analysis.bootstrap_diff(a, b, seed=4)[1:3] != (lo, hi)


def test_only_shared_doc_ids_are_used():
    a = {"d1": 1.0, "d2": 1.0, "only_a": 0.0}
    b = {"d1": 0.0, "d2": 0.0, "only_b": 1.0}
    assert analysis.bootstrap_diff(a, b) == (1.0, 1.0, 1.0, 2)


def test_no_shared_ids_raises():
    with pytest.raises(ValueError):
        analysis.bootstrap_diff({"a": 1.0}, {"b": 1.0})


def test_verdict_requires_every_lower_bound_to_clear_the_margin():
    ok = {"blank": (0.3, 0.1, 0.5, 24), "shuffled": (0.2, 0.05, 0.4, 24)}
    assert analysis.verdict(ok, 0.05) is True
    assert analysis.verdict(ok, 0.06) is False
    assert analysis.verdict({**ok, "mismatched": (0.1, -0.01, 0.2, 24)}, 0.0) is False
    assert analysis.verdict({}, 0.0) is False
