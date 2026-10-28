# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Latent-transfer report logic on synthetic item rows (no torch needed)."""
from __future__ import annotations

import pytest

pytest.importorskip("pydantic")

from bench.latent_transfer import report, tasks  # noqa: E402
from bench.latent_transfer.common import ARMS  # noqa: E402

N_EVAL = 4


@pytest.fixture(scope="module")
def ds():
    return tasks.build(7, 2, N_EVAL)


def _wrong(item: tasks.EvalItem, truth: str) -> str:
    return next(c for c in item.candidates if c != truth)


def _rows(ds: tasks.Dataset, skip: tuple[str, str] | None = None) -> list[dict]:
    docs = {d.doc_id: d for d in ds.eval_docs}
    rows = []
    for item in ds.eval_items:
        own = tasks.gold_under(item, docs[item.doc_id])
        preds = {
            "paired": own,
            "text_context": own,
            "blank": _wrong(item, own),
            "no_prefix": _wrong(item, own),
            "shuffled": tasks.gold_under(item, docs[ds.shuffle_source[item.doc_id]]),
            "mismatched": tasks.gold_under(item, ds.mismatched[item.doc_id]),
        }
        for arm in ARMS:
            if skip != (item.item_id, arm):
                rows.append({"item_id": item.item_id, "doc_id": item.doc_id, "family": item.family,
                             "attribute": item.attribute, "arm": arm, "pred": preds[arm],
                             "scores": [0.0, -1.0], "tie": False, "receipt_id": None})
    return rows


def test_score_rows_marks_correct_and_follow(ds):
    recs = report.score_rows(_rows(ds), ds)
    paired = [r for r in recs if r["arm"] == "paired"]
    assert all(r["correct"] for r in paired) and all("follow" not in r for r in paired)
    assert not any(r["correct"] for r in recs if r["arm"] in ("blank", "no_prefix"))
    assert all(r["follow"] for r in recs if r["arm"] in ("shuffled", "mismatched"))


def test_score_rows_rejects_missing_row(ds):
    item = ds.eval_items[0]
    with pytest.raises(ValueError, match="one row per"):
        report.score_rows(_rows(ds, skip=(item.item_id, "blank")), ds)


def test_analyse_means_diffs_and_follow(ds):
    res = report.analyse(report.score_rows(_rows(ds), ds))
    for fam in tasks.FAMILIES:
        assert res["arm_means"][fam]["paired"] == 1.0 and res["arm_means"][fam]["blank"] == 0.0
        blank = res["diffs"][fam]["paired-blank"]
        assert (blank["mean"], blank["lo"], blank["hi"], blank["n"]) == (1.0, 1.0, 1.0, N_EVAL)
        assert res["diffs"][fam]["text_context-paired"]["mean"] == 0.0
        assert res["follow"][fam]["shuffled"]["mean"] == 1.0 == res["follow"][fam]["mismatched"]["mean"]
        assert set(res["gate"][fam]) == {"paired-blank", "paired-shuffled", "paired-mismatched", "paired-no_prefix"}


def test_write_new_json_never_overwrites(tmp_path):
    first = report.write_new_json(tmp_path, "20260101T000000Z", {"a": 1})
    second = report.write_new_json(tmp_path, "20260101T000000Z", {"a": 2})
    assert first != second and first.read_text(encoding="utf-8") != second.read_text(encoding="utf-8")


def test_overall_verdict_follows_the_preregistered_rule():
    assert report.overall_verdict(True, False, True) == "SUPPORTED"
    assert report.overall_verdict(True, True, True) == "SUPPORTED, including the unseen task family"
    assert report.overall_verdict(True, True, False) == "NOT SUPPORTED"
    assert report.overall_verdict(False, True, True) == "NOT SUPPORTED"


def test_check_phase_agreement_rejects_drift():
    base = {"steps": 10, "lr": 0.001, "batch_size": 4, "train_seed": 0, "n_eval": 3}
    same = {p: {"config": dict(base)} for p in ("prepare", "train", "encode", "evaluate")}
    assert report.check_phase_agreement(same)["steps"] == 10
    drifted = dict(same, evaluate={"config": dict(base, steps=11)})
    with pytest.raises(RuntimeError, match="different configs"):
        report.check_phase_agreement(drifted)
