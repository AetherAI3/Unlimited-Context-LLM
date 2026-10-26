# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Latent-transfer bench task generator: determinism, splits, controls and gold labels."""
from __future__ import annotations

import json
import random

import pytest

pytest.importorskip("pydantic")

from bench.latent_transfer import tasks  # noqa: E402

SEED = 7


@pytest.fixture(scope="module")
def ds() -> tasks.Dataset:
    return tasks.build(SEED, 48, 24)


def _all_values() -> set[str]:
    return {v.lower() for vals in tasks.ATTRIBUTES.values() for v in vals}


def test_build_is_deterministic_for_one_seed_and_differs_for_another(ds):
    assert tasks.build(SEED, 48, 24) == ds
    assert tasks.dataset_digest(tasks.build(SEED, 48, 24)) == tasks.dataset_digest(ds)
    assert tasks.dataset_digest(tasks.build(SEED + 1, 48, 24)) != tasks.dataset_digest(ds)


def test_changing_n_eval_does_not_change_train_docs(ds):
    other = tasks.build(SEED, 48, 10)
    assert other.train_docs == ds.train_docs
    assert other.train_examples == ds.train_examples


def test_name_lists_are_sized_disjoint_single_capitalised_words():
    assert len(tasks.TRAIN_NAMES) == 48 and len(tasks.EVAL_NAMES) == 24
    assert len(set(tasks.TRAIN_NAMES)) == 48 and len(set(tasks.EVAL_NAMES)) == 24
    assert not set(tasks.TRAIN_NAMES) & set(tasks.EVAL_NAMES)
    for name in (*tasks.TRAIN_NAMES, *tasks.EVAL_NAMES):
        assert name.isalpha() and name[0].isupper() and name[1:].islower()
        assert name.lower() not in _all_values()


def test_train_and_eval_docs_use_disjoint_names(ds):
    train = {d.name for d in ds.train_docs}
    evals = {d.name for d in ds.eval_docs}
    assert train and evals and not train & evals
    assert [d.doc_id for d in ds.train_docs[:2]] == ["train-0000", "train-0001"]
    assert ds.eval_docs[0].doc_id == "eval-0000"


def test_no_training_example_uses_an_unseen_or_yes_no_template(ds):
    names = {d.doc_id: d.name for d in ds.train_docs}
    assert len(ds.train_examples) == len(ds.train_docs) * len(tasks.ATTRIBUTES)
    for ex in ds.train_examples:
        n = names[ex.doc_id]
        allowed = {q.format(n=n) for q in tasks.TRAIN_QUESTIONS[ex.attribute]}
        banned = {q.format(n=n) for qs in tasks.UNSEEN_QUESTIONS.values() for q in qs}
        banned |= {q.format(n=n, v=v) for q in tasks.YES_NO.values() for v in _all_values()}
        assert ex.question in allowed and ex.question not in banned
        doc = next(d for d in ds.train_docs if d.doc_id == ex.doc_id)
        assert ex.answer == doc.values[ex.attribute]


def test_eval_items_cover_four_per_family_per_doc_with_expected_shapes(ds):
    assert len(ds.eval_items) == len(ds.eval_docs) * 8
    ids = [it.item_id for it in ds.eval_items]
    assert len(set(ids)) == len(ids)
    docs = {d.doc_id: d for d in ds.eval_docs}
    for doc in ds.eval_docs:
        mine = [it for it in ds.eval_items if it.doc_id == doc.doc_id]
        for family in ("unseen_template", "yes_no"):
            assert sorted(it.attribute for it in mine if it.family == family) == sorted(
                tasks.ATTRIBUTES
            )
    for it in ds.eval_items:
        assert it.item_id == f"{it.doc_id}:{it.family}:{it.attribute}"
        name = docs[it.doc_id].name
        if it.family == "unseen_template":
            assert it.claim is None
            assert it.candidates == tuple(tasks.ATTRIBUTES[it.attribute])
            assert it.question in {q.format(n=name) for q in tasks.UNSEEN_QUESTIONS[it.attribute]}
        else:
            assert it.family == "yes_no"
            assert it.candidates == ("Yes", "No")
            assert it.claim in tasks.ATTRIBUTES[it.attribute]
            surface = it.claim if it.attribute == "city" else it.claim.lower()
            assert it.question == tasks.YES_NO[it.attribute].format(n=name, v=surface)


def test_yes_no_question_lowercases_non_city_values():
    item = tasks.build(SEED, 1, 2).eval_items
    pets = [i for i in item if i.family == "yes_no" and i.attribute == "pet"]
    assert pets and pets[0].claim.lower() in pets[0].question
    assert pets[0].claim not in pets[0].question  # capitalised claim, lower-cased surface


def test_shuffle_source_is_a_derangement_covering_every_eval_doc(ds):
    ids = [d.doc_id for d in ds.eval_docs]
    assert sorted(ds.shuffle_source) == sorted(ids)
    assert sorted(ds.shuffle_source.values()) == sorted(ids)
    assert all(k != v for k, v in ds.shuffle_source.items())


def test_shuffle_source_needs_at_least_two_eval_docs():
    with pytest.raises(ValueError):
        tasks.build(SEED, 4, 1)


def test_blank_keeps_name_and_drops_facts(ds):
    for doc in ds.eval_docs:
        blank = ds.blanks[doc.doc_id]
        assert blank.name == doc.name and blank.doc_id == doc.doc_id
        assert blank.text == f"Record for {doc.name}. No other details are on file."
        assert tasks.blank(doc) == blank


def test_mismatched_changes_every_value_and_keeps_name_order_variants(ds):
    for doc in ds.eval_docs:
        mm = ds.mismatched[doc.doc_id]
        assert (mm.name, mm.order, mm.variants) == (doc.name, doc.order, doc.variants)
        assert set(mm.values) == set(doc.values)
        for attr, value in doc.values.items():
            assert mm.values[attr] != value
            assert mm.values[attr] in tasks.ATTRIBUTES[attr]
        assert mm.text != doc.text
    one = tasks.mismatched(ds.eval_docs[0], random.Random(1))
    assert all(one.values[a] != ds.eval_docs[0].values[a] for a in one.values)


def test_gold_under_own_doc_matches_the_doc(ds):
    docs = {d.doc_id: d for d in ds.eval_docs}
    for it in ds.eval_items:
        doc = docs[it.doc_id]
        gold = tasks.gold_under(it, doc)
        if it.family == "unseen_template":
            assert gold == doc.values[it.attribute]
        else:
            assert gold == ("Yes" if it.claim == doc.values[it.attribute] else "No")
        assert gold in it.candidates


def test_gold_under_mismatched_doc_flips_yes_no_claims(ds):
    docs = {d.doc_id: d for d in ds.eval_docs}
    for it in ds.eval_items:
        if it.family != "yes_no":
            continue
        own = tasks.gold_under(it, docs[it.doc_id])
        other = tasks.gold_under(it, ds.mismatched[it.doc_id])
        if own == "Yes":
            assert other == "No"


def test_anchor_digests_are_stable_and_distinct(ds):
    digests = [tasks.anchor_digest(d) for d in ds.eval_docs]
    assert digests == [tasks.anchor_digest(d) for d in tasks.build(SEED, 48, 24).eval_docs]
    assert len(set(digests)) == len(digests)
    assert all(len(d) == 64 for d in digests)
    anchor = tasks.anchor(ds.eval_docs[0])
    assert set(anchor) == {"doc_id", "name", "values", "order", "variants", "text"}
    assert tasks.anchor_digest(ds.blanks["eval-0000"]) != digests[0]


def test_yes_no_truth_rate_is_balanced_at_200_docs():
    big = tasks.build(SEED, 4, 200)
    docs = {d.doc_id: d for d in big.eval_docs}
    yn = [i for i in big.eval_items if i.family == "yes_no"]
    assert len(yn) == 800
    rate = sum(tasks.gold_under(i, docs[i.doc_id]) == "Yes" for i in yn) / len(yn)
    assert 0.35 <= rate <= 0.65


def test_every_doc_text_contains_each_value_and_the_record_prefix(ds):
    for doc in (*ds.train_docs, *ds.eval_docs):
        assert doc.text.startswith(f"Record for {doc.name}. ")
        assert sorted(doc.order) == sorted(tasks.ATTRIBUTES)
        for attr, value in doc.values.items():
            surface = value if attr == "city" else value.lower()
            assert surface in doc.text
            assert tasks.FACTS[attr][doc.variants[attr]].format(n=doc.name, v=surface) in doc.text
    orders = {d.order for d in ds.eval_docs}
    assert len(orders) > 1  # field order is shuffled per document


def test_json_round_trip_is_equal_and_float_free(ds):
    payload = tasks.to_json(ds)
    wire = json.loads(json.dumps(payload))
    assert tasks.from_json(wire) == ds
    assert tasks.to_json(tasks.from_json(wire)) == payload
    assert len(tasks.dataset_digest(ds)) == 64
