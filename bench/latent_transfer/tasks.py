# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Synthetic single-person records, question families and controls for the latent-transfer bench.

Everything is deterministic from ``random.Random`` streams derived from one integer seed. Each
stream (train docs, eval docs, mismatches, permutation, items, training questions) is independent,
so changing ``n_eval`` never changes the training documents. The constants below are normative and
must match ``PREREGISTRATION.md``.
"""
from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from aether_context.contracts import digest

ATTRIBUTES = {
    "city": ("Paris", "London", "Tokyo", "Cairo", "Berlin", "Madrid", "Rome", "Sydney"),
    "colour": ("Red", "Blue", "Green", "Yellow", "Purple", "Orange", "Black", "White"),
    "pet": ("Cat", "Dog", "Horse", "Rabbit", "Parrot", "Turtle", "Goat", "Fox"),
    "job": ("Doctor", "Teacher", "Pilot", "Farmer", "Lawyer", "Chef", "Nurse", "Baker"),
}
FACTS = {  # two sentence variants per attribute; {v} is lower-cased except for cities
    "city": ("{n} lives in {v}.", "{n}'s home is in {v}."),
    "colour": ("{n}'s favourite colour is {v}.", "{n} likes the colour {v} best."),
    "pet": ("{n} has a pet {v}.", "{n} keeps a {v} at home."),
    "job": ("{n} works as a {v}.", "{n} earns a living as a {v}."),
}
TRAIN_QUESTIONS = {
    "city": ("Where does {n} live?", "Which city is {n}'s home?", "What city does {n} live in?"),
    "colour": (
        "What is {n}'s favourite colour?",
        "Which colour does {n} like best?",
        "Name {n}'s favourite colour.",
    ),
    "pet": (
        "What pet does {n} have?",
        "Which animal does {n} keep?",
        "What kind of pet does {n} own?",
    ),
    "job": (
        "What is {n}'s job?",
        "What does {n} do for work?",
        "Which profession does {n} have?",
    ),
}
UNSEEN_QUESTIONS = {  # never used in training
    "city": ("In which city would you find {n}'s house?", "{n} resides in what city?"),
    "colour": (
        "If {n} picked one colour, which would it be?",
        "{n} prefers what colour above all others?",
    ),
    "pet": ("{n} looks after which animal?", "Which creature lives with {n}?"),
    "job": ("What is {n}'s occupation?", "{n} is employed as what?"),
}
YES_NO = {  # unseen task family, never trained
    "city": "Is it true that {n} lives in {v}? Answer Yes or No.",
    "colour": "Is {n}'s favourite colour {v}? Answer Yes or No.",
    "pet": "Does {n} have a pet {v}? Answer Yes or No.",
    "job": "Does {n} work as a {v}? Answer Yes or No.",
}

FAMILIES = ("unseen_template", "yes_no")
YES_NO_CANDIDATES = ("Yes", "No")
YES_PROBABILITY = 0.5

TRAIN_NAMES = (
    "Alice", "Bruno", "Carla", "Dmitri", "Elena", "Felix", "Gloria", "Hector", "Irene", "Jonas",
    "Katya", "Leon", "Marta", "Nikolai", "Pavel", "Quentin", "Rosa", "Stefan", "Tessa", "Ulrich",
    "Vera", "Walter", "Xavier", "Yasmin", "Zoltan", "Andrei", "Beatrice", "Conrad", "Daphne",
    "Edmund", "Fiona", "Gustav", "Helena", "Ivan", "Judith", "Konrad", "Lena", "Magnus", "Nadia",
    "Oscar", "Petra", "Rafael", "Sabine", "Tobias", "Ursula", "Viktor", "Wanda", "Yusuf",
)
EVAL_NAMES = (
    "Ada", "Boris", "Clara", "Dario", "Esther", "Farid", "Greta", "Hugo", "Ingrid", "Jorge",
    "Kira", "Lukas", "Mara", "Nils", "Olga", "Pedro", "Rita", "Soren", "Talia", "Umar",
    "Vlad", "Willa", "Yara", "Zane",
)


@dataclass(frozen=True)
class Document:
    doc_id: str
    name: str
    values: dict[str, str]
    order: tuple[str, ...]
    variants: dict[str, int]
    text: str


@dataclass(frozen=True)
class TrainExample:
    doc_id: str
    attribute: str
    question: str
    answer: str


@dataclass(frozen=True)
class EvalItem:
    item_id: str
    doc_id: str
    family: str
    attribute: str
    question: str
    candidates: tuple[str, ...]
    claim: str | None


@dataclass(frozen=True)
class Dataset:
    train_docs: tuple[Document, ...]
    eval_docs: tuple[Document, ...]
    blanks: dict[str, Document]
    mismatched: dict[str, Document]
    shuffle_source: dict[str, str]
    train_examples: tuple[TrainExample, ...]
    eval_items: tuple[EvalItem, ...]


def _surface(attribute: str, value: str) -> str:
    """Text form of a value: cities keep their capital, everything else is lower-cased."""
    return value if attribute == "city" else value.lower()


def _render(
    name: str, values: Mapping[str, str], order: tuple[str, ...], variants: Mapping[str, int]
) -> str:
    sentences = [FACTS[a][variants[a]].format(n=name, v=_surface(a, values[a])) for a in order]
    return f"Record for {name}. " + " ".join(sentences)


def _stream(seed: int, label: str) -> random.Random:
    return random.Random(f"{seed}:{label}")


def _name_cycle(rng: random.Random, names: tuple[str, ...], count: int) -> list[str]:
    """Every name once per pass (shuffled), repeating only when ``count`` exceeds the list."""
    out: list[str] = []
    while len(out) < count:
        out.extend(rng.sample(names, len(names)))
    return out[:count]


def _make_docs(
    rng: random.Random, prefix: str, names: tuple[str, ...], count: int
) -> tuple[Document, ...]:
    docs = []
    for i, name in enumerate(_name_cycle(rng, names, count)):
        order = tuple(rng.sample(list(ATTRIBUTES), len(ATTRIBUTES)))
        variants = {a: rng.randrange(2) for a in ATTRIBUTES}
        values = {a: rng.choice(ATTRIBUTES[a]) for a in ATTRIBUTES}
        docs.append(
            Document(
                f"{prefix}-{i:04d}",
                name,
                values,
                order,
                variants,
                _render(name, values, order, variants),
            )
        )
    return tuple(docs)


def blank(doc: Document) -> Document:
    """Control: same name, no facts."""
    return Document(
        doc.doc_id, doc.name, {}, (), {}, f"Record for {doc.name}. No other details are on file."
    )


def mismatched(doc: Document, rng: random.Random) -> Document:
    """Control: same name/order/variants, every value swapped for a different one."""
    values = {a: rng.choice([v for v in ATTRIBUTES[a] if v != doc.values[a]]) for a in doc.order}
    return Document(
        doc.doc_id,
        doc.name,
        values,
        doc.order,
        dict(doc.variants),
        _render(doc.name, values, doc.order, doc.variants),
    )


def anchor(doc: Document) -> dict[str, Any]:
    """JSON-safe, float-free description of a document (input to ``anchor_digest``)."""
    return {
        "doc_id": doc.doc_id,
        "name": doc.name,
        "values": dict(doc.values),
        "order": list(doc.order),
        "variants": dict(doc.variants),
        "text": doc.text,
    }


def anchor_digest(doc: Document) -> str:
    return digest(anchor(doc))


def gold_under(item: EvalItem, doc: Document) -> str:
    """The correct answer to ``item`` if ``doc`` were the truth."""
    actual = doc.values[item.attribute]
    if item.family == "unseen_template":
        return actual
    return "Yes" if actual == item.claim else "No"


def _derangement(rng: random.Random, doc_ids: list[str]) -> dict[str, str]:
    """Seeded cyclic shift by one over a shuffled order: no fixed point, covers every id."""
    if not doc_ids:
        return {}
    if len(doc_ids) < 2:
        raise ValueError("a shuffled control needs at least two eval docs")
    shuffled = rng.sample(doc_ids, len(doc_ids))
    return {shuffled[i]: shuffled[(i + 1) % len(shuffled)] for i in range(len(shuffled))}


def _train_examples(rng: random.Random, docs: tuple[Document, ...]) -> tuple[TrainExample, ...]:
    return tuple(
        TrainExample(d.doc_id, a, rng.choice(TRAIN_QUESTIONS[a]).format(n=d.name), d.values[a])
        for d in docs
        for a in ATTRIBUTES
    )


def _eval_items(rng: random.Random, docs: tuple[Document, ...]) -> tuple[EvalItem, ...]:
    items = []
    for doc in docs:
        for a in ATTRIBUTES:
            question = rng.choice(UNSEEN_QUESTIONS[a]).format(n=doc.name)
            items.append(
                EvalItem(
                    f"{doc.doc_id}:unseen_template:{a}",
                    doc.doc_id,
                    "unseen_template",
                    a,
                    question,
                    tuple(ATTRIBUTES[a]),
                    None,
                )
            )
        for a in ATTRIBUTES:
            truth = doc.values[a]
            if rng.random() < YES_PROBABILITY:
                claim = truth
            else:
                claim = rng.choice([v for v in ATTRIBUTES[a] if v != truth])
            question = YES_NO[a].format(n=doc.name, v=_surface(a, claim))
            items.append(
                EvalItem(
                    f"{doc.doc_id}:yes_no:{a}",
                    doc.doc_id,
                    "yes_no",
                    a,
                    question,
                    YES_NO_CANDIDATES,
                    claim,
                )
            )
    return tuple(items)


def build(seed: int, n_train: int, n_eval: int) -> Dataset:
    """Generate the full dataset. Names repeat (shuffled passes) if a split exceeds its list."""
    train_docs = _make_docs(_stream(seed, "train"), "train", TRAIN_NAMES, n_train)
    eval_docs = _make_docs(_stream(seed, "eval"), "eval", EVAL_NAMES, n_eval)
    mm_rng = _stream(seed, "mismatch")
    return Dataset(
        train_docs=train_docs,
        eval_docs=eval_docs,
        blanks={d.doc_id: blank(d) for d in eval_docs},
        mismatched={d.doc_id: mismatched(d, mm_rng) for d in eval_docs},
        shuffle_source=_derangement(_stream(seed, "perm"), [d.doc_id for d in eval_docs]),
        train_examples=_train_examples(_stream(seed, "train_q"), train_docs),
        eval_items=_eval_items(_stream(seed, "items"), eval_docs),
    )


def _doc_from_json(d: Mapping[str, Any]) -> Document:
    return Document(
        d["doc_id"],
        d["name"],
        dict(d["values"]),
        tuple(d["order"]),
        {k: int(v) for k, v in d["variants"].items()},
        d["text"],
    )


def to_json(ds: Dataset) -> dict[str, Any]:
    """JSON-safe, float-free form (strings, ints, lists, dicts) that ``from_json`` inverts."""
    return {
        "train_docs": [anchor(d) for d in ds.train_docs],
        "eval_docs": [anchor(d) for d in ds.eval_docs],
        "blanks": {k: anchor(v) for k, v in ds.blanks.items()},
        "mismatched": {k: anchor(v) for k, v in ds.mismatched.items()},
        "shuffle_source": dict(ds.shuffle_source),
        "train_examples": [
            {
                "doc_id": e.doc_id,
                "attribute": e.attribute,
                "question": e.question,
                "answer": e.answer,
            }
            for e in ds.train_examples
        ],
        "eval_items": [
            {
                "item_id": i.item_id,
                "doc_id": i.doc_id,
                "family": i.family,
                "attribute": i.attribute,
                "question": i.question,
                "candidates": list(i.candidates),
                "claim": i.claim,
            }
            for i in ds.eval_items
        ],
    }


def from_json(d: Mapping[str, Any]) -> Dataset:
    return Dataset(
        train_docs=tuple(_doc_from_json(x) for x in d["train_docs"]),
        eval_docs=tuple(_doc_from_json(x) for x in d["eval_docs"]),
        blanks={k: _doc_from_json(v) for k, v in d["blanks"].items()},
        mismatched={k: _doc_from_json(v) for k, v in d["mismatched"].items()},
        shuffle_source=dict(d["shuffle_source"]),
        train_examples=tuple(TrainExample(**e) for e in d["train_examples"]),
        eval_items=tuple(
            EvalItem(
                i["item_id"],
                i["doc_id"],
                i["family"],
                i["attribute"],
                i["question"],
                tuple(i["candidates"]),
                i["claim"],
            )
            for i in d["eval_items"]
        ),
    )


def dataset_digest(ds: Dataset) -> str:
    return digest(to_json(ds))
