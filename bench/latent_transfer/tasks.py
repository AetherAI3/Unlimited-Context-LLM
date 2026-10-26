# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Synthetic single-person records, question families and controls for the latent-transfer bench.

Everything is deterministic from ``random.Random`` streams derived from one integer seed. Each
stream (train docs, eval docs, mismatches, permutation, items, training questions) is independent,
so changing ``n_eval`` never changes the training documents. The constants below are normative and
must match ``PREREGISTRATION.md``.

Reproducibility note: Python only guarantees the ``random.random()`` sequence of a seeded
generator across versions; ``choice``/``sample``/``shuffle`` may change between releases. The
committed dataset digest (``dataset_digest``) is the source of truth, not the seed.
"""
from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
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
TRUE_CLAIMS_PER_DOC = 2  # of the 4 yes/no items per eval doc, exactly this many are true

TRAIN_NAMES = (
    "Aaron", "Abigail", "Adrian", "Agnes", "Alan", "Albert", "Alexa", "Alfred", "Alice", "Alina",
    "Amelia", "Anders", "Andrei", "Angela", "Anika", "Anton", "Arjun", "Astrid", "Aurelio", "Axel",
    "Bastian", "Beatrice", "Benedict", "Bernard", "Bianca", "Birgit", "Bjorn", "Blanca", "Bruno",
    "Bertram", "Camila", "Carla", "Carmen", "Cecilia", "Cedric", "Celeste", "Charlotte",
    "Christoph", "Cornelia", "Conrad", "Cyrus", "Dagmar", "Damian", "Daniela", "Darius", "Davina",
    "Dmitri", "Dominic", "Donato", "Dorothea", "Dylan", "Eamon", "Edmund", "Edith", "Eduardo",
    "Elena", "Eleanor", "Elias", "Elisa", "Emil", "Emilia", "Enrique", "Erik", "Esme", "Eugene",
    "Fabian", "Farah", "Felicia", "Felix", "Fernando", "Filip", "Fiona", "Florian", "Frances",
    "Franz", "Freya", "Gabriel", "Gaspar", "Gemma", "Georg", "Gerald", "Gianna", "Gilbert",
    "Gloria", "Gregor", "Gunnar", "Gustav", "Giselle", "Hannah", "Hans", "Harriet", "Hector",
    "Heidi", "Helena", "Henrik", "Herman", "Hilda", "Horace", "Hana", "Ian", "Ibrahim", "Igor",
    "Ilse", "Imani", "Irene", "Isabel", "Isidor", "Ivan", "Ilana", "Jacqueline", "Jakob", "Janet",
    "Jasper", "Javier", "Jelena", "Joachim", "Johanna", "Jonas", "Josephine", "Judith", "Julian",
    "Juliet", "Jurgen", "Kamal", "Karen", "Karl", "Katarina", "Katya", "Kenji", "Klara", "Konrad",
    "Kristof", "Kurt", "Lars", "Laura", "Lazlo", "Lena", "Leon", "Leonora", "Liam", "Lidia",
    "Linus", "Livia", "Lorenzo", "Lucia", "Ludwig", "Luther", "Magnus", "Malik", "Manuel",
    "Margit", "Marta", "Matteo", "Mathilde", "Maxim", "Melina", "Miriam", "Mirko", "Monica",
    "Moritz", "Nadia", "Natalia", "Nathan", "Nestor", "Nicola", "Nikolai", "Nina", "Noah", "Nora",
    "Norbert", "Octavia", "Olaf", "Oliver", "Orson", "Oscar", "Osvaldo", "Otto", "Odette", "Pablo",
    "Paloma", "Pascal", "Patrice", "Paula", "Pavel", "Petra", "Philip", "Pilar", "Priya",
    "Quentin", "Rafael", "Ramon", "Raquel", "Reinhard", "Renata", "Ricardo", "Rolf", "Roman",
    "Rosa", "Rosalind", "Rufus", "Sabine", "Salma", "Samuel", "Santiago", "Sara", "Sebastian",
    "Selma", "Serena", "Sergei", "Silvia", "Simone", "Sofia", "Stefan", "Stella", "Svetlana",
    "Tamara", "Tariq", "Teodor", "Tessa", "Theo", "Theresa", "Thomas", "Tobias", "Tomas", "Trudy",
    "Ulrich", "Ulla", "Uri", "Ursula", "Valentin", "Valeria", "Vera", "Viktor", "Vincent",
    "Vivian", "Vesna", "Walter", "Wanda", "Wendell", "Wilhelm", "Winifred", "Wolfram", "Xavier",
    "Ximena", "Yasmin", "Yusuf", "Yolanda", "Yuri", "Zoltan", "Zora", "Zachary",
)
EVAL_NAMES = (
    "Ada", "Adela", "Adnan", "Alberto", "Alma", "Amir", "Anselm", "Arvid", "Aziz", "Alexei",
    "Anneliese", "Barnaby", "Beata", "Bogdan", "Boris", "Brigitte", "Bruna", "Bartosz", "Caleb",
    "Camille", "Casimir", "Clara", "Claudio", "Constance", "Corinna", "Cosmo", "Callum", "Dario",
    "Delia", "Desmond", "Dieter", "Dina", "Dorian", "Dunstan", "Dalia", "Edda", "Egon", "Elif",
    "Elmer", "Emeric", "Ernesto", "Esther", "Evander", "Ezra", "Eliska", "Farid", "Fatima",
    "Fenella", "Fergus", "Finn", "Flavia", "Frieda", "Fritz", "Gael", "Gideon", "Greta",
    "Gretchen", "Guido", "Gwen", "Gerda", "Gaia", "Hadrian", "Halima", "Hartmut", "Hedda", "Hugo",
    "Humberto", "Hanne", "Ida", "Ignacio", "Ingrid", "Iona", "Irina", "Isolde", "Ivo", "Isaac",
    "Jaime", "Jarek", "Joaquin", "Jorge", "Jovan", "Judit", "Jonathan", "Jelka", "Kasimir",
    "Katrin", "Kira", "Kleo", "Kofi", "Kostas", "Kristina", "Kaspar", "Lamar", "Leopold", "Leila",
    "Lorna", "Lothar", "Lukas", "Luna", "Lucian", "Larissa", "Mara", "Marcel", "Mario", "Marit",
    "Mehmet", "Mercedes", "Mikhail", "Milena", "Mina", "Murat", "Nabil", "Natasha", "Neville",
    "Nico", "Nigel", "Nils", "Noemi", "Nuno", "Olga", "Omar", "Ondrej", "Orla", "Osman", "Ottilie",
    "Paolo", "Pedro", "Penelope", "Percival", "Petr", "Phoebe", "Placido", "Prisca", "Radek",
    "Raimund", "Regina", "Rita", "Rodrigo", "Rosalie", "Rudolf", "Ruth", "Reza", "Saskia",
    "Sergio", "Silas", "Soren", "Sven", "Sybil", "Szilvia", "Samira", "Sandor", "Talia", "Tamsin",
    "Tancred", "Teresa", "Tibor", "Tilda", "Timur", "Tristan", "Tove", "Udo", "Ugo", "Umar", "Una",
    "Uwe", "Valentina", "Vasco", "Vlad", "Vigdis", "Veronika", "Wenzel", "Wilma", "Willa",
    "Wiktor", "Wojtek", "Yara", "Yvonne", "Yannick", "Yevgenia", "Zane", "Zdenek", "Zeno",
    "Zinaida", "Zofia", "Zuzana", "Abel", "Ambrose", "Anatole", "Bela", "Benno", "Borys", "Cato",
    "Cyril", "Dennis", "Ellis", "Emmett", "Eskil", "Fabio", "Gerhard", "Gregoire", "Hubert",
    "Idris", "Ilya", "Ismael", "Jerome", "Joris", "Kajetan", "Lazar", "Leandro", "Marek",
    "Matthias", "Milo", "Nikita", "Orlando", "Ottokar", "Pierre", "Rainer", "Rasmus", "Reuben",
    "Stanislav", "Teodoro", "Ulf", "Viggo", "Waldemar", "Yaroslav", "Zacharias", "Zenon", "Aline",
    "Berta", "Dorota", "Elsa", "Hedy", "Inge", "Jana", "Lotte", "Magda", "Olivia",
)


@dataclass(frozen=True)
class Document:
    doc_id: str
    name: str
    values: Mapping[str, str]
    order: tuple[str, ...]
    variants: Mapping[str, int]
    text: str

    def __post_init__(self) -> None:
        # Read-only views so a document cannot drift from its text and anchor digest.
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
        object.__setattr__(self, "order", tuple(self.order))
        object.__setattr__(self, "variants", MappingProxyType(dict(self.variants)))


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


def _make_docs(rng: random.Random, prefix: str, names: list[str]) -> tuple[Document, ...]:
    """Documents for ``names``; ``rng`` only ever draws values, never names."""
    docs = []
    for i, name in enumerate(names):
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


def _eval_names(rng: random.Random, count: int) -> list[str]:
    """Unique eval names: a shuffled prefix, so the first k names do not depend on ``count``."""
    if count > len(EVAL_NAMES):
        raise ValueError(f"n_eval {count} exceeds the {len(EVAL_NAMES)} available eval names")
    pool = list(EVAL_NAMES)
    rng.shuffle(pool)
    return pool[:count]


def _train_names(rng: random.Random, count: int) -> list[str]:
    """Train names are drawn independently per doc and may repeat."""
    return [rng.choice(TRAIN_NAMES) for _ in range(count)]


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
    if item.attribute not in doc.values:
        raise ValueError(f"doc {doc.doc_id} has no value for attribute {item.attribute!r}")
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
        true_attrs = set(rng.sample(list(ATTRIBUTES), TRUE_CLAIMS_PER_DOC))
        for a in ATTRIBUTES:
            truth = doc.values[a]
            if a in true_attrs:
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
    """Generate the full dataset.

    Eval names are unique (``ValueError`` if ``n_eval`` exceeds ``len(EVAL_NAMES)``); train names
    may repeat. Names come from their own streams, so document values do not depend on them.
    """
    train_names = _train_names(_stream(seed, "names_train"), n_train)
    eval_names = _eval_names(_stream(seed, "names_eval"), n_eval)
    train_docs = _make_docs(_stream(seed, "train"), "train", train_names)
    eval_docs = _make_docs(_stream(seed, "eval"), "eval", eval_names)
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
