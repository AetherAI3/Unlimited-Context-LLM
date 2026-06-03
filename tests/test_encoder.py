# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Tests for the B1 static encoder (numpy-only, 256-dim, stateless).

Property tests for ``aether_context.encoder.StaticEncoder``:
- output shape (N, 256) and a single (256,) for ``encode``
- L2 unit-norm
- determinism (same text -> identical vector, across instances)
- lexical structure: similar strings score higher cosine than dissimilar ones,
  with a supervised similarity-margin gate (similar mean >= 0.3, dissimilar <= 0.1)
- empty / whitespace input handled without raising
- throughput smoke

All numpy-only, no network. Tests import the submodule directly (not via the
package ``__init__``), per the build contract.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from aether_context.encoder import (
    StaticEncoder,
    ENCODER_VERSION,
    DEFAULT_DIM,
)
from aether_context.errors import EncoderError


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two unit (or near-unit) vectors."""
    return float(np.dot(a, b))


# ---- version pin ------------------------------------------------------------
def test_encoder_version_pinned():
    assert ENCODER_VERSION == "static_v1"


def test_default_dim_is_256():
    assert DEFAULT_DIM == 256


# ---- shape ------------------------------------------------------------------
def test_encode_returns_256_dim_vector():
    enc = StaticEncoder()
    vec = enc.encode("the quick brown fox")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (256,)
    assert vec.dtype == np.float32


def test_encode_batch_returns_n_by_256_matrix():
    enc = StaticEncoder()
    texts = ["alpha beta", "gamma delta", "epsilon zeta"]
    mat = enc.encode_batch(texts)
    assert isinstance(mat, np.ndarray)
    assert mat.shape == (3, 256)
    assert mat.dtype == np.float32


def test_encode_batch_empty_list_returns_zero_by_256():
    enc = StaticEncoder()
    mat = enc.encode_batch([])
    assert mat.shape == (0, 256)
    assert mat.dtype == np.float32


def test_custom_dim_honored():
    enc = StaticEncoder(dim=128)
    assert enc.encode("hello world").shape == (128,)
    assert enc.encode_batch(["a", "b"]).shape == (2, 128)


# ---- unit-norm --------------------------------------------------------------
def test_encode_is_unit_norm():
    enc = StaticEncoder()
    vec = enc.encode("a moderately long sentence about software systems")
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


def test_encode_batch_rows_are_unit_norm():
    enc = StaticEncoder()
    mat = enc.encode_batch(["one fish", "two fish", "red fish blue fish"])
    norms = np.linalg.norm(mat, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


# ---- determinism ------------------------------------------------------------
def test_encode_is_deterministic_same_instance():
    enc = StaticEncoder()
    a = enc.encode("deterministic output please")
    b = enc.encode("deterministic output please")
    assert np.array_equal(a, b)


def test_encode_is_deterministic_across_instances():
    a = StaticEncoder().encode("stateless and reproducible")
    b = StaticEncoder().encode("stateless and reproducible")
    assert np.array_equal(a, b)


def test_encode_batch_matches_individual_encode():
    enc = StaticEncoder()
    texts = ["first chunk of text", "second different chunk"]
    mat = enc.encode_batch(texts)
    for i, t in enumerate(texts):
        assert np.allclose(mat[i], enc.encode(t), atol=1e-6)


def test_distinct_strings_produce_distinct_vectors():
    enc = StaticEncoder()
    a = enc.encode("the cat sat on the mat")
    b = enc.encode("quantum chromodynamics lattice gauge theory")
    assert not np.array_equal(a, b)


# ---- lexical structure: similar > dissimilar --------------------------------
def test_shared_tokens_lift_cosine():
    enc = StaticEncoder()
    base = enc.encode("authentication module refactor")
    similar = enc.encode("authentication module rewrite")  # 2/3 tokens shared
    dissimilar = enc.encode("banana smoothie recipe")       # 0 tokens shared
    assert _cos(base, similar) > _cos(base, dissimilar)


def test_identical_text_has_cosine_one():
    enc = StaticEncoder()
    a = enc.encode("identical text identical text")
    b = enc.encode("identical text identical text")
    assert np.isclose(_cos(a, b), 1.0, atol=1e-5)


def test_supervised_similarity_margin():
    """Hand-labeled similar vs dissimilar pairs: mean similar cosine must clear a
    margin over mean dissimilar cosine (>= 0.3 vs <= 0.1), per the build plan §12."""
    enc = StaticEncoder()
    similar_pairs = [
        ("the auth module handles login", "the auth module handles logout"),
        ("write tests for the parser", "write tests for the lexer"),
        ("database migration script", "database migration rollback"),
        ("render the user dashboard", "render the admin dashboard"),
        ("encode the context slice", "decode the context slice"),
    ]
    dissimilar_pairs = [
        ("the auth module handles login", "volcanic basalt cooling rates"),
        ("write tests for the parser", "ocean tides lunar gravity"),
        ("database migration script", "baroque harpsichord tuning"),
        ("render the user dashboard", "alpine glacier retreat survey"),
        ("encode the context slice", "medieval falconry guilds"),
    ]
    sim = np.mean([_cos(enc.encode(a), enc.encode(b)) for a, b in similar_pairs])
    dis = np.mean([_cos(enc.encode(a), enc.encode(b)) for a, b in dissimilar_pairs])
    assert sim >= 0.3, f"similar mean cosine {sim:.3f} below 0.3 floor"
    assert dis <= 0.1, f"dissimilar mean cosine {dis:.3f} above 0.1 ceiling"
    assert sim - dis >= 0.3, f"margin {sim - dis:.3f} below 0.3"


def test_tokenizer_is_case_insensitive():
    enc = StaticEncoder()
    a = enc.encode("Hello World")
    b = enc.encode("hello world")
    assert np.allclose(a, b, atol=1e-6)


def test_punctuation_does_not_change_token_set():
    enc = StaticEncoder()
    a = enc.encode("refactor, the, parser")
    b = enc.encode("refactor the parser")
    assert np.allclose(a, b, atol=1e-6)


# ---- empty / degenerate input ----------------------------------------------
def test_empty_string_returns_unit_vector_without_raising():
    enc = StaticEncoder()
    vec = enc.encode("")
    assert vec.shape == (256,)
    assert vec.dtype == np.float32
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


def test_whitespace_only_treated_like_empty():
    enc = StaticEncoder()
    a = enc.encode("")
    b = enc.encode("   \t\n  ")
    assert np.array_equal(a, b)


def test_punctuation_only_treated_like_empty():
    enc = StaticEncoder()
    a = enc.encode("")
    b = enc.encode("!@#$ %^&* ()")
    assert np.array_equal(a, b)


def test_batch_with_empty_entries_stays_unit_norm():
    enc = StaticEncoder()
    mat = enc.encode_batch(["", "real tokens here", "   "])
    norms = np.linalg.norm(mat, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


# ---- input validation -------------------------------------------------------
def test_non_string_input_raises_encoder_error():
    enc = StaticEncoder()
    with pytest.raises(EncoderError):
        enc.encode(12345)  # type: ignore[arg-type]


def test_bad_dim_raises_encoder_error():
    with pytest.raises(EncoderError):
        StaticEncoder(dim=0)
    with pytest.raises(EncoderError):
        StaticEncoder(dim=-1)


def test_encode_batch_non_iterable_raises_encoder_error():
    enc = StaticEncoder()
    with pytest.raises(EncoderError):
        enc.encode_batch(42)  # type: ignore[arg-type]


# ---- throughput smoke -------------------------------------------------------
def test_throughput_smoke():
    """A coarse sanity check that encoding is fast (not a strict benchmark)."""
    enc = StaticEncoder()
    text = "the quick brown fox jumps over the lazy dog " * 20  # ~180 tokens
    n = 200
    start = time.perf_counter()
    for _ in range(n):
        enc.encode(text)
    elapsed = time.perf_counter() - start
    # ~36k tokens total; should be well under a second on any machine.
    assert elapsed < 5.0
