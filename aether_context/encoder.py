"""B1 static encoder — Model2Vec-style, numpy-only, 256-dim, stateless.

Maps text to a fixed-size retrieval embedding the rest of the engine indexes and
searches over. The pipeline is the whole of it:

    text -> regex tokenize (lowercased word tokens)
         -> per-token row (hashed token -> seeded RNG -> deterministic unit row)
         -> mean-pool the rows
         -> L2-normalize
         -> 256-dim float32 unit vector

There is **no shipped multi-hundred-MB asset**: each token's row is *generated on
demand* from a hash of the token, seeded so the same token always yields the same
row (within a process and across processes / instances). Because shared tokens map
to the same row, mean-pooling gives real lexical cosine structure — two strings that
share tokens land closer than two that share none. Distinct tokens draw
near-orthogonal rows in 256-d, so unrelated text sits near cosine 0.

This produces the **256-dim retrieval embedding only** — it is purely a retrieval key
and not an attention mechanism. Swapping in a trained Model2Vec table later is a drop-in
replacement for ``_row_for_token`` / the lookup — the public API stays fixed.

Stateless and shared: a single ``StaticEncoder`` instance is safe to reuse; it holds
only a tiny in-memory cache of generated rows for speed.
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable

import numpy as np

from aether_context._log import get_logger
from aether_context.errors import EncoderError

logger = get_logger(__name__)

#: Version pin for the encoding scheme. Bump when the token->row generation, the
#: pooling, or the normalization changes (vectors produced by different versions are
#: not comparable). Retrieval indexes should record this alongside stored vectors.
ENCODER_VERSION: str = "static_v1"

#: Default embedding dimensionality (the 256-dim retrieval embedding).
DEFAULT_DIM: int = 256

#: Master seed mixed into every per-token hash so the whole table is reproducible but
#: namespaced to this encoder version.
_SEED_NAMESPACE: str = f"aether_context.encoder/{ENCODER_VERSION}"

#: Word tokenizer: runs of letters/digits/underscore. Punctuation and whitespace are
#: separators (so "refactor, the" and "refactor the" tokenize identically).
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase and split ``text`` into word tokens.

    Case-insensitive (lowercased first) and punctuation-insensitive (punctuation is a
    separator, never part of a token). Returns an empty list for empty / whitespace /
    punctuation-only input.
    """
    return _TOKEN_RE.findall(text.lower())


class StaticEncoder:
    """Deterministic, stateless, numpy-only text -> unit-vector encoder.

    Args:
        dim: Output dimensionality. Defaults to :data:`DEFAULT_DIM` (256). Must be a
            positive integer.

    The encoder is immutable after construction (only an internal row cache mutates,
    which is a pure performance detail and never affects output). Reuse one instance.
    """

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise EncoderError(
                f"StaticEncoder dim must be a positive int, got {dim!r}.",
                hint="Pass dim=256 (default) or another positive integer.",
            )
        self._dim: int = dim
        # token -> generated unit row; pure speed cache, never affects determinism.
        self._row_cache: dict[str, np.ndarray] = {}
        # A normalized empty/degenerate fallback so callers always get a unit vector.
        self._empty_vector: np.ndarray = self._make_empty_vector()

    @property
    def dim(self) -> int:
        """The output embedding dimensionality."""
        return self._dim

    @property
    def version(self) -> str:
        """The pinned encoder version string (:data:`ENCODER_VERSION`)."""
        return ENCODER_VERSION

    # -- public API ----------------------------------------------------------
    def encode(self, text: str) -> np.ndarray:
        """Encode one string into a ``(dim,)`` float32 unit vector.

        Empty / whitespace-only / punctuation-only input yields a deterministic unit
        fallback vector (never raises, never returns NaN). Non-string input raises
        :class:`~aether_context.errors.EncoderError`.
        """
        if not isinstance(text, str):
            raise EncoderError(
                f"encode() expects str, got {type(text).__name__}.",
                hint="Pass the text as a Python str (decode bytes first).",
            )
        tokens = _tokenize(text)
        if not tokens:
            return self._empty_vector.copy()
        rows = np.empty((len(tokens), self._dim), dtype=np.float32)
        for i, tok in enumerate(tokens):
            rows[i] = self._row_for_token(tok)
        pooled = rows.mean(axis=0)
        return self._l2_normalize(pooled)

    def encode_batch(self, texts: Iterable[str]) -> np.ndarray:
        """Encode many strings into an ``(N, dim)`` float32 matrix of unit rows.

        An empty iterable yields a ``(0, dim)`` array. Each row is produced exactly as
        :meth:`encode` would produce it, so ``encode_batch`` and per-item ``encode``
        agree. Non-iterable input raises
        :class:`~aether_context.errors.EncoderError`.
        """
        try:
            items = list(texts)
        except TypeError as exc:
            raise EncoderError(
                f"encode_batch() expects an iterable of str, got "
                f"{type(texts).__name__}.",
                hint="Pass a list/tuple of strings, e.g. encode_batch([\"a\", \"b\"]).",
            ) from exc
        if not items:
            return np.empty((0, self._dim), dtype=np.float32)
        out = np.empty((len(items), self._dim), dtype=np.float32)
        for i, text in enumerate(items):
            out[i] = self.encode(text)
        return out

    # -- internals -----------------------------------------------------------
    def _row_for_token(self, token: str) -> np.ndarray:
        """Return the deterministic unit row for ``token``, generating + caching it.

        The row is drawn from a per-token-seeded RNG (Gaussian) and L2-normalized, so
        each distinct token maps to a fixed unit vector that is near-orthogonal to
        other tokens' rows in high dimension. Shared tokens => identical rows => real
        lexical cosine structure after mean-pooling.
        """
        cached = self._row_cache.get(token)
        if cached is not None:
            return cached
        seed = self._seed_for_token(token)
        rng = np.random.default_rng(seed)
        row = rng.standard_normal(self._dim).astype(np.float32)
        row = self._l2_normalize(row)
        self._row_cache[token] = row
        return row

    @staticmethod
    def _seed_for_token(token: str) -> int:
        """Deterministic 64-bit seed for a token (stable across processes / hosts).

        Uses BLAKE2b over a versioned namespace + the token so the table is
        reproducible everywhere and tied to :data:`ENCODER_VERSION`. (Not security
        sensitive — purely a stable hash -> seed.)
        """
        digest = hashlib.blake2b(
            f"{_SEED_NAMESPACE}\x00{token}".encode("utf-8"), digest_size=8
        ).digest()
        return int.from_bytes(digest, "big", signed=False)

    def _make_empty_vector(self) -> np.ndarray:
        """Deterministic unit fallback for empty/degenerate input.

        Generated from a reserved sentinel token so it is a valid unit vector (callers
        can always assume unit norm) yet distinct from any real single-token vector.
        """
        seed = self._seed_for_token("\x00<empty>\x00")
        rng = np.random.default_rng(seed)
        row = rng.standard_normal(self._dim).astype(np.float32)
        return self._l2_normalize_raw(row)

    def _l2_normalize(self, vec: np.ndarray) -> np.ndarray:
        """L2-normalize a vector to unit length as float32.

        A zero / degenerate vector (norm too small to divide safely) falls back to the
        deterministic empty vector so the result is always a finite unit vector.
        """
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            return self._empty_vector.copy()
        return (vec / norm).astype(np.float32)

    @staticmethod
    def _l2_normalize_raw(vec: np.ndarray) -> np.ndarray:
        """L2-normalize without the empty-vector fallback (used to build it)."""
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            # Astronomically unlikely for a Gaussian draw; degrade to a fixed axis.
            out = np.zeros_like(vec, dtype=np.float32)
            out[0] = 1.0
            return out
        return (vec / norm).astype(np.float32)


__all__ = ["StaticEncoder", "ENCODER_VERSION", "DEFAULT_DIM"]
