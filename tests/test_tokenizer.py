"""Tests for the token-count seam (chars/4 estimate + backend preference)."""
import pytest

from aether_context.tokenizer import estimate, from_backend, CHARS_PER_TOKEN


# ---- estimate: the backend-agnostic chars/4 rule ----------------------------
def test_chars_per_token_constant_is_four():
    # CHARS_PER_TOKEN constant — keep the budget math stable.
    assert CHARS_PER_TOKEN == 4


def test_empty_string_is_zero_tokens():
    assert estimate("") == 0


def test_matches_chars_over_four_on_ascii():
    assert estimate("a" * 4) == 1
    assert estimate("a" * 8) == 2
    assert estimate("hello world!") == len("hello world!") // 4


def test_monotonic_in_length():
    prev = -1
    for n in range(0, 200, 7):
        cur = estimate("x" * n)
        assert cur >= prev
        prev = cur


def test_short_nonempty_is_at_least_one_token():
    # a 1-3 char string still costs a token for budgeting (never undercount to zero)
    assert estimate("a") >= 1
    assert estimate("abc") >= 1


def test_estimate_rejects_non_string():
    with pytest.raises(TypeError):
        estimate(1234)  # type: ignore[arg-type]


# ---- from_backend: prefer a real tokenizer if present -----------------------
def test_from_backend_uses_count_tokens_when_available():
    class HasCounter:
        name = "fake"
        context_window = 2048

        def count_tokens(self, text: str) -> int:
            return 999

    f = from_backend(HasCounter())
    assert f("anything") == 999


def test_from_backend_falls_back_to_estimate_when_absent():
    class NoCounter:
        name = "bare"
        context_window = 2048

    f = from_backend(NoCounter())
    assert f("a" * 8) == estimate("a" * 8)


def test_from_backend_none_falls_back_to_estimate():
    f = from_backend(None)
    assert f("a" * 12) == estimate("a" * 12)


def test_from_backend_falls_back_when_backend_counter_raises():
    class BrokenCounter:
        name = "broken"
        context_window = 2048

        def count_tokens(self, text: str) -> int:
            raise RuntimeError("tokenizer exploded")

    # fail-soft: a broken backend tokenizer must not break budget math
    f = from_backend(BrokenCounter())
    assert f("a" * 8) == estimate("a" * 8)
