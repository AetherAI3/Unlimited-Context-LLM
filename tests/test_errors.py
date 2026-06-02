"""Tests for typed errors — every error carries a non-empty .hint with the fix."""
import pytest

from aether_context.errors import (
    AetherContextError,
    PoolBudgetError,
    OllamaNotRunning,
    ModelNotPulled,
    BackendUnavailable,
    EncoderError,
    PoolCorrupt,
)

ALL_ERRORS = [
    AetherContextError,
    PoolBudgetError,
    OllamaNotRunning,
    ModelNotPulled,
    BackendUnavailable,
    EncoderError,
    PoolCorrupt,
]


# ---- hierarchy --------------------------------------------------------------
@pytest.mark.parametrize("cls", ALL_ERRORS)
def test_every_error_subclasses_base(cls):
    assert issubclass(cls, AetherContextError)
    assert issubclass(cls, Exception)


# ---- .hint contract ---------------------------------------------------------
@pytest.mark.parametrize("cls", ALL_ERRORS)
def test_every_error_has_a_nonempty_hint(cls):
    err = cls("something went wrong")
    assert isinstance(err.hint, str)
    assert err.hint.strip() != ""


@pytest.mark.parametrize("cls", ALL_ERRORS)
def test_message_is_preserved_in_str(cls):
    err = cls("boom")
    assert "boom" in str(err)


def test_custom_hint_overrides_default():
    err = OllamaNotRunning("daemon down", hint="do this instead")
    assert err.hint == "do this instead"


def test_hint_is_surfaced_in_str_for_actionable_errors():
    err = ModelNotPulled("qwen2.5 not pulled")
    # the fix command should be discoverable from the string form
    assert "ollama pull" in str(err).lower() or "ollama pull" in err.hint.lower()


def test_raise_and_catch_as_base():
    with pytest.raises(AetherContextError):
        raise PoolBudgetError("over budget")


def test_pool_budget_error_hint_mentions_pool():
    err = PoolBudgetError("over ceiling")
    assert "pool" in err.hint.lower()


def test_ollama_not_running_hint_mentions_serve():
    err = OllamaNotRunning("connection refused")
    assert "ollama serve" in err.hint.lower()
