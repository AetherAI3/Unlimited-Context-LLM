# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Tests for the terminal presentation seam (:mod:`aether_context.ui`).

Hermetic: no tty, no network, no subprocess. Streams are ``io.StringIO`` stand-ins whose
``isatty``/``encoding`` we control, which is exactly what the capability probes read.

The property that matters most here is the *degradation*: a console that cannot do color or
cannot encode box-drawing must get readable ASCII, never escape soup and never a
``UnicodeEncodeError`` mid-report.
"""
from __future__ import annotations

import pytest

from aether_context.ui import Console, _ascii_fold


class _Stream:
    """A capture stream with controllable ``isatty()`` and ``encoding``.

    Written from scratch rather than subclassing ``io.StringIO``, whose ``encoding`` attribute
    is read-only — and ``encoding`` is exactly what the unicode probe reads. ``print()`` only
    needs ``write``, so this is the whole contract.
    """

    def __init__(self, *, tty: bool = False, encoding: str = "utf-8") -> None:
        self._chunks: list[str] = []
        self._tty = tty
        self.encoding = encoding

    def write(self, text: str) -> int:
        self._chunks.append(text)
        return len(text)

    def isatty(self) -> bool:
        return self._tty

    def getvalue(self) -> str:
        return "".join(self._chunks)


# --- color probe -------------------------------------------------------------
def test_no_color_on_a_pipe(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-tty stream gets no ANSI, so piped/captured output stays plain text."""
    # Arrange
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)

    # Act
    console = Console(_Stream(tty=False))

    # Assert
    assert console.color is False
    assert console.style("hello", "bold", "cyan") == "hello"


def test_no_color_env_beats_a_real_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    """`NO_COLOR` disables styling even on an interactive terminal (no-color.org)."""
    # Arrange
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)

    # Act / Assert
    assert Console(_Stream(tty=True)).color is False


def test_force_color_enables_styling_off_a_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    """`FORCE_COLOR` turns ANSI back on for a pipe — how CI logs keep their color."""
    # Arrange
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")

    # Act
    console = Console(_Stream(tty=False))

    # Assert
    assert console.color is True
    assert "\033[" in console.style("hello", "bold")


def test_term_dumb_disables_color(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `TERM=dumb` terminal cannot render SGR, so we do not send any."""
    # Arrange
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("TERM", "dumb")

    # Act / Assert
    assert Console(_Stream(tty=True)).color is False


# --- unicode probe -----------------------------------------------------------
def test_cp1252_stream_falls_back_to_ascii_glyphs() -> None:
    """A cp1252 console cannot encode the box/glyph set, so it gets the ASCII table."""
    # Arrange
    console = Console(_Stream(encoding="cp1252"))

    # Act / Assert
    assert console.unicode is False
    assert console.glyph("ok") == "+"
    assert console.glyph("fail") == "x"


def test_utf8_stream_keeps_the_glyphs() -> None:
    """A UTF-8 stream renders the real check/cross glyphs."""
    # Arrange / Act
    console = Console(_Stream(encoding="utf-8"))

    # Assert
    assert console.unicode is True
    assert console.glyph("ok") == "✓"


def test_output_is_ascii_folded_for_a_non_unicode_stream() -> None:
    """Prose is folded too, not just glyphs — an em dash must not surface as `?`."""
    # Arrange
    stream = _Stream(encoding="cp1252")
    console = Console(stream)

    # Act
    console.line("local-first — numpy-only")

    # Assert
    written = stream.getvalue()
    assert "—" not in written
    assert "?" not in written
    assert "local-first - numpy-only" in written


def test_banner_encodes_cleanly_on_a_cp1252_stream() -> None:
    """The whole banner survives a cp1252 encode — the crash this fallback exists to prevent."""
    # Arrange
    stream = _Stream(encoding="cp1252")
    console = Console(stream)

    # Act
    console.banner("Unlimited Context", "aether-context 0.0.0")

    # Assert: round-trips through the console's real encoding without raising.
    stream.getvalue().encode("cp1252")


def test_ascii_fold_replaces_unmappable_characters() -> None:
    """Anything outside the table (an interpolated path, a model name) still degrades safely."""
    # Act / Assert
    assert _ascii_fold("≈ 5 GB → done") == "~ 5 GB -> done"
    assert "?" in _ascii_fold("модель")  # unmappable: replaced, never raised


# --- composed output ---------------------------------------------------------
def test_check_prints_the_bracket_kind_and_its_fix() -> None:
    """`[ok]`/`[fail]` bracket text is the doctor's long-standing contract; the fix rides along."""
    # Arrange
    stream = _Stream()
    console = Console(stream)

    # Act
    console.check("fail", "ollama daemon not reachable", "ollama serve")

    # Assert
    out = stream.getvalue()
    assert "[fail]" in out
    assert "ollama daemon not reachable" in out
    assert "ollama serve" in out


@pytest.mark.parametrize(
    ("fraction", "expected_filled"),
    [(0.0, 0), (0.5, 8), (1.0, 16), (-3.0, 0), (99.0, 16)],
)
def test_bar_is_clamped_and_proportional(fraction: float, expected_filled: int) -> None:
    """The meter fills proportionally and clamps, so an out-of-range ratio cannot corrupt it."""
    # Arrange
    console = Console(_Stream())

    # Act
    bar = console.bar(fraction, slots=16)

    # Assert
    assert len(bar) == 16
    assert bar.count("█") == expected_filled
