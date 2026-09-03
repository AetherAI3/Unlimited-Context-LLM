# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Terminal presentation seam for the ``aether-context`` console script.

Stdlib only — the core dependency contract is numpy and nothing else, so there is no
``rich``/``colorama`` here. This module owns *how* CLI output looks; the commands in
:mod:`aether_context.cli` own *what* it says.

Three degradations, all decided once at import of the writer and never at each call site:

``color``
    ANSI is emitted only when the stream is a real tty and the environment does not veto it
    (``NO_COLOR`` wins over everything, ``FORCE_COLOR`` turns it back on, ``TERM=dumb``
    disables). On Windows the VT100 mode is switched on explicitly; if that call fails the
    styling silently falls back to plain text rather than printing escape soup.

``unicode``
    Box-drawing and glyphs are only used when the stream's encoding can actually represent
    them. A ``cp1252`` console (still the Windows default for a redirected pipe) gets the
    ASCII table instead of a ``UnicodeEncodeError``.

``width``
    Rules and padding follow ``COLUMNS``/the real terminal size, clamped so output stays
    readable in a 40-column pane and does not sprawl on an ultrawide one.

Because color is off whenever stdout is not a tty, every command's output is plain text under
pytest, in a pipe, and in CI — so tests assert on words, never on escape sequences.
"""
from __future__ import annotations

import os
import shutil
import sys
from typing import IO, Final

#: Rules and headers never render narrower/wider than this, whatever the terminal reports.
_MIN_WIDTH: Final = 40
_MAX_WIDTH: Final = 78

# --- ANSI SGR codes ---------------------------------------------------------------------
_RESET: Final = "\033[0m"
_CODES: Final[dict[str, str]] = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "blue": "\033[34m",
}

#: Status markers. The bracket text is load-bearing: `doctor` has always reported `[ok]` /
#: `[fail]` / `[skip]`, scripts grep for it, and the suite asserts on it — so color and glyphs
#: decorate that text, they never replace it.
_MARKS: Final[dict[str, tuple[str, str, str]]] = {
    # key: (unicode glyph, ascii glyph, color)
    "ok": ("✓", "+", "green"),
    "warn": ("△", "!", "yellow"),
    "fail": ("✗", "x", "red"),
    "skip": ("·", "-", "dim"),
}


class Console:
    """A styled writer bound to one stream, with its capabilities resolved once.

    Construct with the stream you intend to write to (``sys.stdout`` for reports,
    ``sys.stderr`` for errors) so the capability probe matches the destination: piping stdout
    to a file must not strip color from an error still going to the terminal.
    """

    def __init__(self, stream: IO[str] | None = None) -> None:
        self.stream: IO[str] = stream if stream is not None else sys.stdout
        self.color: bool = _supports_color(self.stream)
        self.unicode: bool = _supports_unicode(self.stream)
        self.width: int = _terminal_width()

    # --- primitives ---------------------------------------------------------------------
    def style(self, text: str, *names: str) -> str:
        """Wrap ``text`` in the named SGR styles, or return it unchanged when color is off."""
        if not self.color or not names:
            return text
        codes = "".join(_CODES[n] for n in names if n in _CODES)
        return f"{codes}{text}{_RESET}" if codes else text

    def line(self, text: str = "") -> None:
        """Write one line to the bound stream, folded to ASCII when the stream can't encode.

        Every command's output funnels through here, so the fold covers prose too — not just
        the box-drawing set. Without it a ``cp1252`` pipe turns each em dash in a sentence into
        a ``?``, which looks like a bug in the tool rather than a limitation of the console.
        """
        print(_ascii_fold(text) if not self.unicode else text, file=self.stream)

    def glyph(self, kind: str) -> str:
        """The bare status glyph for ``kind`` (``ok``/``warn``/``fail``/``skip``)."""
        uni, ascii_, color = _MARKS[kind]
        return self.style(uni if self.unicode else ascii_, color)

    # --- composed output ----------------------------------------------------------------
    def banner(self, title: str, subtitle: str = "") -> None:
        """The product wordmark: a boxed title with an optional subtitle beneath it."""
        inner = min(self.width, _MAX_WIDTH) - 2
        if self.unicode:
            top, bottom, side = "╭" + "─" * inner + "╮", \
                "╰" + "─" * inner + "╯", "│"
        else:
            top = bottom = "+" + "-" * inner + "+"
            side = "|"
        self.line(self.style(top, "cyan"))
        self.line(
            self.style(side, "cyan")
            + self.style(title.center(inner), "bold", "cyan")
            + self.style(side, "cyan")
        )
        if subtitle:
            self.line(
                self.style(side, "cyan")
                + self.style(subtitle.center(inner), "dim")
                + self.style(side, "cyan")
            )
        self.line(self.style(bottom, "cyan"))

    def heading(self, text: str) -> None:
        """A section header: a blank line, then the title, then a rule the same width."""
        self.line()
        self.line(self.style(text, "bold"))
        char = "─" if self.unicode else "-"
        self.line(self.style(char * min(len(text), self.width), "dim"))

    def field(self, label: str, value: str, *, pad: int = 12) -> None:
        """An indented ``label   value`` row, labels left-aligned to a common column."""
        self.line(f"  {self.style(label.ljust(pad), 'dim')} {value}")

    def check(self, kind: str, text: str, fix: str = "") -> None:
        """One diagnostic row — ``glyph [kind] text`` — plus an optional indented fix line.

        ``kind`` is one of ``ok``/``warn``/``fail``/``skip`` and is printed literally inside
        the brackets, which is the contract the doctor's output has always had.
        """
        self.line(f"  {self.glyph(kind)} [{kind}] {text}")
        if fix:
            self.line(f"        {self.style('fix:', 'dim')} {self.style(fix, 'bold')}")

    def step(self, index: int, total: int, text: str) -> None:
        """A numbered wizard step header, e.g. ``[1/3] Pool size``."""
        self.line()
        self.line(f"{self.style(f'[{index}/{total}]', 'cyan')} {self.style(text, 'bold')}")

    def note(self, text: str) -> None:
        """A dimmed aside — context the user can skim past."""
        self.line(self.style(f"  {text}", "dim"))

    def command(self, text: str, comment: str = "") -> None:
        """A copy-pasteable command line, optionally trailed by a dimmed comment."""
        tail = f"   {self.style('# ' + comment, 'dim')}" if comment else ""
        self.line(f"  {self.style(text, 'bold', 'cyan')}{tail}")

    def bar(self, fraction: float, slots: int = 24) -> str:
        """A fixed-width meter for ``fraction`` in [0, 1], as a string (never printed)."""
        fraction = max(0.0, min(1.0, fraction))
        filled = round(fraction * slots)
        full, empty = ("█", "░") if self.unicode else ("#", ".")
        return self.style(full * filled, "cyan") + self.style(empty * (slots - filled), "dim")


# --- text folding ---------------------------------------------------------------------------
#: Typographic characters used in CLI prose, mapped to their ASCII equivalents.
_FOLD: Final[dict[int, str]] = str.maketrans({
    "—": "-", "–": "-", "≈": "~", "→": "->", "·": "-", "…": "...",
    "“": '"', "”": '"', "‘": "'", "’": "'", "×": "x",
})


def _ascii_fold(text: str) -> str:
    """Replace typographic characters with ASCII, dropping anything still unrepresentable.

    The translation table covers what the CLI's own prose uses; the final pass is a backstop
    for interpolated values (a path, a model name, an exception message) that could carry
    anything at all.
    """
    folded = text.translate(_FOLD)
    return folded.encode("ascii", "replace").decode("ascii")


# --- capability probes ----------------------------------------------------------------------
def _supports_color(stream: IO[str]) -> bool:
    """Decide whether to emit ANSI on ``stream``.

    ``NO_COLOR`` (any value, per no-color.org) disables unconditionally; ``FORCE_COLOR``
    re-enables it even off a tty, which is how CI logs keep their color. Otherwise color needs
    a real tty, a terminal that is not ``dumb``, and — on Windows — a console that accepts the
    VT100 mode switch.
    """
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if os.environ.get("TERM") == "dumb":
        return False
    try:
        if not stream.isatty():
            return False
    except (AttributeError, ValueError):  # detached/closed stream
        return False
    if sys.platform == "win32":
        return _enable_windows_vt()
    return True


def _enable_windows_vt() -> bool:
    """Turn on VT100 processing for the Windows console. False if it cannot be enabled.

    Windows Terminal and conhost on Windows 10+ support ANSI once
    ``ENABLE_VIRTUAL_TERMINAL_PROCESSING`` is set on the output handle. Older consoles reject
    the flag, and we would rather print clean text than raw escape bytes.
    """
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:  # noqa: BLE001 - any failure here just means "no color"
        return False


def _supports_unicode(stream: IO[str]) -> bool:
    """True when ``stream``'s encoding can represent the box-drawing/glyph set.

    Probed by actually encoding the widest character we use. A Windows console still running
    ``cp1252`` fails here and gets the ASCII table, instead of raising ``UnicodeEncodeError``
    in the middle of a report.
    """
    encoding = getattr(stream, "encoding", None) or "ascii"
    try:
        "─╭✓█".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def _terminal_width() -> int:
    """The usable output width, clamped to a readable range."""
    try:
        columns = shutil.get_terminal_size().columns
    except (OSError, ValueError):
        columns = _MAX_WIDTH
    return max(_MIN_WIDTH, min(columns, _MAX_WIDTH))
