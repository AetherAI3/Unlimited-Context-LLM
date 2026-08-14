"""Render the Unlimited-Context-LLM social preview at GitHub's 1280x640 spec.

The previous asset was 537x405 — upscaled AND letterboxed into a 2:1 frame on
every link card, with text too small to survive the downscale. This is built at
the target size and checked at card size (~500px wide) before shipping.

Deterministic: no network, no image model. Run it again and you get the same
bytes, so the asset can be regenerated from source rather than hunted for.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

W, H = 1280, 640

# Palette lifted from the README badges so the card matches the repo it fronts.
GROUND = (9, 15, 22)
GROUND_LIFT = (14, 24, 34)
CYAN_BRIGHT = (34, 211, 238)
TEAL = (20, 184, 166)       # 14b8a6 — Python badge
SKY = (14, 165, 233)        # 0ea5e9 — Local-first badge
WHITE = (240, 247, 250)
DIM = (128, 152, 166)
DIMMER = (74, 94, 107)

F = "C:/Windows/Fonts/"


def black(s: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(F + "seguibl.ttf", s)      # Segoe UI Black


def semi(s: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(F + "seguisb.ttf", s)      # Segoe UI Semibold


def mono(s: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(F + "consolab.ttf", s)     # Consolas Bold


def _ground() -> Image.Image:
    """Vertical lift, then a soft cyan bloom screened in behind the headline."""
    img = Image.new("RGB", (W, H), GROUND)
    d = ImageDraw.Draw(img)
    for y in range(H):
        t = y / H
        d.line(
            [(0, y), (W, y)],
            fill=tuple(round(GROUND[i] + (GROUND_LIFT[i] - GROUND[i]) * t) for i in range(3)),
        )

    bloom = Image.new("RGB", (W, H), (0, 0, 0))
    bd = ImageDraw.Draw(bloom)
    cx, cy, r = 210, 290, 640
    steps = 40
    for i in range(steps):
        k = 1 - i / steps
        rad = int(r * (i + 1) / steps)
        v = int(30 * k * k)
        bd.ellipse(
            [cx - rad, cy - rad // 2, cx + rad, cy + rad // 2],
            fill=(v // 6, int(v * 0.75), v),
        )
    # Screen blend: light only adds, nothing darkens.
    return ImageChops.screen(img, bloom)


def _memory_motif(d: ImageDraw.ImageDraw, x: int, y: int, w: int) -> None:
    """The product in one glyph: a small bright native window, then the pool.

    The left segment is what the model holds on its own. Everything to its right
    is what the engine keeps reachable — same row, wildly different length.
    """
    h = 12
    d.rounded_rectangle([x, y, x + 74, y + h], radius=6, fill=CYAN_BRIGHT)
    seg_x = x + 90
    n, gap = 26, 5
    seg_w = (w - 90 - (n - 1) * gap) // n
    for i in range(n):
        k = 1 - (i / n) * 0.84
        col = tuple(round(SKY[c] * k + GROUND[c] * (1 - k)) for c in range(3))
        sx = seg_x + i * (seg_w + gap)
        d.rounded_rectangle([sx, y, sx + seg_w, y + h], radius=4, fill=col)


def build() -> Image.Image:
    img = _ground()
    d = ImageDraw.Draw(img)
    L = 84  # left margin

    # Eyebrow — names the repo without competing with the headline.
    d.text((L, 74), "AetherAI3  /  Unlimited-Context-LLM", font=mono(25), fill=DIM)

    # Headline. Two lines, large enough to hold at ~40% scale on a link card.
    d.text((L, 138), "A billion-token memory", font=black(88), fill=WHITE)
    d.text((L, 238), "for any local LLM", font=black(88), fill=CYAN_BRIGHT)

    # Accent rule tying the headline to the subline.
    d.rectangle([L, 358, L + 132, 364], fill=TEAL)

    # The three things a reader needs before deciding to click.
    d.text((L, 394), "Open-source  ·  local-first  ·  runs on your own Ollama",
           font=semi(34), fill=DIM)

    _memory_motif(d, L, 474, W - L * 2)

    d.text((L, 528), "Python 3.10+      Apache-2.0      No account, no cloud, no API key",
           font=semi(26), fill=DIMMER)

    return img


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    img = build()
    target = out_dir / "social-preview.png"
    img.save(target, "PNG", optimize=True)
    print(f"wrote {target.name}  {img.size[0]}x{img.size[1]}  "
          f"{target.stat().st_size / 1024:.0f} KB")

    # Legibility check at roughly the width a link card actually renders. Run
    # with --check and LOOK at it before shipping a change: the failure mode of
    # a social preview is text that only works at full size.
    if "--check" in sys.argv:
        check = out_dir / "social-preview.card-check.png"
        img.resize((500, 250), Image.LANCZOS).save(check, "PNG")
        print(f"wrote {check.name}  500x250  (throwaway; delete when done)")


if __name__ == "__main__":
    main()
