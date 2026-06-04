"""
Session status bar — live pool-fill. The denominator is the disk size the user
selected in Unlimited Context (`reach_tokens(pool_gb) = pool_gb x 233M`), so the
bar always reads against THEIR chosen pool, not a fixed number. Cheap: two ints
+ a string, off the hot path. Clamps at 100% (witnesses fade stale slices to
hold the line — it recycles, never hard-stops).
"""

from __future__ import annotations

# Reach per GB — mirrors aether_context.config.TOKENS_PER_GB. Kept inline so the
# status bar stays dependency-light (the bar is "two ints + a string", off the
# hot path) and testable without importing the numpy-backed engine.
TOKENS_PER_GB = 233_000_000

# Phase -> (label, kaomoji). Tracks the agent's current activity.
PHASES: dict[str, tuple[str, str]] = {
    "anchoring": ("anchoring context", "＿φ(°-°=)"),  # encoding new context
    "scanning": ("scanning repo", "(ノ￣ー￣)ノ⌨"),  # repo search / reading
    "reasoning": ("reasoning", "(๑•̀ㅂ•́)و✧✎"),
    "grounding": ("grounding", "( Ò﹏Ó)✎"),  # running tests / type-check
    "paging": ("paging", "(⌨_⌨)"),  # slice load
}


def _human(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(int(n))


def render(used_tokens: int, pool_gb: int, phase: str = "reasoning", width: int = 30) -> str:
    """Render one status-bar line. `used_tokens` = tokens encoded in the pool;
    `pool_gb` = the user's selected pool size (sets the denominator)."""
    cap = pool_gb * TOKENS_PER_GB
    frac = 0.0 if cap <= 0 else min(1.0, used_tokens / cap)
    filled = round(frac * width)
    bar = "█" * filled + "░" * (width - filled)
    label, kao = PHASES.get(phase, PHASES["reasoning"])
    return (
        f"  {label} {kao} local/cache  "
        f"[ {_human(used_tokens)} / {_human(cap)} tokens ] "
        f"|{bar}| {frac * 100:.1f}%"
    )
