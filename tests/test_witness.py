# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Tests for the +/- retention witness (page-replacement scoring + budget eviction).

The witness is a *pure scoring function over access events*: each slice id carries a
retention score that **hardens** when the slice is touched with salience, **fades** as
idle time elapses, and **re-hardens** when the slice is relevant again. Eviction drops
the lowest-score slices first, stopping the instant the pool is back under its ceiling.

All tests are numpy-only and never touch the network. Math (geometric mean of
surprise x impact x uniqueness, tanh squash, uniqueness = 1/(1+neighbors)) is a pure
retention policy over access events.
"""
from aether_context.witness import (
    Witness,
    retention_score,
    squash,
    uniqueness_from_neighbors,
    DEFAULT_DECAY_RATE,
    SALIENT_THRESHOLD,
)


# ---- pure scoring math ------------------------------------------------------
def test_retention_score_is_geometric_mean_of_drivers():
    # geometric mean: a single weak driver can't be masked by strong others
    assert retention_score(1.0, 1.0, 1.0) == 1.0
    assert retention_score(0.0, 1.0, 1.0) == 0.0  # one zero driver -> zero
    mid = retention_score(0.5, 0.5, 0.5)
    assert abs(mid - 0.5) < 1e-9


def test_retention_score_clamps_inputs_to_unit_interval():
    # out-of-range inputs are clamped to [0,1] before the geometric mean
    assert retention_score(5.0, 5.0, 5.0) == 1.0
    assert retention_score(-1.0, 0.5, 0.5) == 0.0


def test_squash_is_monotone_and_bounded():
    assert squash(0.0, 1.0) == 0.0
    assert 0.0 < squash(1.0, 1.0) < 1.0
    assert squash(10.0, 1.0) > squash(1.0, 1.0)  # bigger magnitude -> higher
    assert squash(5.0, 0.0) == 0.0  # non-positive scale is a safe no-op


def test_uniqueness_decreases_with_neighbor_count():
    assert uniqueness_from_neighbors(0) == 1.0
    assert uniqueness_from_neighbors(1) == 0.5
    assert uniqueness_from_neighbors(9) == 0.1
    assert uniqueness_from_neighbors(-5) == 1.0  # negative neighbor count clamps to 0


# ---- harden (touch with salience) -------------------------------------------
def test_touch_registers_a_slice_with_its_salience_as_score():
    w = Witness()
    w.touch("a", salience=0.8, now=0.0)
    assert "a" in w.ids()
    assert abs(w.score("a") - 0.8) < 1e-9


def test_touch_clamps_salience_into_unit_interval():
    w = Witness()
    w.touch("hi", salience=2.0, now=0.0)
    assert w.score("hi") == 1.0
    w.touch("lo", salience=-1.0, now=0.0)
    assert w.score("lo") == 0.0


def test_unknown_slice_has_zero_score():
    w = Witness()
    assert w.score("nope") == 0.0


# ---- re-harden on re-touch lifts a faded slice ------------------------------
def test_reharden_lifts_a_faded_slice():
    w = Witness()
    w.touch("x", salience=0.9, now=0.0)
    w.decay(now=20.0)              # let it fade over 20 idle units
    faded = w.score("x")
    assert faded < 0.9            # it really faded
    w.touch("x", salience=0.9, now=20.0)  # relevant again -> re-harden
    assert w.score("x") > faded   # re-touch lifted it back up
    assert w.score("x") >= 0.9 - 1e-9     # restored to (at least) the fresh salience


def test_reharden_takes_the_stronger_of_decayed_and_new_salience():
    # re-touching with a weak salience must not *lower* a still-strong slice
    w = Witness()
    w.touch("s", salience=0.9, now=0.0)
    before = w.score("s")
    w.touch("s", salience=0.1, now=0.0)  # same instant, weak salience
    assert w.score("s") >= before        # never demoted by a weaker re-touch


# ---- fade / decay is monotone in elapsed time -------------------------------
def test_decay_is_monotone_in_elapsed_time():
    scores = []
    for t in (0.0, 5.0, 10.0, 20.0, 40.0):
        w = Witness()
        w.touch("m", salience=1.0, now=0.0)
        w.decay(now=t)
        scores.append(w.score("m"))
    # strictly non-increasing as elapsed time grows
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    assert scores[0] > scores[-1]   # it does actually decay


def test_decay_never_goes_negative():
    w = Witness()
    w.touch("d", salience=1.0, now=0.0)
    w.decay(now=10_000.0)  # very long idle
    assert w.score("d") >= 0.0


def test_decay_default_rate_is_positive():
    assert DEFAULT_DECAY_RATE > 0.0


# ---- rank() orders ids by score, highest first ------------------------------
def test_rank_orders_ids_by_score_descending():
    w = Witness()
    w.touch("low", salience=0.2, now=0.0)
    w.touch("high", salience=0.9, now=0.0)
    w.touch("mid", salience=0.5, now=0.0)
    assert w.rank() == ["high", "mid", "low"]


def test_rank_reflects_decay_after_differential_aging():
    w = Witness()
    w.touch("fresh", salience=0.6, now=0.0)
    w.touch("stale", salience=0.6, now=0.0)
    w.touch("fresh", salience=0.6, now=30.0)  # fresh re-touched recently
    w.decay(now=30.0)                          # stale has aged 30 units, fresh 0
    ranked = w.rank()
    assert ranked.index("fresh") < ranked.index("stale")


# ---- budget_evict: lowest score first, stops at ceiling ---------------------
def test_budget_evict_drops_lowest_score_first():
    w = Witness()
    w.touch("keep", salience=0.9, now=0.0)
    w.touch("drop", salience=0.1, now=0.0)
    # ceiling allows only one slice (bytes_per_slice=100, ceiling=100)
    evicted = w.budget_evict(ceiling_bytes=100, bytes_per_slice=100)
    assert evicted == ["drop"]          # the lowest score went first
    assert "drop" not in w.ids()
    assert "keep" in w.ids()


def test_budget_evict_stops_exactly_at_ceiling():
    w = Witness()
    for i in range(10):
        w.touch(f"s{i}", salience=(i + 1) / 10.0, now=0.0)
    # 10 slices * 100 bytes = 1000; ceiling 500 -> keep 5, evict 5
    evicted = w.budget_evict(ceiling_bytes=500, bytes_per_slice=100)
    assert len(evicted) == 5            # stopped the moment it fit
    assert len(w.ids()) == 5
    # the 5 survivors are the highest scores
    survivors = set(w.ids())
    assert survivors == {"s5", "s6", "s7", "s8", "s9"}


def test_budget_evict_noop_when_already_under_ceiling():
    w = Witness()
    w.touch("a", salience=0.5, now=0.0)
    evicted = w.budget_evict(ceiling_bytes=10_000, bytes_per_slice=100)
    assert evicted == []
    assert "a" in w.ids()


# ---- the headline property: hardened survives an eviction that drops stale ---
def test_hardened_survives_eviction_that_drops_stale():
    """A salient, freshly-touched slice must outlive a stale low-salience one even
    though the stale one was added first — eviction is by *score*, never recency."""
    w = Witness()
    w.touch("tail_event", salience=0.95, now=0.0)   # rare, high-salience -> HOT
    w.touch("chatter", salience=0.15, now=0.0)      # low-salience filler
    w.decay(now=40.0)                                # both age, but chatter started low
    w.touch("tail_event", salience=0.95, now=40.0)  # tail event is relevant again
    # budget for a single slice -> the witness must keep the hardened one
    evicted = w.budget_evict(ceiling_bytes=100, bytes_per_slice=100)
    assert "chatter" in evicted
    assert "tail_event" not in evicted
    assert w.ids() == ["tail_event"]


def test_salient_threshold_is_in_unit_interval():
    assert 0.0 < SALIENT_THRESHOLD < 1.0


# ---- forget / membership convenience ----------------------------------------
def test_forget_removes_a_slice():
    w = Witness()
    w.touch("a", salience=0.5, now=0.0)
    w.forget("a")
    assert "a" not in w.ids()
    assert w.score("a") == 0.0


def test_forget_unknown_id_is_a_safe_noop():
    w = Witness()
    w.forget("ghost")  # must not raise
    assert w.ids() == []


# ---- temporal lock-in (anti-thrash) -----------------------------------------
def test_pin_protects_a_fresh_slice_from_comparable_churn():
    """A just-touched (paged-in) slice survives a wave of equal-salience churn: the
    lock-in bonus lifts it above older slices of the same base score."""
    w = Witness(pin_periods=3.0, pin_bonus=0.25)
    w.touch("old1", salience=0.30, now=0.0)
    w.touch("old2", salience=0.30, now=1.0)
    w.touch("fresh", salience=0.30, now=10.0)  # just paged in -> locked in
    # room for two; evaluate eviction at the fresh slice's tick
    evicted = w.budget_evict(ceiling_bytes=200, bytes_per_slice=100, now=10.0)
    assert evicted == ["old1"]              # an unpinned, comparable-score slice went first
    assert "fresh" in w.ids()               # the freshly paged-in slice was protected


def test_pin_never_overrides_a_load_bearing_salience():
    """The lock-in is a *small* bonus: it cannot save a fresh low slice at the expense of a
    genuinely high-salience one, so salience still wins where it matters."""
    w = Witness(pin_periods=3.0, pin_bonus=0.25)
    w.touch("load_bearing", salience=0.95, now=0.0)  # old, high salience, not re-touched
    w.touch("fresh_lo", salience=0.30, now=10.0)     # just paged in, but low value
    evicted = w.budget_evict(ceiling_bytes=100, bytes_per_slice=100, now=10.0)
    assert evicted == ["fresh_lo"]          # 0.30+0.25 still loses to 0.95
    assert w.ids() == ["load_bearing"]


def test_pin_disabled_falls_back_to_pure_score_eviction():
    """With the lock-in disabled the governor is pure salience order, ignoring recency."""
    w = Witness(pin_periods=0.0, pin_bonus=0.0)
    w.touch("old_hi", salience=0.90, now=0.0)
    w.touch("fresh_lo", salience=0.10, now=10.0)
    evicted = w.budget_evict(ceiling_bytes=100, bytes_per_slice=100, now=10.0)
    assert evicted == ["fresh_lo"]          # recency is irrelevant; lowest score goes


def test_eviction_order_without_now_is_ascending_score():
    """eviction_order(now=None) is simply most-evictable (lowest score) first."""
    w = Witness()
    w.touch("hi", salience=0.9, now=0.0)
    w.touch("lo", salience=0.1, now=0.0)
    w.touch("mid", salience=0.5, now=0.0)
    assert w.eviction_order() == ["lo", "mid", "hi"]


# -- permanence ---------------------------------------------------------------


def test_a_pinned_slice_never_fades() -> None:
    """Salience cannot express 'keep this because of what it is'.

    A slice nothing queries decays exactly like one nothing needs, so an
    agent's doctrine fades out of reach precisely because it is background.
    """
    w = Witness(decay_rate=0.5)
    w.touch("doctrine", salience=0.9, now=0.0)
    w.touch("chatter", salience=0.9, now=0.0)
    w.pin("doctrine")

    for tick in range(1, 60):
        w.decay(float(tick))

    assert w.score("doctrine") == 0.9, "a pinned base must be preserved exactly"
    assert w.score("chatter") < 0.01
    assert w.is_permanent("doctrine")
    assert not w.is_permanent("chatter")


def test_a_pinned_slice_is_never_evictable() -> None:
    w = Witness()
    w.touch("doctrine", salience=0.01, now=0.0)   # deliberately the weakest
    for i in range(10):
        w.touch(f"loud-{i}", salience=1.0, now=0.0)
    w.pin("doctrine")

    order = w.eviction_order()
    assert order[-1] == "doctrine", "the weakest slice must still rank last once pinned"
    assert order.index("doctrine") == len(order) - 1


def test_budget_eviction_cannot_reach_a_pinned_slice_at_any_pressure() -> None:
    """The guarantee has to hold at the moment it matters most.

    Ranking pinned slices last is not enough: once the ceiling falls below the
    pinned count the evictable prefix reaches them anyway, and the pool drops
    the agent's doctrine to make room for whatever it was reading.
    """
    w = Witness()
    for name in ("doctrine", "skills", "grounding"):
        w.touch(name, salience=0.05, now=0.0)
        w.pin(name)
    for i in range(20):
        w.touch(f"spill-{i}", salience=0.99, now=0.0)

    # A ceiling of one slice, with three pinned. Nothing pinned may go.
    evicted = w.budget_evict(ceiling_bytes=1, bytes_per_slice=1, now=1.0)

    assert "doctrine" not in evicted
    assert "skills" not in evicted
    assert "grounding" not in evicted
    for name in ("doctrine", "skills", "grounding"):
        assert w.score(name) > 0.0, f"{name} was dropped from the witness"


def test_unpin_returns_a_slice_to_ordinary_decay() -> None:
    w = Witness(decay_rate=0.5)
    w.touch("temp", salience=0.9, now=0.0)
    w.pin("temp")
    for tick in range(1, 20):
        w.decay(float(tick))
    assert w.score("temp") == 0.9

    w.unpin("temp")
    assert not w.is_permanent("temp")
    for tick in range(20, 80):
        w.decay(float(tick))
    assert w.score("temp") < 0.01


def test_pinning_an_unseen_id_is_remembered() -> None:
    """Callers should not have to order pin and touch."""
    w = Witness()
    w.pin("later")
    assert w.is_permanent("later")
    w.touch("later", salience=0.5, now=0.0)
    for tick in range(1, 40):
        w.decay(float(tick))
    assert w.score("later") == 0.5
