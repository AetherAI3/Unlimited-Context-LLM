# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Tests for aether_context.mpo — the MPO context chain (chain slices, expand a hit)."""
import numpy as np

from aether_context.mpo import ChainItem, MpoChain


def _vec(seed, dim=256):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _item(cid, vec, c_t=(0.0, 10.0)):
    return ChainItem(id=cid, vector=vec, c_t=c_t)


class TestEmbed:
    def test_deterministic(self):
        c = MpoChain()
        assert np.allclose(c._embed((0.3, 0.7)), c._embed((0.3, 0.7)))

    def test_unit_norm(self):
        c = MpoChain()
        assert abs(float(np.linalg.norm(c._embed((0.2, 0.9)))) - 1.0) < 1e-6

    def test_smooth(self):
        c = MpoChain()
        near = float(np.dot(c._embed((0.50, 0.50)), c._embed((0.51, 0.50))))
        far = float(np.dot(c._embed((0.50, 0.50)), c._embed((0.05, 0.95))))
        assert near > far


class TestCoupling:
    def test_semantic_drives_coupling(self):
        c = MpoChain()
        v, e = _vec(1), c._embed((0.5, 0.5))
        assert c.coupling(v, e, v, e) > c.coupling(_vec(2), e, v, e)

    def test_chain_proximity_modulates(self):
        c = MpoChain()
        v = _vec(3)
        e_near, e_far = c._embed((0.5, 0.5)), c._embed((0.0, 1.0))
        assert c.coupling(v, e_near, v, e_near) >= c.coupling(v, e_far, v, e_near)


class TestExpand:
    def test_hits_come_first(self):
        c = MpoChain(width=2)
        items = [_item(f"x{i}", _vec(i), c_t=(float(i), 10.0)) for i in range(6)]
        assert c.expand(["x0"], items, width=2)[0] == "x0"

    def test_pulls_coupled_slices(self):
        c = MpoChain(width=3)
        base = _vec(10)
        items = [
            _item("x0", base, c_t=(0.5, 10.0)),
            _item("x1", base * 0.99 + 0.01 * _vec(11), c_t=(0.5, 10.0)),
            _item("z", _vec(99), c_t=(0.0, 200.0)),
        ]
        out = c.expand(["x0"], items, width=2)
        assert out[0] == "x0"
        assert out.index("x1") < out.index("z")

    def test_empty_candidates_returns_hits(self):
        assert MpoChain().expand(["a"], []) == ["a"]

    def test_dedup_and_truncation_via_width(self):
        c = MpoChain(width=1)
        items = [_item(f"x{i}", _vec(i), c_t=(float(i), 10.0)) for i in range(5)]
        out = c.expand(["x0"], items, width=1, hops=1)
        assert out[0] == "x0" and len(out) == 2 and len(set(out)) == len(out)
