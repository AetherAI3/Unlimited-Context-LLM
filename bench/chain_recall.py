# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""chain_recall — does the MPO context chain pull more *connected* context than cosine alone?

Plants threads (groups of related slices) into a pool alongside distractors, then for each
thread queries with one member and measures **thread recall@k**: of the other members of that
thread, how many land in the k-slice working set. Compares cosine-only retrieval to
cosine + MPO chain expansion. The chain should pull in more of the connected thread.

Usage: python -m bench.chain_recall
"""
from __future__ import annotations

import numpy as np

from aether_context.encoder import StaticEncoder
from aether_context.mpo import ChainItem, MpoChain

N_THREADS = 40
THREAD_SIZE = 5
DISTRACTORS = 400
K = 8
FANOUT = 6
_TOPICS = [
    "deploy pipeline build sign ship installer release", "auth token refresh session cookie login",
    "database schema migration index query plan", "cache layer eviction warm cold hit rate",
    "vector encode slice pool recover retrieval", "stream chunk window overflow spill encode",
]
_NOISE = "groceries weather sports movie recipe travel garden music coffee bicycle".split()


def _cosine_topk(q, items, k):
    sims = [(float(np.dot(q, it.vector)), it.id) for it in items]
    sims.sort(key=lambda p: p[0], reverse=True)
    return [sid for _, sid in sims[:k]]


def main() -> int:
    rng = np.random.default_rng(0)
    enc = StaticEncoder(dim=256)
    chain = MpoChain(width=K, hops=1)
    items: list[ChainItem] = []
    threads: list[list[str]] = []
    t = 0.0

    for ti in range(N_THREADS):
        topic = _TOPICS[ti % len(_TOPICS)]
        ids = []
        for j in range(THREAD_SIZE):
            text = topic + " " + " ".join(rng.choice(topic.split(), size=3))
            cid = f"t{ti}_{j}"
            items.append(ChainItem(cid, enc.encode(text), c_t=(t, float(len(text.split())))))
            ids.append(cid)
            t += 1.0
        threads.append(ids)
    for d in range(DISTRACTORS):
        text = " ".join(rng.choice(_NOISE, size=6))
        items.append(ChainItem(f"d{d}", enc.encode(text), c_t=(t, float(len(text.split())))))
        t += 1.0

    by_id = {it.id: it for it in items}
    cos_hits = chain_hits = total = 0
    for ids in threads:
        query = by_id[ids[0]].vector  # query with the first member
        others = set(ids[1:])
        total += len(others)
        # cosine-only top-k
        cos = set(_cosine_topk(query, items, K))
        cos_hits += len(cos & others)
        # cosine recall (wider) -> chain expand to k
        wide = _cosine_topk(query, items, K * FANOUT)
        cand = [by_id[i] for i in wide]
        order = chain.expand([ids[0]], cand, width=K)
        ch = set(order[:K])
        chain_hits += len(ch & others)

    print(f"threads={N_THREADS} size={THREAD_SIZE} distractors={DISTRACTORS} k={K}")
    print(f"  cosine-only thread recall@{K}: {cos_hits / total:.3f}")
    print(f"  + MPO chain   thread recall@{K}: {chain_hits / total:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
