# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Receiver process of the latent-transfer bench.

Reads only the receiver-safe artifacts (questions, anchors, capsule index, capsule manifests and
bytes, and the projector so the adapter id matches the trained pair). It never opens
``dataset.json``: gold answers, blank/mismatched documents and the shuffle map are not visible
here. Every capsule is admitted through the backend with a freshly issued grant; a refusal in a
scored arm aborts the run, because a refused handoff is a bench failure, not a data point.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from aether_context.contracts import digest
from aether_context.contracts.latent_v1 import ContextHandoffGrantV1, LatentCapsuleV1, verify_latent_use
from aether_context.latent import handoff
from aether_context.latent.tensor_codec import load_tensor
from bench.latent_transfer import cost
from bench.latent_transfer.common import (
    ARMS,
    ARMS_CAPSULE,
    PROBE_DOCS,
    BackendFactory,
    append_jsonl,
    chunks,
    first_token,
    flop_pair,
    open_text,
    phase_record,
    read_json,
    write_json,
)

RECEIVER_FILES = frozenset({"questions.json", "anchors.json", "capsule_index.json", "projector.pt"})
RECEIVER_DIRS = frozenset({"capsules"})
PROBE_EXPECTED = {
    "tampered_byte": "TENSOR_DIGEST_MISMATCH",
    "wrong_capsule_grant": "CAPSULE_NOT_GRANTED",
    "expired_grant": "GRANT_EXPIRED",
    "wrong_principal": "PRINCIPAL_MISMATCH",
    "stale_fence": "STALE_FENCE",
    "wrong_recipient_model": "RECIPIENT_MISMATCH",
    "untrained_codebook": "SOURCE_MISMATCH",
}
EXPIRED_AGE_S = 1000


class EvaluationAbort(RuntimeError):
    """A scored handoff was refused, an anchor failed its digest, or an integrity probe misfired."""


class ReceiverView:
    """The only door from the evaluate phase into the run directory."""

    def __init__(self, run_dir: Path) -> None:
        self._root = run_dir

    def path(self, rel: str) -> Path:
        if rel not in RECEIVER_FILES and rel.split("/")[0] not in RECEIVER_DIRS:
            raise PermissionError(f"receiver may not open {rel!r}")
        return self._root / rel

    def json(self, rel: str) -> Any:
        return read_json(self.path(rel))

    def text(self, rel: str) -> str:
        return self.path(rel).read_text(encoding="utf-8")

    def bytes(self, rel: str) -> bytes:
        return self.path(rel).read_bytes()


class CapsuleStore:
    """Capsule manifests and bytes by digest; a manifest must hash to its own file name."""

    def __init__(self, view: ReceiverView) -> None:
        self._view = view
        self._cache: dict[str, tuple[LatentCapsuleV1, bytes]] = {}

    def get(self, name: str) -> tuple[LatentCapsuleV1, bytes]:
        if name not in self._cache:
            capsule = LatentCapsuleV1.model_validate_json(self._view.text(f"capsules/{name}.json"))
            if digest(capsule) != name:
                raise EvaluationAbort(f"capsule manifest {name} does not hash to its file name")
            self._cache[name] = (capsule, self._view.bytes(f"capsules/{name}.bin"))
        return self._cache[name]


class Run:
    """Per-call state: backend, config-derived constants and the receipts log."""

    def __init__(self, backend: Any, config: dict[str, Any], receipts_fh: Any, grants_fh: Any) -> None:
        self.backend = backend
        self.config = config
        self.receipts_fh = receipts_fh
        self.grants_fh = grants_fh
        self.adapter_id = backend.adapter_id
        self.policy_digest = digest({"policy": config["policy"]})
        self.injected = 0
        self.verified = 0

    def grant(self, capsule: LatentCapsuleV1, grant_id: str, issued_at: int) -> ContextHandoffGrantV1:
        c = self.config
        return handoff.issue_grant(
            capsule, grant_id=grant_id, lane_id=c["lane_id"], sender_principal_id=c["sender_principal"],
            recipient_principal_id=c["receiver_principal"], fence=c["fence"],
            policy_digest=self.policy_digest, issued_at=issued_at, ttl_seconds=c["grant_ttl"])

    def admit(self, capsule: LatentCapsuleV1, grant: ContextHandoffGrantV1, raw: bytes, question: str,
              receipt_id: str, now: int, **kw: Any) -> Any:
        kw.setdefault("principal_id", self.config["receiver_principal"])
        kw.setdefault("current_fence", self.config["fence"])
        kw.setdefault("adapter_id", self.adapter_id)  # cached: the projector is fixed in this phase
        adm = self.backend.admit(capsule, grant, raw, question, receipt_id=receipt_id, now=now, **kw)
        append_jsonl(self.receipts_fh, adm.receipt.model_dump(mode="json"))
        append_jsonl(self.grants_fh, {"receipt_id": receipt_id, "grant": grant.model_dump(mode="json")})
        return adm


def _admit_scored(run: Run, store: CapsuleStore, name: str, q: dict[str, Any], arm: str) -> Any:
    capsule, raw = store.get(name)
    now = int(time.time())
    grant = run.grant(capsule, f"g:{q['item_id']}:{arm}", now)
    adm = run.admit(capsule, grant, raw, q["question"], f"r:{q['item_id']}:{arm}", now)
    if adm.prefix is None:
        raise EvaluationAbort(f"{arm} handoff for {q['item_id']} refused: {adm.receipt.reason_code}")
    verify_latent_use(capsule, grant, adm.receipt)
    run.injected += 1
    run.verified += 1
    return adm


def _probe_outcomes(run: Run, store: CapsuleStore, index: dict[str, dict[str, str]], doc_id: str,
                    q: dict[str, Any]) -> dict[str, str]:
    capsule, raw = store.get(index[doc_id]["paired"])
    blank, _ = store.get(index[doc_id]["blank"])
    now = int(time.time())
    question = q["question"]
    tag = f"p:{doc_id}"
    good = run.grant(capsule, f"{tag}:g", now)

    def code(name: str, cap: LatentCapsuleV1, grant: ContextHandoffGrantV1, data: bytes, **kw: Any) -> str:
        adm = run.admit(cap, grant, data, question, f"{tag}:{name}", now, **kw)
        return adm.receipt.reason_code or adm.receipt.outcome

    forged, forged_raw = run.backend.make_capsule(
        load_tensor(capsule.tensor, raw).astype(np.float32), namespace=capsule.namespace,
        capsule_id=f"{doc_id}:forged", anchor_digest=capsule.anchor_digest,
        base_revision=capsule.base_revision, created_at=capsule.created_at,
        codebook_digest=digest("untrained-codebook"))
    # Right recipient adapter on the grant, wrong source adapter on the capsule: only the source check fails.
    forged_grant = run.grant(forged, f"{tag}:fg", now).model_copy(update={"recipient_adapter_id": run.adapter_id})
    return {
        "tampered_byte": code("tamper", capsule, good, bytes([raw[0] ^ 1]) + raw[1:]),
        "wrong_capsule_grant": code("wrongcap", capsule, run.grant(blank, f"{tag}:bg", now), raw),
        "expired_grant": code("expired", capsule, run.grant(capsule, f"{tag}:eg", now - EXPIRED_AGE_S), raw),
        "wrong_principal": code("principal", capsule, good, raw, principal_id="intruder"),
        "stale_fence": code("fence", capsule, good, raw, current_fence=run.config["fence"] + 1),
        "wrong_recipient_model": code(
            "recipient", capsule, good.model_copy(update={"recipient_model_id": "other/model@0"}), raw),
        "untrained_codebook": code("source", forged, forged_grant, forged_raw),
    }


def run_probes(run: Run, store: CapsuleStore, index: dict[str, dict[str, str]],
               questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    first_q: dict[str, dict[str, Any]] = {}
    for q in questions:
        first_q.setdefault(q["doc_id"], q)
    results = []
    for doc_id in list(first_q)[:PROBE_DOCS]:
        for name, observed in _probe_outcomes(run, store, index, doc_id, first_q[doc_id]).items():
            expected = PROBE_EXPECTED[name]
            results.append({"doc_id": doc_id, "probe": name, "expected": expected, "observed": observed,
                            "matched": observed == expected})
    missed = [r for r in results if not r["matched"]]
    if missed:
        raise EvaluationAbort(f"integrity probes not refused as expected: {missed[:5]}")
    return results


def verify_anchors(store: CapsuleStore, index: dict[str, dict[str, str]], anchors: dict[str, Any]) -> int:
    for doc_id, anchor in anchors.items():
        capsule, _ = store.get(index[doc_id]["paired"])
        if digest(anchor) != capsule.anchor_digest:
            raise EvaluationAbort(f"anchor for {doc_id} does not match the paired capsule's anchor digest")
    return len(anchors)


def _score_arm(run: Run, arm: str, questions: list[dict[str, Any]], build: Callable[[dict[str, Any]], tuple],
               token_ids: dict[str, int], rows_fh: Any) -> dict[str, Any]:
    """``build(q) -> (prefix, context, n_tokens, receipt_id)``; scores in batches, writes item rows."""
    t0 = time.perf_counter()
    n_tokens = 0
    for batch in chunks(questions, run.config["score_batch"]):
        built = [build(q) for q in batch]
        lp = run.backend.next_token_logprobs([b[0] for b in built], [q["question"] for q in batch],
                                             [b[1] for b in built])
        for q, b, row in zip(batch, built, lp):
            scores = [float(row[token_ids[c]]) for c in q["candidates"]]
            best = max(scores)
            append_jsonl(rows_fh, {
                "item_id": q["item_id"], "doc_id": q["doc_id"], "family": q["family"],
                "attribute": q["attribute"], "arm": arm, "pred": q["candidates"][scores.index(best)],
                "scores": scores, "tie": scores.count(best) > 1, "receipt_id": b[3]})
            n_tokens += b[2]
    blocks, head = flop_pair(run.backend)
    return {"wall_s": time.perf_counter() - t0, "items": len(questions), "receiver_tokens": n_tokens,
            "flops": cost.forward_flops(blocks, head, n_tokens, len(questions))}


def evaluate(run_dir: Path, config: dict[str, Any], backend_factory: BackendFactory | None = None, *,
             check_tokens: bool = True) -> dict[str, Any]:
    from bench.latent_transfer.run import default_backend_factory

    t0 = time.perf_counter()
    view = ReceiverView(run_dir)
    backend = (backend_factory or default_backend_factory)(config)
    backend.load_projector(view.path("projector.pt"))
    questions, anchors, index = view.json("questions.json"), view.json("anchors.json"), view.json("capsule_index.json")
    store = CapsuleStore(view)
    tok = backend.tokenizer
    token_ids = {c: first_token(tok, c) for q in questions for c in q["candidates"]}
    for q in questions if check_tokens else ():  # prepare checks too; the receiver cannot read prepare.json
        if len({token_ids[c] for c in q["candidates"]}) != len(q["candidates"]):
            raise EvaluationAbort(f"candidate first tokens collide for {q['item_id']}")
    layouts: dict[str, int] = {}

    def layout_len(q: dict[str, Any]) -> int:
        if q["item_id"] not in layouts:
            lay = backend.layout(q["question"])
            layouts[q["item_id"]] = len(lay.before) + len(lay.after)
        return layouts[q["item_id"]]

    arm_stats: dict[str, Any] = {}
    anchors_verified = 0
    with open_text(run_dir / "items.jsonl") as rows_fh, open_text(run_dir / "receipts.jsonl") as receipts_fh, \
            open_text(run_dir / "grants.jsonl") as grants_fh:
        run = Run(backend, config, receipts_fh, grants_fh)
        probes = run_probes(run, store, index, questions)
        anchors_verified = verify_anchors(store, index, anchors)  # before any arm is scored
        for arm in ARMS:
            if arm in ARMS_CAPSULE:
                def build(q: dict[str, Any], arm: str = arm) -> tuple:
                    adm = _admit_scored(run, store, index[q["doc_id"]][arm], q, arm)
                    return adm.prefix, None, layout_len(q) + int(adm.prefix.shape[0]), adm.receipt.receipt_id
            elif arm == "no_prefix":
                def build(q: dict[str, Any]) -> tuple:
                    return None, None, layout_len(q), None
            else:
                def build(q: dict[str, Any]) -> tuple:
                    ctx = anchors[q["doc_id"]]["text"]
                    n_ctx = len(tok(ctx + "\n\n", add_special_tokens=False).input_ids)
                    return None, ctx, layout_len(q) + n_ctx, None
            before = (run.injected, run.verified)
            arm_stats[arm] = _score_arm(run, arm, questions, build, token_ids, rows_fh)
            arm_stats[arm].update(injected=run.injected - before[0], verified=run.verified - before[1])
    blocks, head = flop_pair(backend)
    record = phase_record(
        "evaluate", config, time.perf_counter() - t0, arms=arm_stats, n_items=len(questions),
        receipts_injected=sum(a["injected"] for a in arm_stats.values()),
        receipts_verified=sum(a["verified"] for a in arm_stats.values()), anchors_verified=anchors_verified,
        n_eval_docs=len(anchors), probes=probes, adapter_id=run.adapter_id,
        codebook_digest=backend.codebook_digest(), base_digest=backend.base_digest(),
        flop_params={"blocks": blocks, "head": head})
    write_json(run_dir / "evaluate.json", record)
    return record
