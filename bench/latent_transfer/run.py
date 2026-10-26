# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""CLI for the latent-transfer experiment (G-W7).

``python -m bench.latent_transfer.run <phase> --run-id ID [options]`` with phase one of
``prepare | train | encode | evaluate | report | all``. Under ``all`` every phase is its own
process, so the sender (``encode``) and the receiver (``evaluate``) never share memory.

Each phase is a function ``(run_dir, config, backend_factory)`` writing ``<phase>.json`` next to
its artifacts under ``bench/latent_transfer/runs/<run_id>/`` (git-ignored). A ``backend_factory``
maps the config to an ``HFSoftPrefixBackend``; tests inject a tiny model, the CLI loads the pinned
Qwen checkpoint.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from aether_context.contracts import NamespaceV1, digest
from bench.latent_transfer import cost, tasks
from bench.latent_transfer.common import (
    REPORT_PATH,
    REPO_ROOT,
    RESULTS_DIR,
    RUNS_ROOT,
    BackendFactory,
    append_jsonl,
    check_pinned,
    chunks,
    first_token,
    flop_pair,
    load_pinned_model,
    open_text,
    phase_record,
    read_json,
    write_json,
)

CONFIG: dict[str, Any] = {
    "model_ref": "Qwen/Qwen2.5-0.5B-Instruct",
    "model_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
    "dataset_seed": 20261025,
    "n_train": 2000,
    "n_eval": 200,
    "slots": 8,
    "d_attn": 512,
    "sender_layer": None,
    "system": "Answer with a single word.",
    "storage_dtype": "float16",
    "steps": 3000,
    "batch_size": 4,
    "lr": 1e-3,
    "train_seed": 0,
    "score_batch": 16,
    "grant_ttl": 300,
    "lane_id": "l7-latent-transfer",
    "fence": 1,
    "sender_principal": "l7-sender",
    "receiver_principal": "l7-receiver",
    "policy": "g-w7-bench/v1",
}
PHASES = ("prepare", "train", "encode", "evaluate", "report")
LOG_EVERY = 25
ENCODE_BATCH = 16
OVERRIDE_KEYS = ("steps", "lr", "n_train", "n_eval", "batch_size")


def make_config(**overrides: Any) -> dict[str, Any]:
    """``CONFIG`` with explicit overrides; unknown keys are rejected so typos cannot go unrecorded."""
    unknown = set(overrides) - set(CONFIG)
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")
    return {**CONFIG, **overrides}


def default_backend_factory(config: dict[str, Any]) -> Any:
    from aether_context.latent.hf_soft_prefix import HFSoftPrefixBackend

    model, tok = load_pinned_model(config["model_ref"], config["model_revision"])
    return HFSoftPrefixBackend(
        model, tok, model_id=f"{config['model_ref']}@{config['model_revision']}", slots=config["slots"],
        d_attn=config["d_attn"], sender_layer=config["sender_layer"], system=config["system"],
        storage_dtype=config["storage_dtype"])


def candidate_token_ids(tokenizer: Any) -> dict[str, dict[str, int]]:
    """First-token ids of every candidate; raises if two candidates of one question collide."""
    groups = {**{a: list(v) for a, v in tasks.ATTRIBUTES.items()}, "yes_no": list(tasks.YES_NO_CANDIDATES)}
    ids: dict[str, dict[str, int]] = {}
    for name, candidates in groups.items():
        ids[name] = {c: first_token(tokenizer, c) for c in candidates}
        if len(set(ids[name].values())) != len(candidates):
            raise ValueError(f"candidate first tokens collide for {name}: {ids[name]}")
    return ids


def prepare(run_dir: Path, config: dict[str, Any], backend_factory: BackendFactory | None = None, *,
            check_tokens: bool = True) -> dict[str, Any]:
    t0 = time.perf_counter()
    ds = tasks.build(config["dataset_seed"], config["n_train"], config["n_eval"])
    first_tokens = None
    if check_tokens:
        backend = (backend_factory or default_backend_factory)(config)
        first_tokens = candidate_token_ids(backend.tokenizer)  # raises before anything is written
    write_json(run_dir / "dataset.json", tasks.to_json(ds))
    write_json(run_dir / "questions.json", [
        {"item_id": i.item_id, "doc_id": i.doc_id, "family": i.family, "attribute": i.attribute,
         "question": i.question, "candidates": list(i.candidates)} for i in ds.eval_items])
    write_json(run_dir / "anchors.json", {d.doc_id: tasks.anchor(d) for d in ds.eval_docs})
    record = phase_record(
        "prepare", config, time.perf_counter() - t0, dataset_digest=tasks.dataset_digest(ds),
        check_tokens=check_tokens, first_token_ids=first_tokens, n_train_docs=len(ds.train_docs),
        n_eval_docs=len(ds.eval_docs), n_train_examples=len(ds.train_examples),
        n_eval_items=len(ds.eval_items))
    write_json(run_dir / "prepare.json", record)
    return record


def _load_prepared(run_dir: Path, config: dict[str, Any]) -> tuple[tasks.Dataset, dict[str, Any]]:
    prepared = read_json(run_dir / "prepare.json")
    check_pinned(config, prepared)
    ds = tasks.from_json(read_json(run_dir / "dataset.json"))
    if tasks.dataset_digest(ds) != prepared["dataset_digest"]:
        raise RuntimeError("dataset.json does not match the digest recorded by prepare")
    return ds, prepared


def train(run_dir: Path, config: dict[str, Any], backend_factory: BackendFactory | None = None,
          ) -> dict[str, Any]:
    from aether_context.latent.training import TrainExample, train_projector

    t0 = time.perf_counter()
    ds, _ = _load_prepared(run_dir, config)
    backend = (backend_factory or default_backend_factory)(config)
    text = {d.doc_id: d.text for d in ds.train_docs}
    examples = [TrainExample(text[e.doc_id], e.question, e.answer) for e in ds.train_examples]
    with open_text(run_dir / "train_log.jsonl") as fh:
        def log(step: int, loss: float) -> None:
            append_jsonl(fh, {"step": step, "mean_loss": loss, "elapsed_s": time.perf_counter() - t0})
            fh.flush()
            print(f"train step {step}/{config['steps']} mean_loss={loss:.4f}", flush=True)

        stats = train_projector(backend, examples, steps=config["steps"], batch_size=config["batch_size"],
                                lr=config["lr"], seed=config["train_seed"], log=log, log_every=LOG_EVERY)
    backend.save_projector(run_dir / "projector.pt")
    blocks, head = flop_pair(backend)
    record = phase_record(
        "train", config, time.perf_counter() - t0, stats=asdict(stats),
        codebook_digest=backend.codebook_digest(), adapter_id=backend.adapter_id,
        base_digest_before=stats.base_digest_before, base_digest_after=stats.base_digest_after,
        n_examples=len(examples), flop_params={"blocks": blocks, "head": head},
        sender_tokens=stats.sender_tokens, receiver_tokens=stats.receiver_tokens,
        head_tokens=stats.head_tokens,
        flops={"sender": cost.forward_flops(blocks, head, stats.sender_tokens, 0),
               "train": cost.train_flops(blocks, head, stats.receiver_tokens, stats.head_tokens)})
    write_json(run_dir / "train.json", record)
    return record


def encode(run_dir: Path, config: dict[str, Any], backend_factory: BackendFactory | None = None,
           ) -> dict[str, Any]:
    """Sender process: one capsule per (eval doc, arm); reads document texts only."""
    t0 = time.perf_counter()
    ds, prepared = _load_prepared(run_dir, config)
    backend = (backend_factory or default_backend_factory)(config)
    backend.load_projector(run_dir / "projector.pt")
    book = backend.codebook_digest()
    ns = NamespaceV1(owner_id="bench", project_id="unlimited-context", objective_id="g-w7",
                     context_cycle_id=run_dir.name)
    jobs = [(doc, arm, source) for doc in ds.eval_docs
            for arm, source in (("paired", doc), ("blank", ds.blanks[doc.doc_id]),
                                ("mismatched", ds.mismatched[doc.doc_id]))]
    index: dict[str, dict[str, str]] = {}
    total_bytes = n_tokens = 0
    (run_dir / "capsules").mkdir(parents=True, exist_ok=True)
    for batch in chunks(jobs, ENCODE_BATCH):
        texts = [source.text for _, _, source in batch]
        n_tokens += sum(len(backend.tokenizer(t, add_special_tokens=False).input_ids) for t in texts)
        for (doc, arm, source), prefix in zip(batch, backend.encode_prefixes(texts)):
            capsule, raw = backend.make_capsule(
                prefix, namespace=ns, capsule_id=f"{doc.doc_id}:{arm}",
                anchor_digest=tasks.anchor_digest(source), base_revision=prepared["dataset_digest"],
                created_at=int(time.time()), codebook_digest=book)
            name = digest(capsule)
            with open_text(run_dir / "capsules" / f"{name}.json") as fh:
                fh.write(capsule.model_dump_json())
            (run_dir / "capsules" / f"{name}.bin").write_bytes(raw)
            index.setdefault(doc.doc_id, {})[arm] = name
            total_bytes += len(raw)
    for doc_id, source_id in ds.shuffle_source.items():
        index[doc_id]["shuffled"] = index[source_id]["paired"]
    write_json(run_dir / "capsule_index.json", index)
    blocks, head = flop_pair(backend)
    record = phase_record(
        "encode", config, time.perf_counter() - t0, n_capsules=len(jobs), n_docs=len(index),
        tensor_bytes=total_bytes, bytes_per_capsule=total_bytes / len(jobs), sender_tokens=n_tokens,
        flop_params={"blocks": blocks, "head": head},
        flops={"sender": cost.forward_flops(blocks, head, n_tokens, 0)}, codebook_digest=book,
        adapter_id=backend.adapter_id, base_digest=backend.base_digest(),
        dataset_digest=prepared["dataset_digest"])
    write_json(run_dir / "encode.json", record)
    return record


def run_phase(phase: str, run_dir: Path, config: dict[str, Any],
              backend_factory: BackendFactory | None = None, **kw: Any) -> dict[str, Any]:
    from bench.latent_transfer.evaluate import evaluate
    from bench.latent_transfer.report import report

    fn = {"prepare": prepare, "train": train, "encode": encode, "evaluate": evaluate, "report": report}[phase]
    return fn(run_dir, config, backend_factory, **kw)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m bench.latent_transfer.run", description=__doc__)
    p.add_argument("phase", choices=(*PHASES, "all"))
    p.add_argument("--run-id", required=True)
    p.add_argument("--steps", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--n-train", type=int)
    p.add_argument("--n-eval", type=int)
    p.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--report-path", type=Path, default=REPORT_PATH)
    return p


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {k: getattr(args, k) for k in OVERRIDE_KEYS if getattr(args, k) is not None}


def _run_all(args: argparse.Namespace) -> int:
    """Run every phase as its own process; refuse up front what would only fail hours later."""
    run_dir = args.runs_root / args.run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        print(f"{run_dir} is not empty; a restart needs a new --run-id (record it in REPORT.md)",
              file=sys.stderr)
        return 2
    if args.report_path.exists():
        print(f"{args.report_path} exists; move it or pass --report-path", file=sys.stderr)
        return 2
    flags = [f"--{k.replace('_', '-')}={v}" for k, v in _overrides(args).items()]
    flags += [f"--runs-root={args.runs_root}", f"--results-dir={args.results_dir}",
              f"--report-path={args.report_path}"]
    for phase in PHASES:
        cmd = [sys.executable, "-m", "bench.latent_transfer.run", phase, "--run-id", args.run_id, *flags]
        print(f"==> {phase}", flush=True)
        code = subprocess.run(cmd, cwd=REPO_ROOT).returncode
        if code != 0:
            print(f"phase {phase} failed with exit code {code}", file=sys.stderr)
            return code
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.phase == "all":
        return _run_all(args)
    config = make_config(**_overrides(args))
    kw: dict[str, Any] = {}
    if args.phase == "report":
        kw = {"results_dir": args.results_dir, "report_path": args.report_path}
    run_phase(args.phase, args.runs_root / args.run_id, config, None, **kw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
