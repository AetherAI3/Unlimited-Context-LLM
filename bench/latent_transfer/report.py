# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Report phase: scores, paired bootstrap differences, verdicts, integrity and cost.

Writes ``results/<UTC ts>.json`` (never over an existing file) and ``REPORT.md`` (never over an
existing file either: a human fills in its Interpretation section). Floats are fine here; nothing in
this module feeds a canonical-JSON digest.
"""
from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from bench.latent_transfer import analysis, tasks
from bench.latent_transfer.common import (
    ARMS,
    ARMS_CAPSULE,
    REPORT_PATH,
    RESULTS_DIR,
    BackendFactory,
    git_pin,
    open_text,
    phase_record,
    read_json,
    read_jsonl,
    write_json,
)

MARGIN = 0.10
RESAMPLES = 2000
BOOT_SEED = 0
ITEMS_PER_DOC = 4
PROBES_PER_DOC = 7
PROBE_DOCS = 20
LOSS_WINDOW = 10
CONTROLS = ("blank", "shuffled", "mismatched", "no_prefix")
GATE_DIFFS = {f"paired-{c}": ("paired", c) for c in CONTROLS}
DIFFS = GATE_DIFFS | {"text_context-paired": ("text_context", "paired")}
FOLLOW_ARMS = ("shuffled", "mismatched")
CostPhases = ("prepare", "train", "encode", "evaluate")


def score_rows(rows: list[dict[str, Any]], ds: tasks.Dataset) -> list[dict[str, Any]]:
    """Add ``correct`` (against the doc's own truth) and, for the two wrong-source arms, ``follow``."""
    want = {(i.item_id, arm) for i in ds.eval_items for arm in ARMS}
    got = [(r["item_id"], r["arm"]) for r in rows]
    if len(got) != len(set(got)) or set(got) != want:
        raise ValueError("items.jsonl does not hold exactly one row per (item, arm)")
    items = {i.item_id: i for i in ds.eval_items}
    docs = {d.doc_id: d for d in ds.eval_docs}
    out = []
    for row in rows:
        item = items[row["item_id"]]
        rec = dict(row, correct=row["pred"] == tasks.gold_under(item, docs[item.doc_id]))
        if row["arm"] == "shuffled":
            source = docs[ds.shuffle_source[item.doc_id]]
            rec["follow"] = row["pred"] == tasks.gold_under(item, source)
        elif row["arm"] == "mismatched":
            rec["follow"] = row["pred"] == tasks.gold_under(item, ds.mismatched[item.doc_id])
        out.append(rec)
    return out


def _mean(values: Mapping[str, float]) -> float:
    return float(np.mean(list(values.values())))


def _ci(t: tuple[float, float, float, int]) -> dict[str, float | int]:
    return {"mean": t[0], "lo": t[1], "hi": t[2], "n": t[3]}


def analyse(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Arm means, paired differences, follow rates and tie rates; ``gate`` holds the raw H1/H2 inputs."""
    arm_means: dict[str, Any] = {}
    diffs: dict[str, Any] = {}
    follow: dict[str, Any] = {}
    gate: dict[str, Any] = {}
    for fam in tasks.FAMILIES:
        per = {a: analysis.per_doc(records, a, fam, expected_per_doc=ITEMS_PER_DOC) for a in ARMS}
        arm_means[fam] = {a: _mean(per[a]) for a in ARMS}
        raw = {name: analysis.bootstrap_diff(per[a], per[b], resamples=RESAMPLES, seed=BOOT_SEED)
               for name, (a, b) in DIFFS.items()}
        diffs[fam] = {name: _ci(t) for name, t in raw.items()}
        gate[fam] = {name: raw[name] for name in GATE_DIFFS}
        follow[fam] = {}
        for arm in FOLLOW_ARMS:
            rates = analysis.per_doc(records, arm, fam, key="follow", expected_per_doc=ITEMS_PER_DOC)
            zero = dict.fromkeys(rates, 0.0)
            follow[fam][arm] = _ci(analysis.bootstrap_diff(rates, zero, resamples=RESAMPLES, seed=BOOT_SEED))
    ties = {a: float(np.mean([r["tie"] for r in records if r["arm"] == a])) for a in ARMS}
    return {"arm_means": arm_means, "diffs": diffs, "follow": follow, "tie_rates": ties, "gate": gate}


def integrity(train: Mapping[str, Any], encode: Mapping[str, Any], ev: Mapping[str, Any],
              receipts_path: Path) -> dict[str, Any]:
    probes = ev["probes"]
    logged = sum(1 for r in read_jsonl(receipts_path) if r["outcome"] == "INJECTED")
    digests = {"train_before": train["base_digest_before"], "train_after": train["base_digest_after"],
               "encode": encode["base_digest"], "evaluate": ev["base_digest"]}
    expected_injected = ev["n_items"] * len(ARMS_CAPSULE)
    checks = {
        "probes_all_matched": bool(probes) and all(p["matched"] for p in probes),
        "probe_count_complete": len(probes) == PROBES_PER_DOC * min(PROBE_DOCS, ev["n_eval_docs"]),
        "every_injected_receipt_verified": ev["receipts_injected"] == ev["receipts_verified"] == expected_injected,
        "receipts_log_matches": logged == ev["receipts_injected"],
        "anchors_verified": ev["anchors_verified"] == ev["n_eval_docs"],
        "base_digest_unchanged": len(set(digests.values())) == 1,
        "same_adapter": train["adapter_id"] == encode["adapter_id"] == ev["adapter_id"],
    }
    return {"checks": checks, "base_digests": digests, "probes_total": len(probes),
            "probes_matched": sum(p["matched"] for p in probes), "receipts_injected": ev["receipts_injected"],
            "receipts_verified": ev["receipts_verified"], "receipts_logged_injected": logged,
            "anchors_verified": ev["anchors_verified"], "n_eval_docs": ev["n_eval_docs"],
            "H4": all(checks.values())}


def cost_block(prepared: Mapping[str, Any], train: Mapping[str, Any], encode: Mapping[str, Any],
               ev: Mapping[str, Any]) -> dict[str, Any]:
    arms = ev["arms"]
    flops = {"train_sender": train["flops"]["sender"], "train": train["flops"]["train"],
             "encode_sender": encode["flops"]["sender"], "evaluate": {a: arms[a]["flops"] for a in ARMS}}
    flops["evaluate_total"] = sum(flops["evaluate"].values())
    training = flops["train_sender"] + flops["train"]
    flops["total_incl_training"] = training + flops["encode_sender"] + flops["evaluate_total"]
    phases = {"prepare": prepared, "train": train, "encode": encode, "evaluate": ev}
    uses = ev["receipts_injected"]
    return {
        "wall_s": {p: r["wall_s"] for p, r in phases.items()} | {"total": sum(r["wall_s"] for r in phases.values())},
        "peak_rss_bytes": {p: r["peak_rss_bytes"] for p, r in phases.items()},
        "flops": flops,
        "capsule_bytes_per_handoff": encode["bytes_per_capsule"],
        "mean_receiver_tokens_per_item": {a: arms[a]["receiver_tokens"] / arms[a]["items"]
                                          for a in ("paired", "text_context")},
        "training_flops_amortised_per_capsule_use": training / uses if uses else None,
        "scored_capsule_uses": uses,
        "device_info": ev["device_info"],
    }


def _loss_summary(train: Mapping[str, Any]) -> dict[str, Any]:
    losses = train["stats"]["losses"]
    return {"steps": len(losses), "first": float(np.mean(losses[:LOSS_WINDOW])),
            "last": float(np.mean(losses[-LOSS_WINDOW:]))}


def write_new_json(results_dir: Path, stamp: str, payload: Any) -> Path:
    """Write ``<stamp>.json`` (or ``<stamp>-N.json``), never replacing a file that exists."""
    for n in range(1000):
        path = results_dir / (f"{stamp}.json" if n == 0 else f"{stamp}-{n}.json")
        if not path.exists():
            write_json(path, payload)
            return path
    raise FileExistsError(f"no free results file name for {stamp}")


def _p(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _ci_rows(prefix: str, table: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [f"| {prefix} | {name} | {_p(x['mean'])} | [{_p(x['lo'])}, {_p(x['hi'])}] | {x['n']} |"
            for name, x in table.items()]


def _cost_rows(c: Mapping[str, Any]) -> list[str]:
    fl, w, rss, tok = c["flops"], c["wall_s"], c["peak_rss_bytes"], c["mean_receiver_tokens_per_item"]
    amort = c["training_flops_amortised_per_capsule_use"]
    rows = [
        ("wall s (prepare / train / encode / evaluate)", " / ".join(f"{w[p]:.1f}" for p in CostPhases)),
        ("wall s total", f"{w['total']:.1f}"),
        ("peak RSS bytes (prepare / train / encode / evaluate)", " / ".join(str(rss[p]) for p in CostPhases)),
        ("FLOPs sender during training (est.)", f"{fl['train_sender']:.3e}"),
        ("FLOPs training (est.)", f"{fl['train']:.3e}"),
        ("FLOPs encode, sender (est.)", f"{fl['encode_sender']:.3e}"),
        ("FLOPs evaluate, all arms (est.)", f"{fl['evaluate_total']:.3e}"),
        ("FLOPs total including training (est.)", f"{fl['total_incl_training']:.3e}"),
        ("capsule bytes per handoff", f"{c['capsule_bytes_per_handoff']:.0f}"),
        ("mean receiver tokens per item, paired", f"{tok['paired']:.1f}"),
        ("mean receiver tokens per item, text_context", f"{tok['text_context']:.1f}"),
        ("training FLOPs per scored capsule use (est.)", "n/a" if amort is None else f"{amort:.3e}"),
        ("scored capsule uses", str(c["scored_capsule_uses"])),
    ]
    return [f"| {k} | {x} |" for k, x in rows]


def render_markdown(res: Mapping[str, Any]) -> str:
    cfg, v, integ, pins = res["config"], res["verdicts"], res["integrity"], res["pins"]
    lines = [
        "# Latent-transfer report (G-W7)", "",
        f"Generated {res['generated_utc']} from run `{res['run_id']}`. "
        f"Model `{pins['model_ref']}` @ `{pins['model_revision']}`, "
        f"code at git `{pins['git']['head']}` (dirty: {pins['git']['dirty']}).", "",
        "## Config", "", "| key | value |", "|---|---|", *[f"| {k} | `{cfg[k]}` |" for k in sorted(cfg)], "",
        f"Dataset digest `{res['digests']['dataset']}`; adapter `{res['digests']['adapter_id']}`; "
        f"training loss {_p(res['train_loss']['first'])} -> {_p(res['train_loss']['last'])} "
        f"over {res['train_loss']['steps']} steps.", "",
        "## Arm means (per-document accuracy, mean over documents)", ""]
    for fam, means in res["arm_means"].items():
        lines += [f"**{fam}**", "", "| arm | accuracy | tie rate |", "|---|---|---|",
                  *[f"| {a} | {_p(means[a])} | {_p(res['tie_rates'][a])} |" for a in ARMS], ""]
    lines += ["## Paired differences (95% paired bootstrap CI over documents)", "",
              "| family | contrast | mean | 95% CI | n docs |", "|---|---|---|---|---|"]
    for fam, d in res["diffs"].items():
        lines += _ci_rows(fam, d)
    lines += ["", "## Hypotheses", "",
              f"Promotion margin: lower CI bound of every paired-minus-control difference >= {MARGIN}.", "",
              f"- H1 (unseen question phrasings): **{'PASS' if v['H1'] else 'FAIL'}**",
              f"- H2 (unseen yes/no task family): **{'PASS' if v['H2'] else 'FAIL'}**",
              f"- H4 (integrity): **{'PASS' if v['H4'] else 'FAIL'}**", "",
              "## Follow rates (share of items answered as the wrong-source document dictates)", "",
              "| family | arm | mean | 95% CI | n docs |", "|---|---|---|---|---|"]
    for fam, d in res["follow"].items():
        lines += _ci_rows(fam, d)
    lines += ["", "## Integrity", "", "| check | result |", "|---|---|",
              *[f"| {k} | {'ok' if ok else 'FAILED'} |" for k, ok in integ["checks"].items()], "",
              f"{integ['probes_matched']}/{integ['probes_total']} probes refused with the expected code; "
              f"{integ['receipts_verified']}/{integ['receipts_injected']} injected receipts verified; "
              f"{integ['anchors_verified']}/{integ['n_eval_docs']} anchors verified.", "",
              "## Cost", "", "| item | value |", "|---|---|", *_cost_rows(res["cost"])]
    di = res["cost"]["device_info"]
    lines += ["", f"Device: {di['platform']}, {di['processor']}, {di['cpu_count']} logical CPUs, "
              f"torch {di['torch']} ({di['torch_threads']} threads), transformers {di['transformers']}, "
              f"numpy {di['numpy']}, device `{di['device']}`. FLOPs are estimates "
              "(2 x parameters x tokens; attention-score FLOPs ignored).", "", "## Interpretation", "", ""]
    return "\n".join(lines)


def report(run_dir: Path, config: dict[str, Any], backend_factory: BackendFactory | None = None, *,
           results_dir: Path | None = None, report_path: Path | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    results_dir = RESULTS_DIR if results_dir is None else results_dir
    report_path = REPORT_PATH if report_path is None else report_path
    if report_path.exists():
        raise FileExistsError(f"{report_path} exists; pass another --report-path or move it first")
    prepared, train, encode, ev = (read_json(run_dir / f"{p}.json") for p in ("prepare", "train", "encode", "evaluate"))
    ds = tasks.from_json(read_json(run_dir / "dataset.json"))
    if tasks.dataset_digest(ds) != prepared["dataset_digest"]:
        raise RuntimeError("dataset.json does not match the digest recorded by prepare")
    res = analyse(score_rows(read_jsonl(run_dir / "items.jsonl"), ds))
    gate = res.pop("gate")
    integ = integrity(train, encode, ev, run_dir / "receipts.jsonl")
    res.update(
        verdicts={"H1": analysis.verdict(gate["unseen_template"], MARGIN),
                  "H2": analysis.verdict(gate["yes_no"], MARGIN), "H4": integ["H4"]},
        integrity=integ, cost=cost_block(prepared, train, encode, ev), train_loss=_loss_summary(train),
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), run_id=run_dir.name, config=config,
        margin=MARGIN, resamples=RESAMPLES, bootstrap_seed=BOOT_SEED,
        pins={"model_ref": config["model_ref"], "model_revision": config["model_revision"], "git": git_pin()},
        digests={"dataset": prepared["dataset_digest"], "codebook": train["codebook_digest"],
                 "adapter_id": train["adapter_id"], "base": train["base_digest_before"]})
    path = write_new_json(results_dir, time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), res)
    with open_text(report_path, "x") as fh:
        fh.write(render_markdown(res))
    record = phase_record("report", config, time.perf_counter() - t0, results_path=str(path),
                          report_path=str(report_path), verdicts=res["verdicts"])
    write_json(run_dir / "report.json", record)
    return res | {"results_path": str(path)}
