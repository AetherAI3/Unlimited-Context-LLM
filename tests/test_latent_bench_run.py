# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""End-to-end latent-transfer run on the tiny random model (no download, no real checkpoint)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("pydantic")

from aether_context.latent.hf_soft_prefix import HFSoftPrefixBackend  # noqa: E402
from bench.latent_transfer import run  # noqa: E402
from bench.latent_transfer.common import ARMS, read_json, read_jsonl  # noqa: E402
from bench.latent_transfer.evaluate import PROBE_EXPECTED, EvaluationAbort, evaluate  # noqa: E402
from tests.latent_tiny import tiny_pair  # noqa: E402

N_TRAIN, N_EVAL, ITEMS_PER_DOC = 6, 3, 8
CONFIG = run.make_config(n_train=N_TRAIN, n_eval=N_EVAL, steps=2, batch_size=2, score_batch=4, slots=2,
                         d_attn=16, system="Answer with one word", model_ref="tiny/llama", model_revision="0")
RECEIVER_FILES = ("questions.json", "anchors.json", "capsule_index.json", "projector.pt")


def factory(config):
    """Same frozen weights on every call, so adapter ids and codebooks agree across phases."""
    model, tok = tiny_pair(seed=0)
    return HFSoftPrefixBackend(model, tok, model_id="tiny/llama@0", slots=config["slots"],
                               d_attn=config["d_attn"], system=config["system"])


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    root = tmp_path_factory.mktemp("bench")
    run_dir = root / "runs" / "t1"
    run.prepare(run_dir, CONFIG, factory, check_tokens=False)
    run.train(run_dir, CONFIG, factory)
    run.encode(run_dir, CONFIG, factory)
    evaluate(run_dir, CONFIG, factory)
    results = run.run_phase("report", run_dir, CONFIG, factory, results_dir=root / "results",
                            report_path=root / "REPORT.md")
    return {"root": root, "run_dir": run_dir, "report": results}


def receiver_copy(src: Path, dst: Path) -> Path:
    """A run directory holding only what the receiver may see."""
    dst.mkdir(parents=True)
    for name in RECEIVER_FILES:
        shutil.copy(src / name, dst / name)
    shutil.copytree(src / "capsules", dst / "capsules")
    return dst


def test_artifacts_exist(finished):
    d = finished["run_dir"]
    for name in ("dataset.json", "questions.json", "anchors.json", "prepare.json", "train.json", "train_log.jsonl",
                 "projector.pt", "capsule_index.json", "encode.json", "items.jsonl", "receipts.jsonl",
                 "evaluate.json", "report.json"):
        assert (d / name).exists(), name
    for phase in ("prepare", "train", "encode", "evaluate"):
        rec = read_json(d / f"{phase}.json")
        assert rec["config"]["steps"] == 2 and rec["wall_s"] >= 0 and "device_info" in rec and "peak_rss_bytes" in rec
    assert len(list((d / "capsules").glob("*.bin"))) == N_EVAL * 3


def test_questions_carry_no_gold_field(finished):
    for q in read_json(finished["run_dir"] / "questions.json"):
        assert set(q) == {"item_id", "doc_id", "family", "attribute", "question", "candidates"}
    for anchor in read_json(finished["run_dir"] / "anchors.json").values():
        assert "text" in anchor


def test_items_cover_every_arm(finished):
    rows = read_jsonl(finished["run_dir"] / "items.jsonl")
    for arm in ARMS:
        assert sum(r["arm"] == arm for r in rows) == N_EVAL * ITEMS_PER_DOC
    capsule_rows = [r for r in rows if r["arm"] in ("paired", "blank", "shuffled", "mismatched")]
    assert all(r["receipt_id"] for r in capsule_rows)
    assert all(r["receipt_id"] is None for r in rows if r not in capsule_rows)


def test_all_probes_refused_with_expected_codes(finished):
    probes = read_json(finished["run_dir"] / "evaluate.json")["probes"]
    assert len(probes) == len(PROBE_EXPECTED) * N_EVAL
    assert all(p["matched"] and p["observed"] == PROBE_EXPECTED[p["probe"]] for p in probes)


def test_every_injected_receipt_verified(finished):
    ev = read_json(finished["run_dir"] / "evaluate.json")
    assert ev["receipts_injected"] == ev["receipts_verified"] == N_EVAL * ITEMS_PER_DOC * 4
    assert ev["anchors_verified"] == N_EVAL
    receipts = read_jsonl(finished["run_dir"] / "receipts.jsonl")
    assert sum(r["outcome"] == "INJECTED" for r in receipts) == ev["receipts_injected"]
    assert sum(r["outcome"] == "REFUSED" for r in receipts) == len(ev["probes"])


def test_report_has_verdicts_and_markdown(finished):
    res = finished["report"]
    assert set(res["verdicts"]) == {"H1", "H2", "H4"} and res["verdicts"]["H4"] is True
    stored = json.loads(Path(res["results_path"]).read_text(encoding="utf-8"))
    assert set(stored["verdicts"]) == {"H1", "H2", "H4"} and stored["pins"]["model_revision"] == "0"
    assert stored["cost"]["flops"]["total_incl_training"] > 0
    md = (finished["root"] / "REPORT.md").read_bytes().decode("utf-8")
    assert "## Interpretation" in md and "H1" in md
    assert "\r" not in md


def test_report_refuses_to_overwrite_report(finished):
    with pytest.raises(FileExistsError):
        run.run_phase("report", finished["run_dir"], CONFIG, factory, results_dir=finished["root"] / "r2",
                      report_path=finished["root"] / "REPORT.md")
    assert not (finished["root"] / "r2").exists()


def test_evaluate_needs_only_receiver_files(finished, tmp_path):
    view = receiver_copy(finished["run_dir"], tmp_path / "recv")
    ev = evaluate(view, CONFIG, factory)
    assert ev["receipts_injected"] == ev["receipts_verified"]
    assert not (view / "dataset.json").exists()


def test_scored_arm_refusal_aborts(finished, tmp_path):
    view = receiver_copy(finished["run_dir"], tmp_path / "recv")
    index = read_json(view / "capsule_index.json")
    target = view / "capsules" / f"{next(iter(index.values()))['mismatched']}.bin"
    raw = target.read_bytes()
    target.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    with pytest.raises(EvaluationAbort, match="TENSOR_DIGEST_MISMATCH"):
        evaluate(view, CONFIG, factory)


def test_later_phase_rejects_drifted_config(finished):
    drifted = {**CONFIG, "n_eval": N_EVAL + 1}
    with pytest.raises(RuntimeError, match="differs from prepare.json"):
        run.train(finished["run_dir"], drifted, factory)


def test_prepare_aborts_on_colliding_candidate_tokens(tmp_path):
    # The tiny tokenizer maps most candidate words to <unk>, so first tokens collide.
    with pytest.raises(ValueError, match="collide"):
        run.prepare(tmp_path / "r", CONFIG, factory, check_tokens=True)
    assert not (tmp_path / "r" / "dataset.json").exists()


def test_cli_rejects_unknown_config_keys_and_parses_overrides():
    with pytest.raises(KeyError):
        run.make_config(bogus=1)
    args = run._parser().parse_args(["train", "--run-id", "x", "--steps", "5", "--lr", "0.5", "--n-eval", "9",
                                      "--batch-size", "8"])
    assert run._overrides(args) == {"steps": 5, "lr": 0.5, "n_eval": 9, "batch_size": 8}
