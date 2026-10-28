# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Shared plumbing for the latent-transfer run phases: JSON IO, phase records, git pins."""
from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from bench.latent_transfer import cost

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = Path(__file__).resolve().parent
RUNS_ROOT = BENCH_DIR / "runs"
RESULTS_DIR = BENCH_DIR / "results"
REPORT_PATH = BENCH_DIR / "REPORT.md"
ARMS_CAPSULE = ("paired", "blank", "shuffled", "mismatched")
PROBE_DOCS = 20  # integrity probes run on the first PROBE_DOCS evaluation documents
ARMS = (*ARMS_CAPSULE, "no_prefix", "text_context")
BackendFactory = Callable[[dict[str, Any]], Any]

# Training settings every phase record must agree on (checked by the report phase).
TRAIN_KEYS = ("steps", "batch_size", "lr", "train_seed")
# Config keys that fix the dataset and the model pair; every phase after ``prepare`` must agree.
PINNED_KEYS = (
    "model_ref", "model_revision", "dataset_seed", "n_train", "n_eval", "slots", "d_attn",
    "sender_layer", "system", "storage_dtype",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def append_jsonl(fh: Any, row: Mapping[str, Any]) -> None:
    fh.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def open_text(path: Path, mode: str = "w") -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, mode, encoding="utf-8", newline="\n")


def phase_record(phase: str, config: Mapping[str, Any], wall_s: float, **extra: Any) -> dict[str, Any]:
    """Fields every phase JSON carries, plus the phase's own measurements."""
    return {
        "phase": phase,
        "config": dict(config),
        "wall_s": wall_s,
        "device_info": cost.device_info(),
        "peak_rss_bytes": cost.peak_rss_bytes(),
        **extra,
    }


def check_pinned(config: Mapping[str, Any], prepared: Mapping[str, Any]) -> None:
    """Abort when this phase's config differs from the one ``prepare`` froze the dataset with."""
    frozen = prepared["config"]
    drift = {k: (frozen.get(k), config.get(k)) for k in PINNED_KEYS if frozen.get(k) != config.get(k)}
    if drift:
        raise RuntimeError(f"config differs from prepare.json (key: (prepared, now)): {drift}")


def git_pin() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout.strip()

    status = run("status", "--porcelain")
    return {"head": run("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def flop_pair(backend: Any) -> tuple[int, int]:
    blocks, head = backend.flop_params()
    return int(blocks), int(head)


def chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i: i + size]


def load_pinned_model(model_ref: str, revision: str) -> tuple[Any, Any]:
    """Load the pinned checkpoint from the local HF cache, streaming one tensor at a time.

    ``from_pretrained`` memory-maps ``model.safetensors``; on this shared Windows host that
    page-in failed with an access violation under memory pressure, and reading the whole file
    first ran out of memory. This reads the safetensors header, then each tensor's byte range,
    converting to float32 as it goes, so peak memory is the float32 model plus one tensor. The
    weights are the same (the base digest is recorded by every phase).
    """
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_ref, revision=revision)
    config = AutoConfig.from_pretrained(model_ref, revision=revision)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    weights = Path(hf_hub_download(model_ref, "model.safetensors", revision=revision))
    state = dict(_stream_safetensors(weights))
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    del state
    tied = getattr(config, "tie_word_embeddings", False)
    if unexpected or set(missing) - ({"lm_head.weight"} if tied else set()):
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    if tied:
        model.tie_weights()
    # Non-persistent buffers (rotary inv_freq) are not in the checkpoint: rebuild those modules on CPU.
    for name, module in list(model.named_modules()):
        if any(b.is_meta for b in module.buffers(recurse=False)):
            parent, _, attr = name.rpartition(".")
            setattr(model.get_submodule(parent) if parent else model, attr, type(module)(config=config))
    if any(t.is_meta for t in (*model.parameters(), *model.buffers())):
        raise RuntimeError("tensors left on the meta device after loading")
    return model.eval(), tok


_SAFETENSORS_DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}


def _stream_safetensors(path: Path) -> Iterable[tuple[str, Any]]:
    """Yield ``(name, float32 tensor)`` from a safetensors file, reading one tensor at a time."""
    import struct

    import torch

    with open(path, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len))
        base = 8 + header_len
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            fh.seek(base + start)
            buf = bytearray(fh.read(end - start))
            dtype = getattr(torch, _SAFETENSORS_DTYPES[meta["dtype"]])
            raw = torch.frombuffer(buf, dtype=dtype) if buf else torch.empty(0, dtype=dtype)
            yield name, raw.reshape(meta["shape"]).to(torch.float32)


def first_token(tokenizer: Any, text: str) -> int:
    """First token id of ``text`` as the model would emit it right after the assistant header."""
    return int(tokenizer(text, add_special_tokens=False).input_ids[0])
