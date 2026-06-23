"""swe_eval — SWE-bench codepro eval (Phase A: generation). Drives dsv4-pro through an
agentic file-tool loop over a per-instance checkout, off vs codepro-engine arms, and writes
SWE-bench predictions JSONL. Phase B scoring is bench/swe_scoring.py (official swebench).

Run (dry-run, no key): python -m bench.swe_eval --dry-run
Run (live): OPENROUTER_API_KEY=... python -m bench.swe_eval --instances 1 --arms off,codepro
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from aether_context.encoder import StaticEncoder
from aether_context.session import Session
from bench.api_eval import cached_tokens, cost_usd
from bench.swe_tools import RepoTools, prepare_checkout

DEFAULT_MODEL = "deepseek/deepseek-v4-pro"
DATASET = "princeton-nlp/SWE-bench_Lite"
MODEL_NAME = "dsv4pro-codepro"   # stamped into predictions model_name_or_path


@dataclass
class SweConfig:
    model: str = DEFAULT_MODEL
    arms: tuple[str, ...] = ("off", "codepro")
    instances: int = 0            # 0 = all of lite (300); N = first N (sorted)
    window: int = 8192            # off-arm truncation window
    max_steps: int = 50           # tool steps per instance ("max reasoning" budget)
    pool_gb: int = 50             # codepro overpool reach
    turbovec_bits: int = 8        # codepro TurboVec quant (0/4/8)
    mpo_chain: bool = True        # codepro MPO chain
    recall_k: int = 8             # slices recalled per turn (codepro)
    price_in: float = 0.3
    price_out: float = 1.2
    max_usd: float = 25.0         # hard global spend cap (shared across arms+instances)
    cost_spike_usd: float = 0.50
    dry_run: bool = False
    out_dir: Path = field(default_factory=lambda: Path("runs/swe_eval"))
    work_dir: Path = field(default_factory=lambda: Path("runs/swe_eval/checkouts"))


def select_instances(rows: list[dict], n: int) -> list[dict]:
    """Deterministic instance set: sort by instance_id, take first n (n=0 -> all)."""
    ordered = sorted(rows, key=lambda r: r["instance_id"])
    return ordered if n <= 0 else ordered[:n]


def load_lite(cfg: SweConfig) -> list[dict]:
    """Load SWE-bench-lite test split. Dry-run -> a synthetic 2-instance stand-in."""
    if cfg.dry_run:
        return [
            {"instance_id": "syn__repo-1", "repo": "syn/repo", "base_commit": "HEAD",
             "problem_statement": "Fix add() to add instead of subtract."},
            {"instance_id": "syn__repo-2", "repo": "syn/repo", "base_commit": "HEAD",
             "problem_statement": "Fix VALUE to 2."},
        ]
    from datasets import load_dataset
    ds = load_dataset(DATASET, split="test")
    return [dict(r) for r in ds]


def _truncate(messages: list[dict], window_tokens: int) -> list[dict]:
    """OFF arm: keep system + a tail of messages fitting ~window (chars/4 estimate)."""
    from aether_context.tokenizer import estimate
    budget = window_tokens
    head, tail = messages[:1], []
    for m in reversed(messages[1:]):
        c = estimate(m.get("content") or "")
        if budget - c < 0:
            break
        budget -= c
        tail.insert(0, m)
    return head + tail


_SYS = ("You are an expert software engineer fixing a real bug in a repository. First INVESTIGATE "
        "with read_file, grep, and list_dir to locate the cause. You have a limited tool budget, so "
        "be efficient — do not re-read the same files. When you have enough understanding, stop "
        "calling tools (reply with a brief note) and you will then be asked to produce the fix.")

# The investigation phase exposes ONLY the read-only tools; the model produces the actual
# fix in the forced patch turn below (dsv4-pro reliably emits a unified diff but will not
# drive a stateful editor — it investigates until the budget is gone otherwise).
_READONLY_SCHEMA = [t for t in RepoTools.TOOLS_SCHEMA
                    if t["function"]["name"] in ("read_file", "grep", "list_dir")]

_PATCH_INSTRUCTION = (
    "You have finished investigating and can no longer read files. Call the submit_patch tool "
    "now with your complete fix. The `edits` argument must contain one or more blocks in EXACTLY "
    "this format, one per change:\n\n"
    "<path/relative/to/repo/root>\n"
    "<<<<<<< SEARCH\n"
    "<exact lines that currently exist in the file>\n"
    "=======\n"
    "<replacement lines>\n"
    ">>>>>>> REPLACE\n\n"
    "The SEARCH text must match the current file content EXACTLY (whitespace included), from what "
    "you read. submit_patch is the ONLY tool available — you must call it.")

# Phase-2 exposes ONLY this tool. dsv4-pro compulsively calls tools and will not write from
# memory when asked in plain text (it keeps trying to re-read); giving it a single submit tool
# turns that compulsion into the patch submission.
_SUBMIT_SCHEMA = [{"type": "function", "function": {
    "name": "submit_patch",
    "description": "Submit the final fix as SEARCH/REPLACE edit blocks.",
    "parameters": {"type": "object",
                   "properties": {"edits": {"type": "string",
                                            "description": "One or more SEARCH/REPLACE blocks."}},
                   "required": ["edits"]}}}]


def _submit_edits(out: dict) -> Optional[str]:
    """Return the `edits` string from a submit_patch tool call, or None."""
    for tc in (out.get("tool_calls") or []):
        fn = tc.get("function", {})
        if fn.get("name") == "submit_patch":
            try:
                return (json.loads(fn.get("arguments") or "{}") or {}).get("edits")
            except json.JSONDecodeError:
                return None
    return None

# Aider-style SEARCH/REPLACE block. dsv4-pro will not write a line-numbered unified diff from
# memory (it just keeps trying to re-read), but it can reproduce an exact snippet + replacement.
# Applying via edit_file makes the off-vs-codepro gap a function of whether the model REMEMBERS
# the exact code: off (truncated transcript) misremembers -> SEARCH miss; codepro recalls it.
_BLOCK_RE = re.compile(
    r"(?P<path>[^\n<>=]+?)\n<{5,7} SEARCH\n(?P<search>.*?)\n={5,7}\n(?P<replace>.*?)\n>{5,7} REPLACE",
    re.S)


def _parse_blocks(text: str) -> list[tuple[str, str, str]]:
    """Extract (path, search, replace) edit blocks from the model's patch-turn text."""
    if not text:
        return []
    return [(m.group("path").strip().strip("`").strip(),
             m.group("search"), m.group("replace")) for m in _BLOCK_RE.finditer(text)]


def _empty_record(arm: str, inst: dict, halted: str) -> dict:
    return {
        "instance_id": inst["instance_id"],
        "model_name_or_path": MODEL_NAME if arm == "codepro" else f"{MODEL_NAME}-off",
        "model_patch": "", "arm": arm, "cost_usd": 0.0, "cached_tokens": 0,
        "tool_calls": 0, "redundant_tool_calls": 0, "patch_nonempty": False,
        "halted": halted,
    }


def run_instance(arm: str, inst: dict, cfg: SweConfig, chat, budget: dict,
                 *, repo_url: Optional[str] = None) -> dict:
    """Run ONE SWE instance under ONE arm. Returns a predictions record incl. model_patch.
    OFF: window-truncated transcript. CODEPRO: Session(overpool+turbovec+chain) recall."""
    url = repo_url or f"https://github.com/{inst['repo']}.git"
    workdir = cfg.work_dir / arm / inst["instance_id"]
    try:
        checkout = prepare_checkout(url, inst["base_commit"], workdir)
    except Exception as e:  # clone/checkout failed — record empty patch, keep the batch alive
        return _empty_record(arm, inst, f"checkout_error: {type(e).__name__}")
    tools = RepoTools(checkout)

    session: Optional[Session] = None
    encoder = StaticEncoder(dim=256)
    if arm == "codepro":
        session = Session("swe", pool_gb=cfg.pool_gb,
                          pool_dir=cfg.out_dir / f"pool_{inst['instance_id']}_{arm}",
                          context_window=cfg.window, mpo_chain=cfg.mpo_chain,
                          pool_quantize=cfg.turbovec_bits)

    user = (f"Repository: {inst['repo']} @ {inst['base_commit'][:10]}\n\n"
            f"Problem:\n{inst['problem_statement']}\n\nInvestigate and fix it.")
    transcript: list[dict] = [{"role": "system", "content": _SYS},
                              {"role": "user", "content": user}]
    cost = 0.0
    cached = 0
    halted: Optional[str] = None

    def _account(out: dict) -> bool:
        """Tally cost/cached for one chat call. Returns False if a cost spike trips."""
        nonlocal cost, cached, halted
        usage = out.get("usage", {}) or {}
        call_cost = cost_usd(usage, price_in=cfg.price_in, price_out=cfg.price_out)
        cost += call_cost
        budget["spent"] += call_cost
        cached += cached_tokens(usage)
        if call_cost > cfg.cost_spike_usd:
            halted = "cost_spike"
            return False
        return True

    # ── Phase 1: investigation (read-only tools, bounded) ──────────────
    for _step in range(cfg.max_steps):
        if budget["spent"] >= cfg.max_usd:
            halted = "budget_cap"
            break
        if session is not None:
            qvec = encoder.encode(inst["problem_statement"])
            recalled = session._cold_retrieve(session._key(), qvec, cfg.recall_k)
            mem = "\n".join(f"[mem] {s.text}" for s in recalled) or "(empty)"
            convo = _truncate([transcript[0],
                               {"role": "system", "content": f"Working memory:\n{mem}"}]
                              + transcript[1:], cfg.window)
        else:
            convo = _truncate(transcript, cfg.window)

        out = chat.chat(convo, tools=_READONLY_SCHEMA)
        if not _account(out):
            break

        tcs = out.get("tool_calls") or []
        if not tcs:
            transcript.append({"role": "assistant", "content": out.get("content") or "ready"})
            break  # model signalled it's done investigating -> go produce the patch
        transcript.append({"role": "assistant", "content": out.get("content"),
                           "tool_calls": tcs})
        for tc in tcs:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            try:
                result = tools.dispatch(fn.get("name", ""), args)
            except Exception as e:  # a tool crash must not kill the instance
                result = {"error": f"{type(e).__name__}: {e}"}
            rjson = json.dumps(result)[:1500]
            transcript.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                               "content": rjson})
            if session is not None:
                session.remember(f"{fn.get('name')}({args}): {rjson}")

    # ── Phase 2: forced patch turn(s) — SEARCH/REPLACE blocks, applied to the checkout ──
    # Up to 2 attempts: the second gets feedback on which SEARCH blocks didn't match (so a
    # near-miss can self-correct). model_patch = git diff of whatever applied.
    applied = 0
    instruction = _PATCH_INSTRUCTION
    for _attempt in range(2):
        if halted == "cost_spike" or budget["spent"] >= cfg.max_usd:
            break
        if session is not None:
            qvec = encoder.encode(inst["problem_statement"])
            recalled = session._cold_retrieve(session._key(), qvec, cfg.recall_k)
            mem = "\n".join(f"[mem] {s.text}" for s in recalled) or "(empty)"
            convo = _truncate([transcript[0],
                               {"role": "system", "content": f"Working memory:\n{mem}"}]
                              + transcript[1:], cfg.window)
        else:
            convo = _truncate(transcript, cfg.window)
        convo = convo + [{"role": "user", "content": instruction}]
        out = chat.chat(convo, tools=_SUBMIT_SCHEMA)
        if not _account(out):
            break
        # Prefer the submit_patch tool arg; fall back to any blocks in free text.
        edits = _submit_edits(out)
        blocks = _parse_blocks(edits if edits is not None else (out.get("content") or ""))
        if not blocks:
            break  # nothing usable -> give up (empty patch = unresolved)
        fails = []
        for path, search, replace in blocks:
            res = tools.edit_file(path, search, replace)
            if isinstance(res, dict) and res.get("ok"):
                applied += 1
            else:
                fails.append(f"{path}: {res.get('error') if isinstance(res, dict) else 'failed'}")
        if applied and not fails:
            break
        instruction = (_PATCH_INSTRUCTION + "\n\nThese blocks did NOT match the file and were "
                       "skipped — fix the SEARCH text to match EXACTLY and resend ALL needed "
                       "blocks:\n" + "\n".join(fails))

    patch = tools.current_patch()
    if session is not None:
        session.close()
    return {
        "instance_id": inst["instance_id"],
        "model_name_or_path": MODEL_NAME if arm == "codepro" else f"{MODEL_NAME}-off",
        "model_patch": patch,
        "arm": arm,
        "cost_usd": round(cost, 6),
        "cached_tokens": cached,
        "tool_calls": tools.calls,
        "redundant_tool_calls": tools.redundant,
        "patch_nonempty": bool(patch.strip()),
        "halted": halted,
    }


def _repo_url(inst: dict) -> str:
    """GitHub clone URL for an instance (monkeypatched to file:// in tests)."""
    return f"https://github.com/{inst['repo']}.git"


def _done_ids(path: Path) -> set[str]:
    """instance_ids already written to a predictions JSONL (resume support)."""
    if not path.exists():
        return set()
    ids = set()
    for ln in path.read_text(encoding="utf-8").splitlines():
        try:
            ids.add(json.loads(ln)["instance_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return ids


def _make_live_chat(cfg: SweConfig):
    from aether_context.local_llm import OpenAICompatLLM
    return OpenAICompatLLM(cfg.model, context_window=cfg.window)


def run_swe_eval(cfg: SweConfig, instances: Optional[list[dict]] = None,
                 chat_factory=None) -> dict:
    """Drive every (instance, arm), honoring the shared $ cap and skipping done work.
    Appends each record to runs/swe_eval/predictions_<arm>.jsonl. Resumable."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    rows = instances if instances is not None else select_instances(load_lite(cfg), cfg.instances)
    chat_factory = chat_factory or (_make_live_chat if not cfg.dry_run else (lambda c: None))
    budget = {"spent": 0.0}
    summary: dict[str, Any] = {"arms": {a: {"done": 0, "patched": 0, "cost": 0.0}
                                        for a in cfg.arms},
                               "skipped": 0, "halted": None}

    for arm in cfg.arms:
        pred_path = cfg.out_dir / f"predictions_{arm}.jsonl"
        done = _done_ids(pred_path)
        chat = chat_factory(cfg)
        for inst in rows:
            if inst["instance_id"] in done:
                summary["skipped"] += 1
                continue
            if budget["spent"] >= cfg.max_usd:
                summary["halted"] = "budget_cap"
                break
            rec = run_instance(arm, inst, cfg, chat, budget, repo_url=_repo_url(inst))
            with pred_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            summary["arms"][arm]["done"] += 1
            summary["arms"][arm]["patched"] += int(rec["patch_nonempty"])
            summary["arms"][arm]["cost"] = round(
                summary["arms"][arm]["cost"] + rec["cost_usd"], 6)
            if rec.get("halted") in ("budget_cap", "cost_spike"):
                summary["halted"] = rec["halted"]
                break
        if summary["halted"]:
            break

    summary["total_cost_usd"] = round(budget["spent"], 6)
    (cfg.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _build_config(argv: Optional[list[str]] = None) -> SweConfig:
    p = argparse.ArgumentParser(prog="swe_eval",
                                description="SWE-bench codepro eval (generation phase).")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--arms", default="off,codepro")
    p.add_argument("--instances", type=int, default=0, help="0 = all lite (300); N = first N")
    p.add_argument("--window", type=int, default=8192)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--pool-gb", type=int, default=50)
    p.add_argument("--turbovec-bits", type=int, default=8, choices=(0, 4, 8))
    mpo = p.add_mutually_exclusive_group()
    mpo.add_argument("--mpo-chain", dest="mpo_chain", action="store_true", default=True)
    mpo.add_argument("--no-mpo-chain", dest="mpo_chain", action="store_false")
    p.add_argument("--recall-k", type=int, default=8)
    p.add_argument("--max-usd", type=float, default=25.0)
    p.add_argument("--out", default="runs/swe_eval")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    return SweConfig(
        model=a.model, arms=tuple(x.strip() for x in a.arms.split(",") if x.strip()),
        instances=a.instances, window=a.window, max_steps=a.max_steps, pool_gb=a.pool_gb,
        turbovec_bits=a.turbovec_bits, mpo_chain=a.mpo_chain, recall_k=a.recall_k,
        max_usd=a.max_usd, dry_run=a.dry_run, out_dir=Path(a.out),
        work_dir=Path(a.out) / "checkouts")


def main(argv: Optional[list[str]] = None) -> int:
    cfg = _build_config(argv)
    if not cfg.dry_run and not (os.environ.get("OPENROUTER_API_KEY")
                                or os.environ.get("OPENAI_API_KEY")):
        print("No OPENROUTER_API_KEY — use --dry-run to exercise the harness.")
        return 0
    summary = run_swe_eval(cfg)
    print(json.dumps(summary, indent=2))
    print(f"predictions -> {cfg.out_dir}/predictions_<arm>.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
