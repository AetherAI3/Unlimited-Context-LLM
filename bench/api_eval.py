# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""api_eval — autonomous OpenRouter eval: a lean agent loop works real GitHub issues.

Measures, across one session, ON vs OFF the engine: cost ($), tools called (+ redundant),
coherence drift (per-turn recall correctness), and work outcome (correct triage vs the real
issue labels). Emits JSON + CSV + an optional start->finish line graph.

Run (autonomous): OPENROUTER_API_KEY=... python -m bench.api_eval --model <slug> --plot
Dry-run (no key): python -m bench.api_eval --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from aether_context.encoder import StaticEncoder
from aether_context.session import Session
from aether_context.tokenizer import estimate

DEFAULT_MODEL = "deepseek/deepseek-v3.2"  # cheap DeepSeek reasoning slug; override with --model
DEFAULT_REPO = "microsoft/vscode"
_GH_API = "https://api.github.com"


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
def fetch_issues(repo: str, n: int, cache_dir: Path) -> list[dict]:
    """Fetch up to ``n`` real issues (PRs excluded), cached to disk. stdlib only."""
    owner, name = repo.split("/", 1)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"issues-{owner}-{name}.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if len(data) >= n:
            return data[:n]
    out: list[dict] = []
    page = 1
    while len(out) < n and page <= 5:
        url = (f"{_GH_API}/repos/{owner}/{name}/issues"
               f"?state=all&per_page=100&page={page}")
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                                   "User-Agent": "aether-context-eval"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            batch = json.loads(resp.read().decode("utf-8"))
        if not batch:
            break
        for it in batch:
            if "pull_request" in it:
                continue
            out.append({
                "number": it["number"], "title": it.get("title", ""),
                "body": (it.get("body") or "")[:2000],
                "labels": [lbl["name"] for lbl in it.get("labels", [])],
                "state": it.get("state", "open"),
            })
        page += 1
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out[:n]


# ---------------------------------------------------------------------------
# Synthetic tools over the corpus
# ---------------------------------------------------------------------------
class IssueTools:
    """The two tools the agent can call, answered from the cached issues."""

    TOOLS_SCHEMA = [
        {"type": "function", "function": {
            "name": "lookup_issue",
            "description": "Read an issue's body and labels by number.",
            "parameters": {"type": "object", "properties": {"number": {"type": "integer"}},
                           "required": ["number"]}}},
        {"type": "function", "function": {
            "name": "search_issues",
            "description": "Find issues whose title/labels match a query.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                           "required": ["query"]}}},
    ]

    def __init__(self, issues: list[dict]) -> None:
        self._by_num = {int(i["number"]): i for i in issues}
        self.calls = 0
        self.redundant = 0
        self._seen: set[int] = set()

    def lookup_issue(self, number: int) -> dict:
        self.calls += 1
        n = int(number)
        if n in self._seen:
            self.redundant += 1
        self._seen.add(n)
        it = self._by_num.get(n)
        if not it:
            return {"error": f"no issue {n}"}
        return {"number": n, "body": it["body"], "labels": it["labels"]}

    def search_issues(self, query: str) -> list[dict]:
        self.calls += 1
        q = (query or "").lower()
        return [{"number": i["number"], "title": i["title"]}
                for i in self._by_num.values()
                if q in i["title"].lower() or any(q in lbl.lower() for lbl in i["labels"])][:10]

    def dispatch(self, name: str, args: dict) -> Any:
        if name == "lookup_issue":
            return self.lookup_issue(args.get("number"))
        if name == "search_issues":
            return self.search_issues(args.get("query", ""))
        return {"error": f"unknown tool {name}"}


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------
def cost_usd(usage: dict, *, price_in: float, price_out: float) -> float:
    """USD for one call: (prompt_tokens*price_in + completion_tokens*price_out) per 1M."""
    pin = float(usage.get("prompt_tokens", 0)) / 1e6 * price_in
    pout = float(usage.get("completion_tokens", 0)) / 1e6 * price_out
    return pin + pout


# ---------------------------------------------------------------------------
# Config + deterministic dry-run chat
# ---------------------------------------------------------------------------
@dataclass
class Config:
    model: str = DEFAULT_MODEL
    repo: str = DEFAULT_REPO
    issues_n: int = 60
    turns: int = 40
    window: int = 2048
    arms: tuple[str, ...] = ("off", "on_chain")
    price_in: float = 0.3
    price_out: float = 1.2
    max_calls: int = 400
    judge: bool = False
    dry_run: bool = False
    out_dir: Path = field(default_factory=lambda: Path("."))
    cache_dir: Path = field(default_factory=lambda: Path("bench/.cache"))


class _MockChat:
    """Deterministic stand-in for adapter.chat — exercises the loop with no network.

    Turn parity drives behavior: odd turns emit a lookup_issue tool call; even turns emit a
    final answer naming the correct label of the focus issue (so coherence scores).
    """

    def __init__(self, tools: IssueTools) -> None:
        self._tools = tools
        self._t = 0

    def chat(self, messages, tools=None, *, max_tokens=None):
        self._t += 1
        usage = {"prompt_tokens": sum(estimate(m.get("content") or "") for m in messages),
                 "completion_tokens": 12}
        nums = sorted(self._tools._by_num)
        if self._t % 2 == 1:
            n = nums[(self._t // 2) % len(nums)]
            return {"content": None, "usage": usage, "tool_calls": [
                {"id": f"c{self._t}", "type": "function",
                 "function": {"name": "lookup_issue", "arguments": json.dumps({"number": n})}}]}
        n = nums[((self._t - 1) // 2) % len(nums)]
        labels = self._tools._by_num[n]["labels"]
        ans = f"issue {n} label: {labels[0] if labels else 'none'}"
        return {"content": ans, "usage": usage, "tool_calls": []}


# ---------------------------------------------------------------------------
# The eval
# ---------------------------------------------------------------------------
def _label_of(issue: dict) -> str:
    return issue["labels"][0] if issue.get("labels") else "none"


def _coherent(answer: str, issue: dict) -> bool:
    """Work-outcome ground truth: did the answer name the issue's real primary label?"""
    if not answer:
        return False
    return _label_of(issue).lower() in answer.lower()


def _truncate(messages: list[dict], window_tokens: int) -> list[dict]:
    """Keep system + a tail of messages whose total est. tokens fit ~window (chars/4)."""
    budget = window_tokens
    head = messages[:1]
    tail: list[dict] = []
    for m in reversed(messages[1:]):
        c = estimate(m.get("content") or "")
        if budget - c < 0:
            break
        budget -= c
        tail.insert(0, m)
    return head + tail


def _run_arm(arm: str, cfg: Config, corpus: list[dict], chat_obj) -> dict:
    tools = IssueTools(corpus)
    chat = chat_obj(tools) if cfg.dry_run else chat_obj
    sys_prompt = ("You triage GitHub issues. Use lookup_issue to read an issue, then state its "
                  "primary label as 'issue <n> label: <label>'. Be terse.")
    session: Optional[Session] = None
    encoder = StaticEncoder(dim=256)
    if arm in ("on", "on_chain"):
        session = Session("mock", pool_gb=5, pool_dir=cfg.out_dir / f"pool_{arm}",
                          context_window=cfg.window, mpo_chain=(arm == "on_chain"))
    transcript: list[dict] = [{"role": "system", "content": sys_prompt}]
    series: list[dict] = []
    calls = 0
    total_cost = 0.0
    correct = 0
    asked = 0

    for turn in range(1, cfg.turns + 1):
        if calls >= cfg.max_calls:
            break
        focus_issue = corpus[(turn - 1) % len(corpus)]
        user = f"Issue {focus_issue['number']}: state its primary label."
        if session is not None:
            qvec = encoder.encode(user)
            recalled = session._cold_retrieve(session._key(), qvec, 6)
            ctx = "\n".join(f"[mem] {s.text}" for s in recalled)
            messages = [transcript[0], {"role": "system", "content": f"Working memory:\n{ctx}"},
                        {"role": "user", "content": user}]
        else:  # OFF: full transcript truncated to the window
            messages = _truncate(transcript + [{"role": "user", "content": user}], cfg.window)

        t0 = time.monotonic()
        out = chat.chat(messages, tools=IssueTools.TOOLS_SCHEMA)
        latency = time.monotonic() - t0
        calls += 1
        total_cost += cost_usd(out.get("usage", {}), price_in=cfg.price_in, price_out=cfg.price_out)

        tcs = out.get("tool_calls") or []
        for tc in tcs:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            result = tools.dispatch(fn.get("name", ""), args)
            rtext = f"tool {fn.get('name')} -> {json.dumps(result)[:400]}"
            transcript.append({"role": "user", "content": user})
            transcript.append({"role": "assistant", "content": rtext})
            if session is not None:
                session.remember(rtext)

        answer = out.get("content")
        if answer and not tcs:
            asked += 1
            correct += int(_coherent(answer, focus_issue))
            transcript.append({"role": "user", "content": user})
            transcript.append({"role": "assistant", "content": answer})
            if session is not None:
                session.remember(answer)

        series.append({
            "turn": turn, "cum_cost": round(total_cost, 6), "latency": round(latency, 3),
            "tool_calls": tools.calls, "redundant": tools.redundant,
            "answered": asked, "correct": correct,
            "coherence": round(correct / asked, 3) if asked else 0.0,
            "prompt_tokens": int(out.get("usage", {}).get("prompt_tokens", 0)),
        })

    if session is not None:
        session.close()
    return {
        "cost_usd": round(total_cost, 6),
        "tool_calls": tools.calls,
        "redundant_tool_calls": tools.redundant,
        "answered": asked,
        "correct": correct,
        "coherence": round(correct / asked, 3) if asked else 0.0,
        "series": series,
    }


def _synthetic_corpus(n: int) -> list[dict]:
    """Offline stand-in issues for --dry-run (no network)."""
    labels = ("bug", "docs", "feature", "perf")
    return [
        {"number": i, "title": f"synthetic issue {i} about module {i % 7}",
         "body": f"synthetic body for issue {i}", "labels": [labels[i % len(labels)]],
         "state": "open"}
        for i in range(1, n + 1)
    ]


def run_eval(cfg: Config, corpus: Optional[list[dict]] = None) -> dict:
    if corpus is None:
        corpus = (_synthetic_corpus(cfg.issues_n) if cfg.dry_run
                  else fetch_issues(cfg.repo, cfg.issues_n, cfg.cache_dir))
    chat_obj = _MockChat if cfg.dry_run else _make_live_chat(cfg)
    results: dict[str, Any] = {"model": cfg.model, "repo": cfg.repo,
                               "issues": len(corpus), "arms": {}}
    for arm in cfg.arms:
        results["arms"][arm] = _run_arm(arm, cfg, corpus, chat_obj)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "api_eval_results.json").write_text(json.dumps(results, indent=2),
                                                       encoding="utf-8")
    _write_csv(cfg.out_dir / "api_eval_series.csv", results)
    return results


def _make_live_chat(cfg: Config):
    from aether_context.local_llm import OpenAICompatLLM
    return OpenAICompatLLM(cfg.model, context_window=cfg.window)


def _write_csv(path: Path, results: dict) -> None:
    rows = ["arm,turn,cum_cost,coherence,tool_calls,redundant,prompt_tokens,latency"]
    for arm, data in results["arms"].items():
        for s in data["series"]:
            rows.append(f"{arm},{s['turn']},{s['cum_cost']},{s['coherence']},"
                        f"{s['tool_calls']},{s['redundant']},{s['prompt_tokens']},{s['latency']}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _plot(results: dict, out_dir: Path) -> Optional[Path]:
    """Start->finish line graph (cum cost + coherence vs turn, per arm). matplotlib optional."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping --plot (CSV/JSON still written).")
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for arm, data in results["arms"].items():
        s = data["series"]
        xs = [r["turn"] for r in s]
        ax1.plot(xs, [r["cum_cost"] for r in s], label=arm)
        ax2.plot(xs, [r["coherence"] for r in s], label=arm)
    ax1.set_title("cumulative cost ($)")
    ax1.set_xlabel("turn")
    ax1.legend()
    ax2.set_title("coherence")
    ax2.set_xlabel("turn")
    ax2.set_ylim(0, 1)
    ax2.legend()
    fig.tight_layout()
    p = out_dir / "api_eval_plot.png"
    fig.savefig(p, dpi=120)
    return p


def _print_table(results: dict) -> None:
    print(f"\nmodel={results['model']} repo={results['repo']} issues={results['issues']}")
    print(f"{'arm':<10}{'cost$':>10}{'tools':>8}{'redund':>8}{'coher':>8}{'correct':>9}")
    for arm, d in results["arms"].items():
        print(f"{arm:<10}{d['cost_usd']:>10.4f}{d['tool_calls']:>8}"
              f"{d['redundant_tool_calls']:>8}{d['coherence']:>8.3f}"
              f"{str(d['correct']) + '/' + str(d['answered']):>9}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="api_eval",
                                description="OpenRouter engine eval over GitHub issues.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--issues", type=int, default=60)
    p.add_argument("--turns", type=int, default=40)
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--arms", default="off,on_chain", help="comma list: off,on,on_chain")
    p.add_argument("--price-in", type=float, default=0.3, help="$/1M prompt tokens")
    p.add_argument("--price-out", type=float, default=1.2, help="$/1M completion tokens")
    p.add_argument("--max-calls", type=int, default=400)
    p.add_argument("--out", default=".")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    if not a.dry_run and not (os.environ.get("OPENROUTER_API_KEY")
                              or os.environ.get("OPENAI_API_KEY")):
        print("No OPENROUTER_API_KEY set — skipping live eval. Use --dry-run to test the harness.")
        return 0

    cfg = Config(model=a.model, repo=a.repo, issues_n=a.issues, turns=a.turns, window=a.window,
                 arms=tuple(x.strip() for x in a.arms.split(",") if x.strip()),
                 price_in=a.price_in, price_out=a.price_out, max_calls=a.max_calls,
                 dry_run=a.dry_run, out_dir=Path(a.out))
    results = run_eval(cfg)
    _print_table(results)
    if a.plot:
        path = _plot(results, Path(a.out))
        if path:
            print(f"plot -> {path}")
    print(f"results -> {Path(a.out) / 'api_eval_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
