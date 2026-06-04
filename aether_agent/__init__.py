"""
Aether Coding Agent (neo-lite) — an autonomous local coding agent on top of the
open Unlimited Context engine.

The model is UNMODIFIED and swappable (a config tag). The "Aether-ness" is the
harness: coding persona + tool loop + Unlimited Context as working memory +
autonomy/checkpoint kernel + escalation hatch. Default model: Qwen3-Coder 30B
(Apache-2.0), pulled via Ollama — never bundled.
"""

__all__ = ["run_agent", "AgentResult"]


def __getattr__(name: str):
    # Lazy: importing aether_agent (or its dependency-light submodules like
    # statusbar/kernel/tools) must NOT pull the numpy-backed engine until the
    # agent loop is actually used.
    if name in ("run_agent", "AgentResult"):
        from aether_agent.agent import run_agent, AgentResult

        return {"run_agent": run_agent, "AgentResult": AgentResult}[name]
    raise AttributeError(f"module 'aether_agent' has no attribute {name!r}")
