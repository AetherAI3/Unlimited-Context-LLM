"""
Model adapter — Ollama's OpenAI-compatible chat endpoint. Model is config; the
weights are never touched or bundled. stdlib HTTP only (no extra dependency).

Swap models freely: `aether code --model qwen3-coder:30b` (default).
Use Gemma 4 / Qwen3-Coder-Next for light machines (Gemma 3 carries custom terms;
do not use it).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional

from aether_agent.profiles import for_model

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen3-coder:30b"  # depth build; 256K ctx, Apache-2.0


class OllamaChat:
    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST, timeout: float = 600.0):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        # Per-model sampling (Gemma wants Google's; Qwen wants low-temp determinism).
        self.profile = for_model(model)

    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: Optional[float] = None,
    ) -> dict[str, Any]:
        """One turn against /v1/chat/completions. Returns the assistant message
        dict ({role, content, tool_calls?}). Raises with a clear hint if Ollama
        is down or the model isn't pulled.

        Sampling comes from the model's profile; `temperature` overrides it when
        given (back-compat). top_p/top_k ride along — Ollama's OpenAI-compat layer
        honors them for local models."""
        samp = self.profile.sampling
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": samp["temperature"] if temperature is None else temperature,
            "top_p": samp["top_p"],
            "top_k": samp["top_k"],
            "stream": False,
        }
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(
            f"{self.host}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(
                f"Ollama returned {e.code}: {detail}. "
                f"Is the model '{self.model}' pulled? Try: ollama pull {self.model}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.host} ({e.reason}). "
                "Start it with `ollama serve` (the installer does this)."
            ) from e
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError(f"Empty response from Ollama: {payload}")
        return choices[0].get("message", {})
