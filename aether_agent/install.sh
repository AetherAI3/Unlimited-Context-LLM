#!/bin/sh
# Aether Coding Agent — 1-command install (macOS / Linux / WSL).
#
#   curl -fsSL https://raw.githubusercontent.com/DBarr3/Unlimited-Context/main/aether_agent/install.sh | sh
#
# Installs the engine + agent (pip), Ollama, and pulls the default coding model.
set -e

MODEL="${AETHER_MODEL:-qwen3-coder:30b}"

command -v python3 >/dev/null 2>&1 || { echo "Aether Code needs Python 3.10+ — install from https://python.org"; exit 1; }

echo "Installing aether-context (Unlimited Context engine + Aether agent)…"
python3 -m pip install --upgrade aether-context

if ! command -v ollama >/dev/null 2>&1; then
  echo "Installing Ollama (runs the model locally)…"
  curl -fsSL https://ollama.com/install.sh | sh
fi

# Make sure the daemon is up, then pull the model.
( ollama serve >/dev/null 2>&1 & ) || true
sleep 2
echo "Pulling $MODEL (large, one-time)…"
ollama pull "$MODEL"

echo ""
echo "✓ Aether Code ready. Try it in any repo:"
echo "  aether code \"fix the failing tests\" --pool 5"
echo ""
echo "  (lighter machine? AETHER_MODEL=qwen3-coder-next before install, or --model on the fly)"
