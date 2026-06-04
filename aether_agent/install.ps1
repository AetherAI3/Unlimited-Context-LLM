# Aether Coding Agent — 1-command install (Windows, PowerShell).
#
#   irm https://raw.githubusercontent.com/DBarr3/Unlimited-Context/main/aether_agent/install.ps1 | iex
#
# Installs the engine + agent (pip), checks Ollama, and pulls the default model.
$ErrorActionPreference = "Stop"

$Model = if ($env:AETHER_MODEL) { $env:AETHER_MODEL } else { "qwen3-coder:30b" }

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
  Write-Host "Aether Code needs Python 3.10+ — install from https://python.org"
  exit 1
}

Write-Host "Installing aether-context (Unlimited Context engine + Aether agent)..."
python -m pip install --upgrade aether-context

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
  Write-Host "Install Ollama from https://ollama.com/download, then re-run this."
  exit 1
}

Write-Host "Pulling $Model (large, one-time)..."
ollama pull $Model

Write-Host ""
Write-Host "Aether Code ready. Try it in any repo:"
Write-Host "  aether code ""fix the failing tests"" --pool 5"
Write-Host "  (lighter machine? set AETHER_MODEL=qwen3-coder-next, or use --model)"
