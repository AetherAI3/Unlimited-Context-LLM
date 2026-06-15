"""
Coding tools — OpenAI tool schema + a path-guarded executor.
read/write file · shell · run tests · repo search · git commit.
All paths are confined to the workspace (cwd); output is capped.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

MAX_OUTPUT = 8000


def tool_schema() -> list[dict]:
    def fn(name: str, desc: str, props: dict, required: list[str]) -> dict:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        }

    s = {"type": "string"}
    i = {"type": "integer"}
    return [
        fn("read_file", "Read a file's contents (relative to the workspace).", {"path": s}, ["path"]),
        fn("write_file", "Create or overwrite a file with the given content.", {"path": s, "content": s}, ["path", "content"]),
        fn("run_shell", "Run a shell command in the workspace and return its output.", {"command": s}, ["command"]),
        fn("run_tests", "Run the test suite (default: pytest -q).", {"command": s}, []),
        fn("repo_search", "Search the repository for a string.", {"query": s}, ["query"]),
        fn("git_commit", "Stage all changes and commit with a message.", {"message": s}, ["message"]),
        fn(
            "web_search",
            "Search the public web (DuckDuckGo) and return the top results as titles, urls, and snippets.",
            {"query": s, "limit": i},
            ["query"],
        ),
        fn(
            "web_fetch",
            "Fetch a public web page over http(s) and return its readable text (tags/scripts stripped). "
            "Refuses non-public/internal hosts.",
            {"url": s},
            ["url"],
        ),
    ]


class Tools:
    def __init__(self, cwd: str, test_cmd: str = "pytest -q"):
        # Canonicalize the root (resolve symlinks in the workspace path itself).
        self.cwd = os.path.realpath(cwd)
        self.test_cmd = test_cmd

    def _safe(self, path: str) -> str:
        """Resolve a workspace-relative path, refusing any escape. Canonicalizes
        BEFORE the allowlist check so `..`, absolute paths, and symlinks pointing
        outside the worktree are all rejected. The nearest existing ancestor is
        realpath'd (the non-existent tail of a write target can't be a symlink)."""
        ap = os.path.abspath(os.path.join(self.cwd, path))
        ancestor = ap
        while not os.path.exists(ancestor) and os.path.dirname(ancestor) != ancestor:
            ancestor = os.path.dirname(ancestor)
        real_ancestor = os.path.realpath(ancestor)
        if real_ancestor != self.cwd and not real_ancestor.startswith(self.cwd + os.sep):
            raise ValueError(f"refusing path outside workspace: {path}")
        return ap

    def _run(self, cmd: str, timeout: int = 900) -> str:
        try:
            p = subprocess.run(cmd, shell=True, cwd=self.cwd, capture_output=True, text=True, timeout=timeout)
            out = (p.stdout or "") + (p.stderr or "")
            return f"[exit {p.returncode}]\n{out[:MAX_OUTPUT]}"
        except subprocess.TimeoutExpired:
            return f"[timeout after {timeout}s]"

    def read_file(self, path: str) -> str:
        ap = self._safe(path)
        if not os.path.isfile(ap):
            return f"[no such file: {path}]"
        return Path(ap).read_text(encoding="utf-8", errors="replace")[:MAX_OUTPUT]

    def write_file(self, path: str, content: str) -> str:
        ap = self._safe(path)
        os.makedirs(os.path.dirname(ap) or self.cwd, exist_ok=True)
        Path(ap).write_text(content, encoding="utf-8")
        return f"[wrote {path} · {len(content)} bytes]"

    def run_shell(self, command: str) -> str:
        return self._run(command)

    def run_tests(self, command: str | None = None) -> str:
        return self._run(command or self.test_cmd)

    def repo_search(self, query: str) -> str:
        return self._run(f"grep -rIn -- {json.dumps(query)} . | head -40")

    def git_commit(self, message: str) -> str:
        self._run("git add -A")
        return self._run(f'git commit -q -m {json.dumps(message)} || echo "[nothing to commit]"')

    def execute(self, name: str, args: dict) -> str:
        try:
            if name == "read_file":
                return self.read_file(args["path"])
            if name == "write_file":
                return self.write_file(args["path"], args.get("content", ""))
            if name == "run_shell":
                return self.run_shell(args["command"])
            if name == "run_tests":
                return self.run_tests(args.get("command"))
            if name == "repo_search":
                return self.repo_search(args["query"])
            if name == "git_commit":
                return self.git_commit(args["message"])
            # Network tools — NOT path-jailed (no workspace to confine to); the
            # SSRF guard lives in web.py. Lazy import keeps the file/shell tools
            # free of urllib for the pure-codec test paths.
            if name == "web_search":
                from aether_agent import web

                return web.web_search(args["query"], int(args.get("limit", 5) or 5))
            if name == "web_fetch":
                from aether_agent import web

                return web.web_fetch(args["url"])
            return f"[unknown tool: {name}]"
        except KeyError as e:
            return f"[tool {name}: missing argument {e}]"
        except Exception as e:  # noqa: BLE001
            return f"[tool {name} error: {e}]"
