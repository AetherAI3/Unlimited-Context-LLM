"""Repo file tools for the SWE-bench codepro eval — read/grep/list/edit over a single
git checkout, plus the unified-diff capture used as the SWE-bench model_patch. Pure
(no network); the agent loop in swe_eval.py hosts these over a per-instance checkout."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


class RepoTools:
    """Tools answered from a checked-out repo dir. Edits write into the working tree;
    `current_patch()` returns `git diff` against the base commit = the model_patch."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.calls = 0
        self.redundant = 0
        self._read: set[str] = set()

    def _safe(self, rel: str) -> Path | None:
        """Resolve rel under root; None if it escapes (path-traversal guard)."""
        p = (self.root / rel).resolve()
        try:
            p.relative_to(self.root.resolve())
        except ValueError:
            return None
        return p

    def read_file(self, path: str) -> dict:
        self.calls += 1
        if path in self._read:
            self.redundant += 1
        self._read.add(path)
        p = self._safe(path or "")
        if p is None or not p.is_file():
            return {"error": f"no file {path}"}
        try:
            return {"path": path,
                    "content": p.read_text(encoding="utf-8", errors="replace")[:20000]}
        except OSError as e:
            return {"error": str(e)}

    def grep(self, pattern: str, path_glob: str = "") -> dict:
        self.calls += 1
        cmd = ["git", "grep", "-n", "-I", "-e", pattern or ""]
        if path_glob:
            cmd += ["--", path_glob]
        proc = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True,
                              errors="replace")  # repos contain non-UTF8 bytes
        matches = []
        for ln in proc.stdout.splitlines()[:100]:
            parts = ln.split(":", 2)
            if len(parts) == 3:
                matches.append({"path": parts[0], "lineno": parts[1], "line": parts[2]})
        return {"matches": matches}

    def list_dir(self, path: str = ".") -> dict:
        self.calls += 1
        p = self._safe(path or ".")
        if p is None or not p.is_dir():
            return {"error": f"no dir {path}"}
        entries = sorted(
            (c.name + "/" if c.is_dir() else c.name) for c in p.iterdir()
            if c.name != ".git")
        return {"entries": entries}

    def edit_file(self, path: str, old: str, new: str) -> dict:
        self.calls += 1
        p = self._safe(path or "")
        if p is None or not p.is_file():
            return {"error": f"no file {path}"}
        src = p.read_text(encoding="utf-8", errors="replace")
        if old not in src:
            return {"error": f"old string not found in {path}"}
        p.write_text(src.replace(old, new, 1), encoding="utf-8")
        return {"ok": True, "path": path}

    def current_patch(self) -> str:
        """Unified diff of the working tree vs HEAD (the base commit) = model_patch."""
        proc = subprocess.run(["git", "diff"], cwd=self.root, capture_output=True,
                              text=True, errors="replace")  # binary diffs -> non-UTF8 bytes
        return proc.stdout

    TOOLS_SCHEMA = [
        {"type": "function", "function": {
            "name": "read_file",
            "description": "Read a repo file's contents by path.",
            "parameters": {"type": "object",
                           "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
        {"type": "function", "function": {
            "name": "grep",
            "description": "Search the repo for a regex/string (git grep). Optional path glob.",
            "parameters": {"type": "object",
                           "properties": {"pattern": {"type": "string"},
                                          "path_glob": {"type": "string"}},
                           "required": ["pattern"]}}},
        {"type": "function", "function": {
            "name": "list_dir",
            "description": "List entries in a repo directory.",
            "parameters": {"type": "object",
                           "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
        {"type": "function", "function": {
            "name": "edit_file",
            "description": "Replace the first occurrence of `old` with `new` in a file.",
            "parameters": {"type": "object",
                           "properties": {"path": {"type": "string"},
                                          "old": {"type": "string"}, "new": {"type": "string"}},
                           "required": ["path", "old", "new"]}}},
    ]

    def dispatch(self, name: str, args: dict) -> Any:
        if name == "read_file":
            return self.read_file(args.get("path", ""))
        if name == "grep":
            return self.grep(args.get("pattern", ""), args.get("path_glob", ""))
        if name == "list_dir":
            return self.list_dir(args.get("path", "."))
        if name == "edit_file":
            return self.edit_file(args.get("path", ""), args.get("old", ""), args.get("new", ""))
        return {"error": f"unknown tool {name}"}


def prepare_checkout(repo_url: str, base_commit: str, workdir: Path) -> Path:
    """Make `workdir` a clean checkout of `repo_url` at `base_commit`.

    `workdir` is keyed PER REPO (not per instance): the first instance of a repo clones it
    once (network); every later instance of the same repo just hard-resets to its base_commit
    + cleans — SWE-bench-lite repeats ~12 repos across 300 instances, so this turns ~600 network
    clones into ~12. Used only so the agent's file tools can READ the repo during generation;
    Phase B does its own isolated checkout."""
    workdir = Path(workdir)
    if (workdir / ".git").is_dir():
        # cached clone of this repo — fetch the commit if missing, then hard-reset clean
        have = subprocess.run(["git", "cat-file", "-e", f"{base_commit}^{{commit}}"],
                              cwd=workdir, capture_output=True)
        if have.returncode != 0:
            subprocess.run(["git", "fetch", "-q", "origin", base_commit],
                           cwd=workdir, capture_output=True)  # best-effort; full clone usually has it
        subprocess.run(["git", "checkout", "-q", "-f", base_commit], cwd=workdir,
                       check=True, capture_output=True)
        subprocess.run(["git", "clean", "-qdfx"], cwd=workdir, check=True, capture_output=True)
        return workdir
    workdir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "-q", repo_url, str(workdir)], check=True,
                   capture_output=True)
    subprocess.run(["git", "checkout", "-q", base_commit], cwd=workdir, check=True,
                   capture_output=True)
    return workdir
