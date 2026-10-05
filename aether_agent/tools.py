"""
Coding tools — OpenAI tool schema + a path-guarded executor.
read/write file · shell · run tests · repo search · git commit.
All paths are confined to the workspace (cwd); output is capped.
"""

from __future__ import annotations

import json
import os
import subprocess
import base64
import codecs
import hashlib
import re
import tempfile
from pathlib import Path

MAX_OUTPUT = 8000
MAX_FILE_BYTES = 16 * 1024 * 1024


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
        fn("read_file", "Read a bounded byte or line range and return a SHA-256 digest.", {"path": s, "offset": i, "max_bytes": i, "start_line": i, "max_lines": i}, ["path"]),
        fn("list_directory", "List one bounded directory page with path/type metadata.", {"path": s, "cursor": s, "limit": i}, ["path"]),
        fn("patch_file", "Replace one unique exact text range using an expected SHA-256 digest.", {"path": s, "expected_sha256": s, "old_text": s, "new_text": s, "start_line": i}, ["path", "expected_sha256", "old_text", "new_text"]),
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

    def read_file(self, path: str, offset: int | None = None, max_bytes: int | None = None,
                  start_line: int | None = None, max_lines: int | None = None) -> str:
        ap = self._safe(path)
        if os.path.islink(ap) or not os.path.isfile(ap):
            return f"[no such file: {path}]"
        with open(ap, "rb") as handle:
            before = os.fstat(handle.fileno())
            size = before.st_size
            digest = None
            if size <= MAX_FILE_BYTES:
                hasher = hashlib.sha256()
                utf8 = codecs.getincrementaldecoder("utf-8")("strict")
                while block := handle.read(64 * 1024):
                    if b"\0" in block:
                        return f"[binary file: {path}]"
                    try:
                        utf8.decode(block)
                    except UnicodeDecodeError:
                        return f"[invalid UTF-8 file: {path}]"
                    hasher.update(block)
                try:
                    utf8.decode(b"", final=True)
                except UnicodeDecodeError:
                    return f"[invalid UTF-8 file: {path}]"
                digest = hasher.hexdigest()
            if start_line is not None or max_lines is not None:
                if offset is not None or max_bytes is not None:
                    raise ValueError("line and byte ranges cannot be combined")
                start = start_line or 1
                count = max_lines or 200
                if start < 1 or count < 1 or count > 200:
                    raise ValueError("invalid line range")
                handle.seek(0)
                selected = bytearray()
                current = bytearray()
                line = 1
                selected_lines = 0
                next_line = None
                too_long = False

                def finish_line(has_newline: bool) -> bool:
                    nonlocal line, current, selected_lines, next_line, too_long
                    if line >= start:
                        if len(selected) + len(current) + (1 if selected_lines else 0) > 6000:
                            next_line = line
                            too_long = selected_lines == 0
                            return True
                        if selected_lines:
                            selected.extend(b"\n")
                        selected.extend(current)
                        selected_lines += 1
                        if selected_lines >= count and has_newline:
                            next_line = line + 1
                            return True
                    line += 1
                    current = bytearray()
                    return False

                stopped = False
                while not stopped and (block := handle.read(64 * 1024)):
                    for byte in block:
                        if byte == 0:
                            return f"[binary file: {path}]"
                        if byte == 10:
                            if finish_line(True):
                                stopped = True
                                break
                        elif line >= start:
                            current.append(byte)
                            if len(selected) + len(current) + (1 if selected_lines else 0) > 6000:
                                next_line = line
                                too_long = selected_lines == 0
                                stopped = True
                                break
                if not stopped:
                    finish_line(False)
                if selected_lines == 0 and not too_long:
                    raise ValueError("start_line beyond EOF")
                if too_long:
                    result = {"path": path, "sha256": digest, "start_line": start,
                              "next_start_line": None, "size": size,
                              "validation_scope": "returned_range" if digest is None else "whole_file", "content": "",
                              "note": "line exceeds 6000 bytes; use offset/max_bytes"}
                else:
                    result = {"path": path, "sha256": digest, "start_line": start,
                              "next_start_line": next_line, "size": size,
                              "validation_scope": "returned_range" if digest is None else "whole_file",
                              "content": selected.decode("utf-8")}
            else:
                begin = offset or 0
                count = max_bytes or 4096
                if begin < 0 or begin > size or count < 4 or count > 4096:
                    raise ValueError("invalid byte range")
                handle.seek(begin)
                raw = handle.read(min(size - begin, count))
                if b"\0" in raw:
                    return f"[binary file: {path}]"
                end = len(raw)
                while end > 0:
                    try:
                        chunk = raw[:end].decode("utf-8")
                        break
                    except UnicodeDecodeError:
                        end -= 1
                else:
                    if begin < size:
                        raise ValueError("offset splits a UTF-8 character or invalid UTF-8")
                    chunk = ""
                result = {"path": path, "sha256": digest, "offset": begin,
                          "next_offset": begin + end if begin + end < size else None,
                          "size": size,
                          "validation_scope": "returned_range" if digest is None else "whole_file",
                          "content": chunk}
            after = os.fstat(handle.fileno())
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise ValueError("read conflict: file changed during read")
            return json.dumps(result, ensure_ascii=False)

    def list_directory(self, path: str, cursor: str | None = None, limit: int = 50) -> str:
        ap = self._safe(path)
        if os.path.islink(ap) or not os.path.isdir(ap):
            return f"[no such directory: {path}]"
        if not 1 <= limit <= 100:
            raise ValueError("limit must be from 1 to 100")
        entries = []
        with os.scandir(ap) as scan:
            for entry in scan:
                entries.append(entry)
                if len(entries) > 10000:
                    raise ValueError("directory exceeds 10000-entry listing limit")
        entries.sort(key=lambda entry: entry.name)
        facts = [(entry.name, entry.stat(follow_symlinks=False)) for entry in entries]
        version = hashlib.sha256(repr([(name, st.st_mode, st.st_size, st.st_mtime_ns) for name, st in facts]).encode()).hexdigest()
        rel = os.path.relpath(ap, self.cwd)
        after = ""
        if cursor is not None:
            try:
                decoded = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
            except (ValueError, UnicodeError) as exc:
                raise ValueError("invalid directory cursor") from exc
            if decoded.get("path") != rel or decoded.get("version") != version or not isinstance(decoded.get("after"), str):
                raise ValueError("directory listing conflict: path or contents changed")
            after = decoded["after"]
        remaining = [entry for entry in entries if entry.name > after]
        page = []
        for entry in remaining[:limit]:
            kind = "symlink" if entry.is_symlink() else "directory" if entry.is_dir(follow_symlinks=False) else "file" if entry.is_file(follow_symlinks=False) else "other"
            item = {"path": "./" + os.path.relpath(entry.path, self.cwd).replace(os.sep, "/"), "type": kind}
            if kind == "file":
                item["size"] = entry.stat(follow_symlinks=False).st_size
            if len(json.dumps({"entries": [*page, item]}, ensure_ascii=False).encode()) > 6500:
                break
            page.append(item)
        if not page and remaining:
            raise ValueError("directory entry exceeds output budget")
        next_cursor = None
        if len(page) < len(remaining):
            payload = json.dumps({"path": rel, "version": version, "after": remaining[len(page) - 1].name})
            next_cursor = base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")
        return json.dumps({"path": path, "entries": page, "next_cursor": next_cursor}, ensure_ascii=False)

    def patch_file(self, path: str, expected_sha256: str, old_text: str, new_text: str,
                   start_line: int | None = None) -> str:
        ap = self._safe(path)
        if os.path.islink(ap) or not os.path.isfile(ap):
            raise ValueError("patch target must be a regular file")
        original_stat = os.stat(ap, follow_symlinks=False)
        if original_stat.st_size > MAX_FILE_BYTES:
            raise ValueError("patch target too large")
        original_bytes = Path(ap).read_bytes()
        digest = hashlib.sha256(original_bytes).hexdigest()
        if not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
            raise ValueError("expected_sha256 must be lowercase SHA-256")
        if digest != expected_sha256:
            raise ValueError(f"conflict: file changed since read (current sha256 {digest})")
        if b"\0" in original_bytes:
            raise ValueError("binary patch target")
        original = original_bytes.decode("utf-8")
        if old_text == new_text:
            raise ValueError("patch has no change")
        if not old_text:
            if start_line is None or start_line < 1:
                raise ValueError("start_line required for insertion")
            starts = [0] + [i + 1 for i, char in enumerate(original) if char == "\n"]
            if start_line > len(starts) + (0 if original.endswith("\n") else 1):
                raise ValueError("start_line beyond EOF")
            position = starts[start_line - 1] if start_line <= len(starts) else len(original)
        else:
            position = original.find(old_text)
            if position < 0:
                raise ValueError("hunk does not match")
            if original.find(old_text, position + 1) >= 0:
                raise ValueError("ambiguous hunk: old_text occurs more than once")
            if start_line is not None and original[:position].count("\n") + 1 != start_line:
                raise ValueError("hunk does not match start_line")
        replacement = (original[:position] + new_text + original[position + len(old_text):]).encode("utf-8")
        staged = None
        try:
            with tempfile.NamedTemporaryFile(dir=os.path.dirname(ap), prefix=".aether-patch-", suffix=".tmp", delete=False) as handle:
                staged = handle.name
                handle.write(replacement)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(staged, original_stat.st_mode)
            current_stat = os.stat(ap, follow_symlinks=False)
            if (os.path.islink(ap) or current_stat.st_mode != original_stat.st_mode
                    or current_stat.st_ino != original_stat.st_ino
                    or current_stat.st_mtime_ns != original_stat.st_mtime_ns
                    or Path(ap).read_bytes() != original_bytes):
                raise ValueError("conflict: file changed while patch was staged")
            os.replace(staged, ap)
            return f"[patched {path} · sha256 {hashlib.sha256(replacement).hexdigest()}]"
        finally:
            if staged is not None and os.path.exists(staged):
                os.unlink(staged)

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
                return self.read_file(args["path"], args.get("offset"), args.get("max_bytes"), args.get("start_line"), args.get("max_lines"))
            if name == "list_directory":
                return self.list_directory(args["path"], args.get("cursor"), args.get("limit", 50))
            if name == "patch_file":
                return self.patch_file(args["path"], args["expected_sha256"], args["old_text"], args["new_text"], args.get("start_line"))
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
