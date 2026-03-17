"""
utils/file_ops.py

Foundational file-system and patch utilities.
All functions are pure Python 3.14-compatible.
No FastMCP dependencies.
"""
from __future__ import annotations

import json
import re
import ast
import asyncio
import fnmatch
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Sequence
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CommandResult:
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    command: str
    duration_sec: float


async def _run_subprocess(command: Sequence[str], cwd: Optional[str] = None) -> CommandResult:
    start_t = time.monotonic()
    cmd = [str(x) for x in command]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
        duration = time.monotonic() - start_t
        return CommandResult(
            success=(proc.returncode == 0),
            exit_code=proc.returncode or 0,
            stdout=stdout_bytes.decode(errors="replace"),
            stderr=stderr_bytes.decode(errors="replace"),
            command=" ".join(cmd),
            duration_sec=duration,
        )
    except Exception as exc:
        return CommandResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr=str(exc),
            command=" ".join(cmd),
            duration_sec=time.monotonic() - start_t,
        )


# ---------------------------------------------------------------------------
# File navigation
# ---------------------------------------------------------------------------

def list_files(root: str = ".", include: str = "", exclude: str = "") -> Dict[str, Any]:
    try:
        root_path = Path(root)
        files: List[str] = []
        for p in root_path.rglob("*"):
            if p.is_file():
                rel = str(p.relative_to(root_path))
                if include and include not in rel:
                    continue
                if exclude and exclude in rel:
                    continue
                files.append(rel)
        return {"success": True, "files": files}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def grep_code(pattern: str, path: str) -> Dict[str, Any]:
    matches: List[Dict[str, Any]] = []
    try:
        regex = re.compile(pattern)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, start=1):
                if regex.search(line):
                    matches.append({"line": lineno, "text": line.rstrip("\n")})
        return {"success": True, "matches": matches}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def extract_function(
    path: str,
    symbol: Optional[str] = None,
    symbol_name: Optional[str] = None,
    include_docstring: bool = False,
) -> Dict[str, Any]:
    target_symbol = symbol or symbol_name or ""
    try:
        source = Path(path).read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
        result_source: Optional[str] = None
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and getattr(node, "name", None) == target_symbol
            ):
                start = node.lineno - 1
                end = (
                    node.end_lineno - 1
                    if hasattr(node, "end_lineno") and node.end_lineno is not None
                    else node.lineno - 1
                )
                lines = source.splitlines()
                if not include_docstring and node.body:
                    first_stmt = node.body[0]
                    # ast.Str was removed in Python 3.12; use ast.Constant with str check
                    if (
                        isinstance(first_stmt, ast.Expr)
                        and isinstance(first_stmt.value, ast.Constant)
                        and isinstance(first_stmt.value.value, str)
                    ):
                        ds_end = (
                            first_stmt.end_lineno - 1
                            if hasattr(first_stmt, "end_lineno") and first_stmt.end_lineno is not None
                            else first_stmt.lineno - 1
                        )
                        start = ds_end + 1
                result_source = "\n".join(lines[start : end + 1])
                break
        if result_source is None:
            raise Exception(f"Symbol '{target_symbol}' not found in {path}")
        return {"success": True, "source": result_source}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def extract_patch_block(text: str) -> Dict[str, Any]:
    """Extract the first diff or code block from model output. Prefers ```diff blocks."""
    # Prefer explicit diff blocks
    diff_match = re.search(r"```diff\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if diff_match:
        return {"success": True, "patch_text": diff_match.group(1).strip()}
    # Fallback: any fenced code block
    code_match = re.search(r"```(?:[^\n]*)\n(.*?)```", text, flags=re.DOTALL)
    if code_match:
        return {"success": True, "patch_text": code_match.group(1).strip()}
    return {"success": False, "patch_text": ""}


# ---------------------------------------------------------------------------
# Patch safety helpers
# ---------------------------------------------------------------------------

def parse_patch_manifest(patch_text: str) -> List[str]:
    """
    Extract the list of relative file paths that a unified diff will modify.
    Parses '+++ b/...' lines (git-style) and bare '+++ path' lines.
    /dev/null targets (new-file markers on the --- side) are excluded.
    """
    paths: List[str] = []
    seen: set = set()
    for line in patch_text.splitlines():
        if line.startswith("+++ "):
            raw = line[4:].split("\t")[0].strip()
            if raw.startswith("b/"):
                raw = raw[2:]
            if raw and raw != "/dev/null" and raw not in seen:
                paths.append(raw)
                seen.add(raw)
    return paths


def snapshot_files(paths: List[str], repo_root: Path) -> Dict[str, Optional[bytes]]:
    """
    Capture current file bytes for rollback.
    Files that do not exist are stored as None so rollback knows to delete them.
    """
    snapshot: Dict[str, Optional[bytes]] = {}
    for rel in paths:
        full = repo_root / rel
        snapshot[rel] = full.read_bytes() if full.exists() else None
    return snapshot


def rollback_files(snapshot: Dict[str, Optional[bytes]], repo_root: Path) -> None:
    """
    Restore files to their pre-apply state from a snapshot.
    None entries (files that did not exist) are deleted if they were created.
    """
    for rel, original in snapshot.items():
        full = repo_root / rel
        if original is None:
            if full.exists():
                full.unlink()
        else:
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_bytes(original)
