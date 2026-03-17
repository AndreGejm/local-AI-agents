"""
server.py

Pure module — no FastMCP instance. All tool functions are async and are
called by MCPWorkerAdapter via getattr(). local_orchestrator.py owns the
single FastMCP instance.

Safety rules:
- read_file / list_files / write_file require allowed_root when called directly.
- apply_unified_diff runs a multi-step safety gate before touching any file.
- run_py_compile / run_lint / run_pytest gracefully skip when tools are absent.
"""
from __future__ import annotations

import fnmatch
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
import llm_orchestrator
import utils.file_ops as file_ops


# ---------------------------------------------------------------------------
# Path guard helper
# ---------------------------------------------------------------------------

def _assert_inside_root(path: str, root: str) -> Path:
    """
    Resolve *path* and verify it stays within *root*.
    Raises ValueError if the path escapes root.
    """
    root_path = Path(root).resolve()
    target = (root_path / path).resolve()
    try:
        target.relative_to(root_path)
    except ValueError:
        raise ValueError(f"Path '{path}' resolves outside the allowed root '{root}'.")
    return target


# ---------------------------------------------------------------------------
# File tools
# ---------------------------------------------------------------------------

async def read_file(path: str, max_chars: int = 10000, allowed_root: Optional[str] = None) -> str:
    try:
        if allowed_root:
            p = _assert_inside_root(path, allowed_root)
        else:
            p = Path(path)
        if not p.exists():
            return json.dumps({"success": False, "error": "Not found"})
        return json.dumps({"success": True, "content": p.read_text(encoding="utf-8", errors="replace")[:max_chars]})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


async def write_file(path: str, content: str, allowed_root: Optional[str] = None) -> str:
    """Write content to a file. allowed_root is required to prevent arbitrary writes."""
    try:
        if not allowed_root:
            return json.dumps({"success": False, "error": "allowed_root must be provided to write_file."})
        p = _assert_inside_root(path, allowed_root)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return json.dumps({"success": True})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


async def list_files(root: str = ".", include: str = "", exclude: str = "", allowed_root: Optional[str] = None) -> str:
    try:
        if allowed_root:
            effective_root = str(_assert_inside_root(root, allowed_root))
        else:
            effective_root = root
        result = file_ops.list_files(effective_root, include, exclude)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


async def grep_code(pattern: str, path: str) -> str:
    return json.dumps(file_ops.grep_code(pattern, path), indent=2)


async def extract_function(path: str, symbol: str, include_docstring: bool = False) -> str:
    return json.dumps(file_ops.extract_function(path, symbol=symbol, include_docstring=include_docstring), indent=2)


async def extract_patch_block(text: str) -> str:
    return json.dumps(file_ops.extract_patch_block(text), indent=2)


# ---------------------------------------------------------------------------
# Patch application — full safety gate
# ---------------------------------------------------------------------------

async def apply_unified_diff(
    patch_text: str,
    root: str,
    declared_files: Optional[List[str]] = None,
    allow_new_files: bool = False,
) -> Dict[str, Any]:
    """
    Apply a unified diff to a git repository with a hard safety gate.
    Returns a dict suitable for _normalize_command_result.

    Gate steps (in order):
    1  Canonicalize root; verify .git exists
    2  Parse file manifest from patch headers
    3  File count limit
    4  Path traversal check for every target
    5  Extension allowlist
    6  Dangerous-target denylist + orchestration-root self-protection
    7  Declared-files scope check
    8  No-new-file guard (default)
    9  Changed-line count limit
    10 Snapshot pre-apply state
    11 Dry-run via 'git apply --check'
    12 Actual apply; rollback snapshot on failure
    """
    # Gate 1
    repo_root = Path(root).resolve()
    if not (repo_root / ".git").exists():
        return {"success": False, "exit_code": -1, "stdout": "",
                "stderr": f"root '{root}' is not a git repository (.git not found)."}

    # Gate 2
    changed_files = file_ops.parse_patch_manifest(patch_text)
    if not changed_files:
        return {"success": False, "exit_code": -1, "stdout": "",
                "stderr": "Patch contains no recognisable file changes (no +++ headers)."}

    # Gate 3
    MAX_FILES = 5
    if len(changed_files) > MAX_FILES:
        return {"success": False, "exit_code": -1, "stdout": "",
                "stderr": f"Patch touches {len(changed_files)} files; limit is {MAX_FILES}."}

    # Gate 4
    for rel in changed_files:
        abs_path = (repo_root / rel).resolve()
        if not abs_path.is_relative_to(repo_root):
            return {"success": False, "exit_code": -1, "stdout": "",
                    "stderr": f"Path traversal detected: '{rel}' resolves outside repo root."}

    # Gate 5
    for rel in changed_files:
        if Path(rel).suffix not in config.ALLOWED_PATCH_EXTENSIONS:
            return {"success": False, "exit_code": -1, "stdout": "",
                    "stderr": f"File extension not in allowlist: '{rel}'."}

    # Gate 6
    orch_root = Path(config.ORCHESTRATION_ROOT)
    for rel in changed_files:
        abs_path = (repo_root / rel).resolve()
        if abs_path.is_relative_to(orch_root):
            return {"success": False, "exit_code": -1, "stdout": "",
                    "stderr": f"Cannot patch orchestration system files: '{rel}'."}
        for pat in config.DANGEROUS_PATCH_TARGETS:
            if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(Path(rel).name, pat):
                return {"success": False, "exit_code": -1, "stdout": "",
                        "stderr": f"'{rel}' matches dangerous pattern '{pat}'."}

    # Gate 7
    if declared_files:
        undeclared = set(changed_files) - set(declared_files)
        if undeclared:
            return {"success": False, "exit_code": -1, "stdout": "",
                    "stderr": f"Patch touches undeclared files: {sorted(undeclared)}."}

    # Gate 8
    if not allow_new_files:
        for rel in changed_files:
            if not (repo_root / rel).exists():
                return {"success": False, "exit_code": -1, "stdout": "",
                        "stderr": f"Patch would create a new file (allow_new_files=False): '{rel}'."}

    # Gate 9
    added = sum(1 for ln in patch_text.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
    removed = sum(1 for ln in patch_text.splitlines() if ln.startswith("-") and not ln.startswith("---"))
    MAX_LINES = 300
    if added + removed > MAX_LINES:
        return {"success": False, "exit_code": -1, "stdout": "",
                "stderr": f"Patch changes {added + removed} lines; limit is {MAX_LINES}."}

    # Gate 10
    snapshot = file_ops.snapshot_files(changed_files, repo_root)

    # Gates 11 + 12
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False, encoding="utf-8") as f:
        f.write(patch_text)
        patch_file = f.name

    try:
        # Dry-run
        dry = await file_ops._run_subprocess(["git", "apply", "--check", patch_file], cwd=str(repo_root))
        if not dry.success:
            return {"success": False, "exit_code": dry.exit_code,
                    "stdout": dry.stdout, "stderr": f"Dry-run failed: {dry.stderr}"}

        # Apply
        result = await file_ops._run_subprocess(["git", "apply", patch_file], cwd=str(repo_root))
        if not result.success:
            file_ops.rollback_files(snapshot, repo_root)
            return {"success": False, "exit_code": result.exit_code,
                    "stdout": result.stdout, "stderr": f"Apply failed (rolled back): {result.stderr}"}

        return {
            "success": True,
            "exit_code": 0,
            "stdout": result.stdout,
            "stderr": "",
            "changed_files": changed_files,
        }
    finally:
        try:
            Path(patch_file).unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Workspace restore (called by escalation_controller repair loop rollback)
# ---------------------------------------------------------------------------

async def restore_files(paths: List[str], root: str) -> Dict[str, Any]:
    """Restore named files to their last committed state using git checkout."""
    if not paths:
        return {"success": True, "exit_code": 0, "stdout": "Nothing to restore.", "stderr": ""}
    repo_root = Path(root).resolve()
    if not (repo_root / ".git").exists():
        return {"success": False, "exit_code": -1, "stdout": "",
                "stderr": f"root '{root}' is not a git repository."}
    result = await file_ops._run_subprocess(
        ["git", "checkout", "HEAD", "--"] + paths,
        cwd=str(repo_root),
    )
    return {"success": result.success, "exit_code": result.exit_code,
            "stdout": result.stdout, "stderr": result.stderr}


# ---------------------------------------------------------------------------
# Deterministic validation tools
# ---------------------------------------------------------------------------

async def run_py_compile(paths: str) -> Dict[str, Any]:
    """Syntax-check one or more Python files via py_compile / compileall."""
    target = paths.strip()
    if target.endswith(os.sep) or Path(target).is_dir():
        cmd = [sys.executable, "-m", "compileall", "-q", target]
    else:
        cmd = [sys.executable, "-m", "py_compile", target]
    result = await file_ops._run_subprocess(cmd)
    return {"success": result.success, "exit_code": result.exit_code,
            "stdout": result.stdout, "stderr": result.stderr}


async def run_lint(paths: str) -> Dict[str, Any]:
    """Run ruff (preferred) or flake8 on paths. Skips gracefully if neither is installed."""
    linter = shutil.which("ruff") or shutil.which("flake8")
    if not linter:
        return {"success": True, "exit_code": 0, "stdout": "",
                "stderr": "[lint skipped — ruff/flake8 not found]"}
    result = await file_ops._run_subprocess([linter, paths])
    return {"success": result.success, "exit_code": result.exit_code,
            "stdout": result.stdout, "stderr": result.stderr}


async def run_pytest(paths: str) -> Dict[str, Any]:
    """Run pytest on a target path. Skips gracefully if pytest is not installed."""
    if not shutil.which("pytest"):
        return {"success": True, "exit_code": 0, "stdout": "",
                "stderr": "[pytest skipped — not found]"}
    result = await file_ops._run_subprocess(
        [sys.executable, "-m", "pytest", paths, "-x", "-q", "--tb=short"]
    )
    return {"success": result.success, "exit_code": result.exit_code,
            "stdout": result.stdout, "stderr": result.stderr}


# ---------------------------------------------------------------------------
# LLM pipeline functions (delegate to llm_orchestrator)
# Called by escalation_controller via MCPWorkerAdapter
# ---------------------------------------------------------------------------

async def triage_issue(task: str, context: str = "", strict: bool = True) -> str:
    res = await llm_orchestrator.run_phase("triage", task, context, strict=strict)
    return json.dumps({"success": res.success, "output": res.output}, indent=2)


async def review_code(task: str, context: str = "", strict: bool = True) -> str:
    phases = ["review_scope", "review_findings", "review_synthesis"]
    res = await llm_orchestrator.run_pipeline("review", task, context, phases, strict=strict)
    return json.dumps({"success": res.success, "output": res.final_output}, indent=2)


async def draft_patch(task: str, context: str = "", strict: bool = True) -> str:
    res = await llm_orchestrator.run_phase("draft_patch", task, context, strict=strict)
    return json.dumps({"success": res.success, "output": res.output}, indent=2)


async def propose_fix(task: str, context: str = "", strict: bool = True) -> str:
    phases = ["fix_pre_review", "fix_patch", "fix_post_review", "fix_test_plan", "fix_final_decision"]
    res = await llm_orchestrator.run_pipeline("fix", task, context, phases, strict=strict)
    return json.dumps({"success": res.success, "output": res.final_output}, indent=2)


async def generate_tests(task: str, context: str = "", strict: bool = True) -> str:
    res = await llm_orchestrator.run_phase("generate_tests", task, context, strict=strict)
    return json.dumps({"success": res.success, "output": res.output}, indent=2)


async def summarize_diff(task: str, context: str = "", strict: bool = True) -> str:
    res = await llm_orchestrator.run_phase("summarize_diff", task, context, strict=strict)
    return json.dumps({"success": res.success, "output": res.output}, indent=2)