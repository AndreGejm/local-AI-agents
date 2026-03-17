"""
tests/test_safety_gate.py

Critical invariant tests for the local-expert MCP safety layer.
Run with: python -m pytest tests/test_safety_gate.py -v

These tests validate the gates in server.apply_unified_diff,
the rollback logic in file_ops, and the routing decisions in
escalation_controller without touching a real Ollama instance.

Python 3.14 compatible.
"""
from __future__ import annotations

import asyncio
import os
import sys
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import pytest

# Make sure imports resolve from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

import utils.file_ops as file_ops
import config
from escalation_controller import (
    ControllerConfig,
    EscalationController,
    TaskRequest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_git_repo(tmp_path: Path) -> Path:
    """Create a minimal git repo with one Python file."""
    (tmp_path / ".git").mkdir()
    py_file = tmp_path / "src" / "foo.py"
    py_file.parent.mkdir(parents=True)
    # Use binary mode to avoid CRLF translation on Windows
    py_file.write_bytes(b'def greet(name: str) -> str:\n    return f"Hello, {name}"\n')
    return tmp_path


def _make_valid_patch(rel_path: str, old_line: str, new_line: str) -> str:
    """Build a minimal unified diff."""
    return textwrap.dedent(f"""\
        --- a/{rel_path}
        +++ b/{rel_path}
        @@ -1,1 +1,1 @@
        -{old_line}
        +{new_line}
    """)


# ---------------------------------------------------------------------------
# utils.file_ops — snapshot / rollback
# ---------------------------------------------------------------------------

class TestSnapshotRollback:
    def test_snapshot_captures_existing_file(self, tmp_git_repo: Path) -> None:
        original = b'def greet(name: str) -> str:\n    return f"Hello, {name}"\n'
        snap = file_ops.snapshot_files(["src/foo.py"], tmp_git_repo)
        assert snap["src/foo.py"] == original

    def test_snapshot_stores_none_for_missing_file(self, tmp_git_repo: Path) -> None:
        snap = file_ops.snapshot_files(["src/does_not_exist.py"], tmp_git_repo)
        assert snap["src/does_not_exist.py"] is None

    def test_rollback_restores_modified_file(self, tmp_git_repo: Path) -> None:
        original = (tmp_git_repo / "src" / "foo.py").read_bytes()
        snap = file_ops.snapshot_files(["src/foo.py"], tmp_git_repo)
        (tmp_git_repo / "src" / "foo.py").write_text("# modified", encoding="utf-8")
        file_ops.rollback_files(snap, tmp_git_repo)
        assert (tmp_git_repo / "src" / "foo.py").read_bytes() == original

    def test_rollback_deletes_newly_created_file(self, tmp_git_repo: Path) -> None:
        snap = file_ops.snapshot_files(["src/new.py"], tmp_git_repo)  # None — did not exist
        (tmp_git_repo / "src" / "new.py").write_text("# new", encoding="utf-8")
        file_ops.rollback_files(snap, tmp_git_repo)
        assert not (tmp_git_repo / "src" / "new.py").exists()


# ---------------------------------------------------------------------------
# utils.file_ops — parse_patch_manifest
# ---------------------------------------------------------------------------

class TestParsePatchManifest:
    def test_parses_git_style_paths(self) -> None:
        patch = "--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1 +1 @@\n-old\n+new\n"
        paths = file_ops.parse_patch_manifest(patch)
        assert paths == ["src/foo.py"]

    def test_ignores_dev_null(self) -> None:
        patch = "--- /dev/null\n+++ b/src/new.py\n"
        paths = file_ops.parse_patch_manifest(patch)
        assert paths == ["src/new.py"]

    def test_multiple_files(self) -> None:
        patch = (
            "--- a/a.py\n+++ b/a.py\n"
            "--- a/b.py\n+++ b/b.py\n"
        )
        paths = file_ops.parse_patch_manifest(patch)
        assert set(paths) == {"a.py", "b.py"}


# ---------------------------------------------------------------------------
# server.apply_unified_diff — safety gate
# ---------------------------------------------------------------------------

# We import server here so the test can patch git subprocess calls.
import server  # noqa: E402


@pytest.mark.asyncio
class TestApplyUnifiedDiffGate:
    async def test_rejects_non_git_root(self, tmp_path: Path) -> None:
        patch = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new\n"
        result = await server.apply_unified_diff(patch, str(tmp_path))
        assert result["success"] is False
        assert ".git" in result["stderr"]

    async def test_rejects_path_traversal(self, tmp_git_repo: Path) -> None:
        patch = "--- a/../../evil.py\n+++ b/../../evil.py\n@@ -1 +1 @@\n-x\n+y\n"
        result = await server.apply_unified_diff(patch, str(tmp_git_repo))
        assert result["success"] is False
        # Either traversal detected or no changes found (both are correct rejections)
        assert not result["success"]

    async def test_rejects_yml_extension(self, tmp_git_repo: Path) -> None:
        (tmp_git_repo / ".github").mkdir()
        (tmp_git_repo / ".github" / "ci.yml").write_text("name: CI", encoding="utf-8")
        patch = "--- a/.github/ci.yml\n+++ b/.github/ci.yml\n@@ -1 +1 @@\n-name: CI\n+name: EVIL\n"
        result = await server.apply_unified_diff(patch, str(tmp_git_repo))
        assert result["success"] is False
        assert "allowlist" in result["stderr"] or "deny" in result["stderr"] or "extension" in result["stderr"]

    async def test_rejects_dangerous_pattern(self, tmp_git_repo: Path) -> None:
        (tmp_git_repo / "Makefile").write_text("build:", encoding="utf-8")
        patch = "--- a/Makefile\n+++ b/Makefile\n@@ -1 +1 @@\n-build:\n+evil:\n"
        result = await server.apply_unified_diff(patch, str(tmp_git_repo))
        assert result["success"] is False

    async def test_rejects_undeclared_file(self, tmp_git_repo: Path) -> None:
        patch = "--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1 +1 @@\n-x\n+y\n"
        result = await server.apply_unified_diff(
            patch,
            str(tmp_git_repo),
            declared_files=["src/bar.py"],  # wrong file
        )
        assert result["success"] is False
        assert "undeclared" in result["stderr"]

    async def test_rejects_new_file_by_default(self, tmp_git_repo: Path) -> None:
        patch = "--- /dev/null\n+++ b/src/brand_new.py\n@@ -0,0 +1 @@\n+print('hi')\n"
        result = await server.apply_unified_diff(
            patch, str(tmp_git_repo), allow_new_files=False
        )
        assert result["success"] is False
        assert "new file" in result["stderr"]

    async def test_rejects_orchestration_self_patch(self, tmp_git_repo: Path) -> None:
        # Simulate a patch targeting the orchestration directory itself.
        orch = Path(config.ORCHESTRATION_ROOT)
        # Temporarily make the repo_root equal to the parent of orch so the
        # relative path would point inside orch.
        patch = "--- a/server.py\n+++ b/server.py\n@@ -1 +1 @@\n-x\n+y\n"
        # Point root at the orchestration directory's parent so server.py is inside it.
        result = await server.apply_unified_diff(patch, str(orch.parent))
        # May fail for several reasons; the key is it must not succeed.
        assert result["success"] is False


# ---------------------------------------------------------------------------
# escalation_controller — routing invariants
# ---------------------------------------------------------------------------

class FakeWorker:
    """Stub worker that returns configurable responses."""

    def __init__(self, response: Dict[str, Any]) -> None:
        self._response = response

    async def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        return self._response


@pytest.mark.asyncio
class TestControllerRouting:
    async def test_mutating_task_without_patch_root_escalates(self) -> None:
        worker = FakeWorker({"success": True, "output": "5. Code block\n```diff\n--- a/x.py\n+++ b/x.py\n```"})
        ctrl = EscalationController(worker, ControllerConfig(allowed_patch_root=None))
        result = await ctrl.handle(TaskRequest(task_type="propose_fix", task="fix bug", context="x = 1\n"))
        assert result.status == "escalate"
        assert "allowed_patch_root" in result.reason or "LOCAL_EXPERT_REPO_ROOT" in result.reason

    async def test_read_only_task_proceeds_locally(self) -> None:
        worker = FakeWorker({
            "success": True,
            "output": "1. Symptom summary\n2. Likely causes\n3. Evidence\n4. Reproduction ideas\n5. Minimal patch candidates\n6. Unknowns"
        })
        ctrl = EscalationController(worker, ControllerConfig())
        result = await ctrl.handle(TaskRequest(task_type="triage", task="bug", context="x = 1\n" * 10))
        # Should not escalate purely because of missing patch root (read-only)
        assert result.status != "escalate" or "allowed_patch_root" not in result.reason

    async def test_dangerous_file_target_escalates_for_mutating_task(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        makefile = tmp_path / "Makefile"
        makefile.write_text("build:", encoding="utf-8")
        worker = FakeWorker({"success": True, "output": "some output"})
        ctrl = EscalationController(
            worker,
            ControllerConfig(allowed_patch_root=str(tmp_path)),
        )
        result = await ctrl.handle(
            TaskRequest(task_type="draft_patch", task="change build", file_path=str(makefile))
        )
        assert result.status == "escalate"
        assert "dangerous" in result.reason.lower() or "pattern" in result.reason.lower()

    async def test_escalation_signal_in_prose_does_not_trigger(self) -> None:
        """A prose sentence containing 'ESCALATE:' mid-line must not be treated as an escalation signal."""
        worker = FakeWorker({
            "success": True,
            "output": "You should ESCALATE: your concerns to management.\n1. Symptom summary\n2. Likely causes\n3. Evidence\n4. Reproduction ideas\n5. Minimal patch candidates\n6. Unknowns",
        })
        ctrl = EscalationController(worker, ControllerConfig())
        result = await ctrl.handle(TaskRequest(task_type="triage", task="bug", context="x\n"))
        assert result.status != "escalate"

    async def test_line_start_escalation_signal_is_respected(self) -> None:
        """ESCALATE: at the start of a line must trigger escalation."""
        worker = FakeWorker({
            "success": True,
            "output": "Analysis complete.\nESCALATE: Task requires paid model — too risky.\n",
        })
        ctrl = EscalationController(worker, ControllerConfig())
        result = await ctrl.handle(TaskRequest(task_type="triage", task="bug", context="x\n"))
        assert result.status == "escalate"


# ---------------------------------------------------------------------------
# llm_orchestrator — prompt budget invariant
# ---------------------------------------------------------------------------

from llm_orchestrator import build_phase_prompt  # noqa: E402


class TestPromptBuilding:
    def test_system_instruction_survives_large_context(self) -> None:
        """Header must not be truncated even when context is massive."""
        large_context = "x" * 50_000
        prompt = build_phase_prompt("triage", "find the bug", large_context)
        assert "first-pass bug triage" in prompt.lower()
        assert "find the bug" in prompt

    def test_context_is_trimmed_not_header(self) -> None:
        large_context = "UNIQUE_MARKER_" + ("y" * 50_000)
        prompt = build_phase_prompt("triage", "task", large_context)
        # Header must be present
        assert "System:" in prompt
        assert "Task: task" in prompt
        # Context may be truncated — that is expected
        max_chars = 26_000  # triage limit
        assert len(prompt) <= max_chars + 200  # small tolerance for formatting

    def test_prior_results_are_compressed(self) -> None:
        from llm_orchestrator import PhaseResult
        prior = PhaseResult(
            phase_name="review_scope",
            success=True,
            output="1. Scope reviewed\nAll good.\n2. Coverage limits\nLimited.\n3. Suspicious areas\nNone.",
            duration_sec=1.0,
            validation_passed=True,
        )
        prompt = build_phase_prompt("review_findings", "review this", "ctx", [prior])
        assert "review_scope" in prompt
        assert "1. Scope reviewed" in prompt
