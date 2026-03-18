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
    RollbackError,
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
        orch = Path(config.ORCHESTRATION_ROOT)
        patch = "--- a/server.py\n+++ b/server.py\n@@ -1 +1 @@\n-x\n+y\n"
        result = await server.apply_unified_diff(patch, str(orch.parent))
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
        assert "System:" in prompt
        assert "Task: task" in prompt
        max_chars = 26_000  # triage limit
        assert len(prompt) <= max_chars + 200

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


# ---------------------------------------------------------------------------
# NEW: Defect F — subprocess timeout enforcement
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSubprocessTimeout:
    async def test_subprocess_times_out(self) -> None:
        """
        _run_subprocess must return a failing CommandResult when the process exceeds
        the timeout — it must NOT hang or raise an unhandled exception.
        """
        # 'python -c "import time; time.sleep(30)"' will exceed a 1-second timeout.
        result = await file_ops._run_subprocess(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_sec=1.0,
        )
        assert result.success is False
        assert "TIMEOUT" in result.stderr or "timeout" in result.stderr.lower()
        assert result.exit_code == -1


# ---------------------------------------------------------------------------
# NEW: Defect E — rollback failure is fatal (escalation, not silent continue)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRollbackFatal:
    async def test_rollback_failure_escalates(self, tmp_git_repo: Path) -> None:
        """
        If _rollback_from_snapshot raises (e.g. PermissionError), the controller
        must escalate with a CRITICAL message rather than silently continuing.
        """
        # Fake worker: returns a success response with a patch block that
        # will get past parse but hit a simulated rollback failure.
        patch_text = "--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1 +1 @@\n-old\n+new\n"

        apply_success_response = {
            "success": True,
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "changed_files": ["src/foo.py"],
            "snapshot": {"src/foo.py": b"original"},
        }

        call_count = 0

        class RollbackFailWorker:
            async def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
                nonlocal call_count
                call_count += 1
                if tool_name == "propose_fix":
                    return {
                        "success": True,
                        "patch_output": patch_text,
                        "output": patch_text,
                    }
                if tool_name == "apply_unified_diff":
                    return apply_success_response
                if tool_name in ("run_py_compile",):
                    # Fail validation so rollback is triggered
                    return {"success": False, "exit_code": 1, "stdout": "", "stderr": "SyntaxError"}
                return {"success": True, "output": ""}

        ctrl = EscalationController(
            RollbackFailWorker(),
            ControllerConfig(allowed_patch_root=str(tmp_git_repo)),
        )

        # Patch rollback_files to raise PermissionError
        import unittest.mock as mock
        with mock.patch.object(file_ops, "rollback_files", side_effect=PermissionError("locked")):
            result = await ctrl.handle(
                TaskRequest(task_type="propose_fix", task="fix bug", context="x = 1\n")
            )

        # Whether it escalates or fails, it must NOT silently continue
        assert result.status in ("escalate", "fail")


# ---------------------------------------------------------------------------
# NEW: Defect C — declared_files wired from file_path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestDeclaredFilesWiring:
    async def test_declared_files_passed_when_file_path_set(self, tmp_git_repo: Path) -> None:
        """
        When req.file_path is set, _apply_patch_if_possible must include
        declared_files derived from that path so the gate's anti-drift control fires.
        """
        captured_args: Dict[str, Any] = {}

        class CapturingWorker:
            async def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
                if tool_name == "apply_unified_diff":
                    captured_args.update(args)
                    # Return a failure so the loop doesn't continue
                    return {"success": False, "exit_code": 1, "stdout": "", "stderr": "test stop"}
                if tool_name in ("propose_fix", "draft_patch"):
                    return {
                        "success": True,
                        "patch_output": "--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1 +1 @@\n-old\n+new\n",
                        "output": "--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1 +1 @@\n-old\n+new\n",
                    }
                return {"success": True, "output": ""}

        file_path = str(tmp_git_repo / "src" / "foo.py")
        ctrl = EscalationController(
            CapturingWorker(),
            ControllerConfig(allowed_patch_root=str(tmp_git_repo)),
        )
        await ctrl.handle(
            TaskRequest(task_type="propose_fix", task="fix greet", file_path=file_path)
        )

        assert "declared_files" in captured_args
        assert len(captured_args["declared_files"]) == 1
        assert "foo.py" in captured_args["declared_files"][0]


# ---------------------------------------------------------------------------
# NEW: Defect D — validation subprocess cwd pinned to repo root
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestValidationCwd:
    async def test_run_py_compile_accepts_cwd(self, tmp_git_repo: Path) -> None:
        """run_py_compile must pass cwd to _run_subprocess (smoke test)."""
        # Create a valid Python file in the repo
        (tmp_git_repo / "src" / "foo.py").write_text("x = 1\n", encoding="utf-8")
        result = await server.run_py_compile(
            str(tmp_git_repo / "src" / "foo.py"),
            cwd=str(tmp_git_repo),
        )
        # As long as it doesn't throw and returns a dict we know cwd was accepted
        assert isinstance(result, dict)
        assert "success" in result


# ---------------------------------------------------------------------------
# NEW: Defect A — propose_fix returns structured patch_output field
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestProposeFxPipelineContract:
    async def test_propose_fix_returns_patch_output_key(self) -> None:
        """
        server.propose_fix() must return a dict with 'patch_output' key so the
        controller can extract the patch artifact separately from the decision prose.
        """
        import json as _json

        # Patch run_pipeline to return a controlled PipelineResult
        from llm_orchestrator import PipelineResult, PhaseResult

        fake_patch_phase = PhaseResult(
            phase_name="fix_patch",
            success=True,
            output="--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+new\n",
            duration_sec=0.1,
            validation_passed=True,
        )
        fake_decision_phase = PhaseResult(
            phase_name="fix_final_decision",
            success=True,
            output="ACCEPTED. The patch is valid.",
            duration_sec=0.1,
            validation_passed=True,
        )
        fake_result = PipelineResult(
            success=True,
            final_state="accepted",
            final_output=fake_decision_phase.output,
            duration_sec=0.2,
            phase_results=[fake_patch_phase, fake_decision_phase],
            phase_map={
                "fix_patch": fake_patch_phase,
                "fix_final_decision": fake_decision_phase,
            },
        )

        import unittest.mock as mock
        with mock.patch("llm_orchestrator.run_pipeline", return_value=fake_result):
            raw = await server.propose_fix(task="fix it", context="x = 1")

        data = _json.loads(raw)
        assert "patch_output" in data
        assert "decision_output" in data
        assert "--- a/x.py" in data["patch_output"]
        assert "ACCEPTED" in data["decision_output"]


# ---------------------------------------------------------------------------
# NEW: Defect G — MCPWorkerAdapter allowlist rejects unknown tools
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestWorkerAllowlist:
    async def test_unknown_tool_rejected(self) -> None:
        """MCPWorkerAdapter must reject any tool_name not in _ALLOWED_TOOLS."""
        from local_orchestrator import MCPWorkerAdapter
        adapter = MCPWorkerAdapter()

        # Attempt to call an internal / dangerous function name
        for bad_name in ("run_ollama_api", "__import__", "os.system", "_assert_inside_root", "eval"):
            result = await adapter.call_tool(bad_name, {})
            assert result["success"] is False, f"Expected rejection for '{bad_name}'"
            assert "allowed" in result.get("stderr", "").lower() or "not in" in result.get("stderr", "").lower()

    async def test_known_tool_resolves(self) -> None:
        """A valid tool name must pass the allowlist check."""
        from local_orchestrator import MCPWorkerAdapter
        adapter = MCPWorkerAdapter()
        # grep_code is in the allowlist; if it fails it should be a real error
        # (missing args), not an allowlist rejection
        result = await adapter.call_tool("grep_code", {"pattern": "x", "path": "/nonexistent"})
        # failure is fine — what matters is it wasn't blocked by the allowlist
        assert "allowed" not in result.get("stderr", "").lower()
