from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Union

from fastmcp import FastMCP
import server

from escalation_controller import (
    ControllerConfig,
    ControllerResult,
    EscalationController,
    TaskRequest,
)

# Initialize the FastMCP agent
mcp = FastMCP("Local Agent Expert (Orchestrated)")

class WorkerClient(Protocol):
    async def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        ...


@dataclass
class OrchestratorRequest:
    task_type: str
    task: str
    context: str = ""
    file_path: Optional[str] = None
    symbol_name: Optional[str] = None
    diff_text: Optional[str] = None
    pytest_target: Optional[str] = None
    lint_target: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorResponse:
    status: str
    reason: str
    route: str
    controller_result: Dict[str, Any]
    escalation_bundle: Optional[Dict[str, Any]] = None


class LocalOrchestrator:
    """
    High-level orchestration layer.

    Responsibilities:
    - validate and normalize requests
    - invoke the local escalation controller
    - build a clean escalation bundle when the local path is rejected
    """

    def __init__(
        self,
        worker: WorkerClient,
        controller_config: Optional[ControllerConfig] = None,
    ) -> None:
        self.worker = worker
        self.controller = EscalationController(worker, controller_config)

    async def handle_or_fail(self, request: OrchestratorRequest) -> str:
        """Handle request and return JSON string for MCP output."""
        result = await self.handle(request)
        return json.dumps(asdict(result), ensure_ascii=False, indent=2)

    async def handle(self, request: OrchestratorRequest) -> OrchestratorResponse:
        normalized = self._normalize_request(request)

        controller_result = await self.controller.handle(
            TaskRequest(
                task_type=normalized.task_type,  # type: ignore[arg-type]
                task=normalized.task,
                context=normalized.context,
                file_path=normalized.file_path,
                symbol_name=normalized.symbol_name,
                diff_text=normalized.diff_text,
                pytest_target=normalized.pytest_target,
                lint_target=normalized.lint_target,
            )
        )

        if controller_result.status == "escalate":
            bundle = self._build_escalation_bundle(normalized, controller_result)
            return OrchestratorResponse(
                status="escalate",
                reason=controller_result.reason,
                route="stronger_model",
                controller_result=self._serialize_controller_result(controller_result),
                escalation_bundle=bundle,
            )

        if controller_result.status == "success":
            return OrchestratorResponse(
                status="success",
                reason=controller_result.reason,
                route="local_worker",
                controller_result=self._serialize_controller_result(controller_result),
                escalation_bundle=None,
            )

        return OrchestratorResponse(
            status="fail",
            reason=controller_result.reason,
            route="local_worker",
            controller_result=self._serialize_controller_result(controller_result),
            escalation_bundle=None,
        )

    def _normalize_request(self, req: OrchestratorRequest) -> OrchestratorRequest:
        task_type = req.task_type.strip()
        task = req.task.strip()

        if not task:
            raise ValueError("task cannot be empty")

        allowed = {
            "triage",
            "review",
            "draft_patch",
            "generate_tests",
            "summarize_diff",
            "propose_fix",
        }
        if task_type not in allowed:
            raise ValueError(f"unsupported task_type: {task_type}")

        context = req.context or ""

        # Optional convenience: if file_path exists and no explicit context was provided,
        # keep context empty here and let the controller decide whether to slice or read.
        if req.file_path:
            file_path = str(Path(req.file_path))
        else:
            file_path = None

        return OrchestratorRequest(
            task_type=task_type,
            task=task,
            context=context,
            file_path=file_path,
            symbol_name=req.symbol_name,
            diff_text=req.diff_text,
            pytest_target=req.pytest_target,
            lint_target=req.lint_target,
            metadata=req.metadata,
        )

    def _build_escalation_bundle(
        self,
        req: OrchestratorRequest,
        controller_result: ControllerResult,
    ) -> Dict[str, Any]:
        """
        Build a compact payload for Antigravity or a stronger model.
        """
        bundle: Dict[str, Any] = {
            "task_type": req.task_type,
            "task": req.task,
            "reason_for_escalation": controller_result.reason,
            "attempts": controller_result.attempts,
            "metadata": req.metadata,
            "controller_metadata": controller_result.escalation_metadata,
        }

        if req.file_path:
            bundle["file_path"] = req.file_path
        if req.symbol_name:
            bundle["symbol_name"] = req.symbol_name
        if req.pytest_target:
            bundle["pytest_target"] = req.pytest_target
        if req.lint_target:
            bundle["lint_target"] = req.lint_target
        if req.diff_text:
            bundle["diff_text"] = req.diff_text[:12000]
        if req.context:
            bundle["context"] = req.context[:12000]

        if controller_result.local_result:
            bundle["local_result"] = controller_result.local_result

        if controller_result.validations:
            bundle["validations"] = [asdict(v) for v in controller_result.validations]

        return bundle

    def _serialize_controller_result(self, result: ControllerResult) -> Dict[str, Any]:
        return {
            "status": result.status,
            "reason": result.reason,
            "tool_used": result.tool_used,
            "attempts": result.attempts,
            "local_result": result.local_result,
            "validations": [asdict(v) for v in result.validations],
            "escalation_metadata": result.escalation_metadata,
        }


# -----------------------------------------------------------------------------
# Worker adapter for server-side tools
#
# This connects the orchestrator logic back to the tools in server.py

class MCPWorkerAdapter:
    """
    Adapter that allows the orchestrator to call server-side tools.
    """

    async def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke a tool exposed by the local server.
        """
        # Retrieve tool function from server module
        fn = getattr(server, tool_name, None)
        if fn is None or not callable(fn):
            return {
                "success": False,
                "stderr": f"Tool '{tool_name}' not found in server module.",
            }
        # Execute the tool. All server tools are async.
        try:
            result = await fn(**args)
        except Exception as exc:
            return {
                "success": False,
                "stderr": str(exc),
            }
        # Attempt to parse JSON responses into dicts
        if isinstance(result, str):
            try:
                return json.loads(result)
            except Exception:
                return {"success": True, "output": result}
        if isinstance(result, dict):
            return result
        return {"success": True, "output": result}


# -----------------------------------------------------------------------------
# FastMCP Tool Definitions
#
# These are the entry points Antigravity uses.

_ORCHESTRATOR: Optional[LocalOrchestrator] = None

def get_orchestrator() -> LocalOrchestrator:
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None:
        _ORCHESTRATOR = LocalOrchestrator(
            worker=MCPWorkerAdapter(),
            controller_config=ControllerConfig(
                max_context_chars_local=30000,
                max_file_lines_local=1200,
                max_diff_lines_local=200,
                max_functions_local=40,
                max_local_repair_attempts=1,
                max_validation_failures_before_escalate=1,
                max_patch_chars=25000,
                allowed_patch_root=".",
            ),
        )
    return _ORCHESTRATOR

@mcp.tool()
async def triage_issue(issue: str, context: str = "") -> str:
    """Perform first-pass bug triage for a bounded issue and context."""
    orch = get_orchestrator()
    return await orch.handle_or_fail(OrchestratorRequest("triage", issue, context))

@mcp.tool()
async def review_code(task: str, context: str = "") -> str:
    """Perform a gated technical review. Orchestrator may escalate to stronger models."""
    orch = get_orchestrator()
    return await orch.handle_or_fail(OrchestratorRequest("review", task, context))

@mcp.tool()
async def draft_patch(task: str, context: str = "") -> str:
    """Draft a minimal code patch. Gated by complexity and context limits."""
    orch = get_orchestrator()
    return await orch.handle_or_fail(OrchestratorRequest("draft_patch", task, context))

@mcp.tool()
async def generate_tests(task: str, context: str = "") -> str:
    """Generate focused regression tests. Orchestrated for reliability."""
    orch = get_orchestrator()
    return await orch.handle_or_fail(OrchestratorRequest("generate_tests", task, context))

@mcp.tool()
async def summarize_diff(task: str, context: str = "") -> str:
    """Summarize a provided diff or bounded code change."""
    orch = get_orchestrator()
    return await orch.handle_or_fail(OrchestratorRequest("summarize_diff", task, context))

@mcp.tool()
async def propose_fix(task: str, context: str = "") -> str:
    """Run a multi-phase fix pipeline with deterministic validation and auto-escalation."""
    orch = get_orchestrator()
    return await orch.handle_or_fail(OrchestratorRequest("propose_fix", task, context))

# Re-expose stable utilities from server.py (Direct pass-through)
@mcp.tool()
async def read_file(path: str, max_chars: int = 10000) -> str:
    """Read a file from the repository and optionally truncate its content."""
    return await server.read_file(path, max_chars)

@mcp.tool()
async def write_file(path: str, content: str) -> str:
    """Write content to a file within the repository."""
    return await server.write_file(path, content)

@mcp.tool()
async def list_files(root: str = ".", include: str = "", exclude: str = "") -> str:
    """List files under a directory with optional filters."""
    return await server.list_files(root, include, exclude)

@mcp.tool()
async def grep_code(pattern: str, path: str) -> str:
    """Search for a pattern in a file and return matching lines."""
    return await server.grep_code(pattern, path)

@mcp.tool()
async def extract_function(path: str, symbol: str, include_docstring: bool = False) -> str:
    """Extract source code of a function/class from a Python module."""
    return await server.extract_function(path, symbol, include_docstring)

if __name__ == "__main__":
    mcp.run()