from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Protocol, Sequence


Status = Literal["success", "fail", "escalate"]
TaskType = Literal[
    "triage",
    "review",
    "draft_patch",
    "generate_tests",
    "summarize_diff",
    "propose_fix",
]


class WorkerClient(Protocol):
    """Protocol for your MCP/local worker adapter."""

    async def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        ...


@dataclass
class EscalationDecision:
    status: Status
    tool_name: Optional[str] = None
    reason: str = ""
    complexity_score: int = 0
    sliced_context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    success: bool
    step: str
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ControllerConfig:
    max_context_chars_local: int = 30_000
    max_file_lines_local: int = 1_200
    max_diff_lines_local: int = 200
    max_functions_local: int = 40
    max_local_repair_attempts: int = 1
    max_validation_failures_before_escalate: int = 1
    max_patch_chars: int = 25_000
    allowed_patch_root: Optional[str] = None


@dataclass
class TaskRequest:
    task_type: TaskType
    task: str
    context: str = ""
    file_path: Optional[str] = None
    symbol_name: Optional[str] = None
    diff_text: Optional[str] = None
    pytest_target: Optional[str] = None
    lint_target: Optional[str] = None


@dataclass
class ControllerResult:
    status: Status
    reason: str
    tool_used: Optional[str] = None
    local_result: Optional[Dict[str, Any]] = None
    validations: List[ValidationResult] = field(default_factory=list)
    attempts: int = 0
    escalation_metadata: Dict[str, Any] = field(default_factory=dict)


class EscalationController:
    def __init__(self, worker: WorkerClient, config: Optional[ControllerConfig] = None) -> None:
        self.worker = worker
        self.config = config or ControllerConfig()

    async def handle(self, req: TaskRequest) -> ControllerResult:
        decision = await self._preflight_decide(req)
        if decision.status == "escalate":
            return ControllerResult(
                status="escalate",
                reason=decision.reason,
                attempts=0,
                escalation_metadata=decision.metadata,
            )

        local_args = {
            "task": req.task,
            "context": decision.sliced_context if decision.sliced_context is not None else req.context,
            "strict": True,
        }

        attempts = 0
        validations: List[ValidationResult] = []

        if req.task_type in ("triage", "review", "generate_tests", "summarize_diff"):
            result = await self.worker.call_tool(decision.tool_name, local_args)
            parsed = self._normalize_worker_result(result)
            if parsed["status"] == "escalate":
                return ControllerResult(
                    status="escalate",
                    reason=parsed["reason"],
                    tool_used=decision.tool_name,
                    local_result=parsed,
                    attempts=1,
                )
            return ControllerResult(
                status=parsed["status"],
                reason=parsed["reason"],
                tool_used=decision.tool_name,
                local_result=parsed,
                attempts=1,
            )

        if req.task_type in ("draft_patch", "propose_fix"):
            while attempts <= self.config.max_local_repair_attempts:
                attempts += 1

                result = await self.worker.call_tool(decision.tool_name, local_args)
                parsed = self._normalize_worker_result(result)

                if parsed["status"] == "escalate":
                    return ControllerResult(
                        status="escalate",
                        reason=parsed["reason"],
                        tool_used=decision.tool_name,
                        local_result=parsed,
                        attempts=attempts,
                    )

                if parsed["status"] != "success":
                    return ControllerResult(
                        status="fail",
                        reason=parsed["reason"],
                        tool_used=decision.tool_name,
                        local_result=parsed,
                        attempts=attempts,
                    )

                # Attempt to extract the patch using the worker's extract_patch_block tool
                patch_text: Optional[str] = None
                try:
                    ext_raw = await self.worker.call_tool(
                        "extract_patch_block",
                        {"text": parsed.get("output", "")},
                    )
                    # ext_raw may be a dict or other structure; normalize if necessary
                    if isinstance(ext_raw, dict):
                        patch_text = ext_raw.get("patch_text") or None
                    # If ext_raw is not a dict (adapter may return string), fallback to regex
                except Exception:
                    patch_text = None
                # Fallback heuristic extraction if tool did not produce a patch
                if not patch_text:
                    patch_text = self._extract_patch_or_code(parsed.get("output", ""))
                if not patch_text:
                    return ControllerResult(
                        status="escalate",
                        reason="Local worker returned no extractable patch/code.",
                        tool_used=decision.tool_name,
                        local_result=parsed,
                        attempts=attempts,
                    )

                if len(patch_text) > self.config.max_patch_chars:
                    return ControllerResult(
                        status="escalate",
                        reason="Patch output too large for safe local application.",
                        tool_used=decision.tool_name,
                        local_result=parsed,
                        attempts=attempts,
                    )

                apply_result = await self._apply_patch_if_possible(patch_text)
                validations.append(apply_result)
                if not apply_result.success:
                    return ControllerResult(
                        status="escalate",
                        reason=f"Patch application failed: {apply_result.stderr or apply_result.stdout}",
                        tool_used=decision.tool_name,
                        local_result=parsed,
                        validations=validations,
                        attempts=attempts,
                    )

                post_validations = await self._run_post_patch_validations(req)
                validations.extend(post_validations)

                failed_validations = [v for v in validations if not v.success and v.step in {"py_compile", "lint", "pytest"}]
                if not failed_validations:
                    return ControllerResult(
                        status="success",
                        reason="Patch applied and validations passed.",
                        tool_used=decision.tool_name,
                        local_result=parsed,
                        validations=validations,
                        attempts=attempts,
                    )

                if attempts > self.config.max_local_repair_attempts:
                    return ControllerResult(
                        status="escalate",
                        reason="Deterministic validation failed after local repair attempts.",
                        tool_used=decision.tool_name,
                        local_result=parsed,
                        validations=validations,
                        attempts=attempts,
                    )

                failure_bundle = self._build_failure_bundle(req, failed_validations)
                local_args["context"] = self._merge_contexts(local_args["context"], failure_bundle)

            return ControllerResult(
                status="escalate",
                reason="Exceeded local repair attempts.",
                tool_used=decision.tool_name,
                validations=validations,
                attempts=attempts,
            )

        return ControllerResult(
            status="escalate",
            reason=f"Unsupported task type routing: {req.task_type}",
        )

    async def _preflight_decide(self, req: TaskRequest) -> EscalationDecision:
        tool_map: Dict[TaskType, str] = {
            "triage": "triage_issue",
            "review": "review_code",
            "draft_patch": "draft_patch",
            "generate_tests": "generate_tests",
            "summarize_diff": "summarize_diff",
            "propose_fix": "propose_fix",
        }

        complexity = 0
        metadata: Dict[str, Any] = {}

        context = req.context

        if req.file_path and not context and req.symbol_name:
            sliced = await self._try_extract_symbol(req.file_path, req.symbol_name)
            if sliced:
                context = sliced
                metadata["used_scope_slice"] = True
            else:
                full_text = Path(req.file_path).read_text(encoding="utf-8", errors="replace")
                context = full_text
                metadata["used_scope_slice"] = False

        if req.diff_text:
            diff_lines = len(req.diff_text.splitlines())
            metadata["diff_lines"] = diff_lines
            if diff_lines > self.config.max_diff_lines_local:
                return EscalationDecision(
                    status="escalate",
                    reason=f"Diff too large for safe local handling ({diff_lines} lines).",
                    complexity_score=100,
                    metadata=metadata,
                )

        context_chars = len(context)
        metadata["context_chars"] = context_chars
        if context_chars > self.config.max_context_chars_local:
            return EscalationDecision(
                status="escalate",
                reason=f"Context too large for reliable local handling ({context_chars} chars).",
                complexity_score=100,
                metadata=metadata,
            )

        if req.file_path and Path(req.file_path).exists():
            file_text = Path(req.file_path).read_text(encoding="utf-8", errors="replace")
            line_count = len(file_text.splitlines())
            function_count = self._count_functions(file_text)

            metadata["file_lines"] = line_count
            metadata["function_count"] = function_count

            if line_count > self.config.max_file_lines_local:
                return EscalationDecision(
                    status="escalate",
                    reason=f"File too large for reliable local review ({line_count} lines).",
                    complexity_score=100,
                    metadata=metadata,
                )

            if function_count > self.config.max_functions_local:
                return EscalationDecision(
                    status="escalate",
                    reason=f"Too many functions for reliable local reasoning ({function_count}).",
                    complexity_score=100,
                    metadata=metadata,
                )

            complexity += min(line_count // 100, 10)
            complexity += min(function_count // 5, 10)

        if req.task_type in ("propose_fix", "review"):
            complexity += 10
        if req.task_type == "summarize_diff":
            complexity += 2
        if req.task_type == "triage":
            complexity += 4

        if "async " in context:
            complexity += 5
            metadata["async_detected"] = True
        if "thread" in context or "lock" in context:
            complexity += 5
            metadata["concurrency_detected"] = True

        if complexity >= 20:
            return EscalationDecision(
                status="escalate",
                reason=f"Complexity score too high for reliable local execution ({complexity}).",
                complexity_score=complexity,
                metadata=metadata,
            )

        return EscalationDecision(
            status="success",
            tool_name=tool_map[req.task_type],
            reason="Local handling allowed.",
            complexity_score=complexity,
            sliced_context=context,
            metadata=metadata,
        )

    async def _try_extract_symbol(self, file_path: str, symbol_name: str) -> Optional[str]:
        try:
            result = await self.worker.call_tool(
                "extract_function",
                {"path": file_path, "symbol_name": symbol_name},
            )
            text = self._extract_text_field(result)
            return text.strip() if text.strip() else None
        except Exception:
            return None

    async def _apply_patch_if_possible(self, patch_text: str) -> ValidationResult:
        try:
            result = await self.worker.call_tool(
                "apply_unified_diff",
                {
                    "patch_text": patch_text,
                    "root": self.config.allowed_patch_root or ".",
                },
            )
            parsed = self._normalize_command_result("apply_patch", result)
            return parsed
        except Exception as exc:
            return ValidationResult(
                success=False,
                step="apply_patch",
                stderr=str(exc),
            )

    async def _run_post_patch_validations(self, req: TaskRequest) -> List[ValidationResult]:
        results: List[ValidationResult] = []

        compile_target = req.file_path or "."
        # Normalize arguments to match the server's deterministic helper signatures
        results.append(
            await self._safe_tool_command(
                "py_compile",
                "run_py_compile",
                {"paths": compile_target},
            )
        )

        if req.lint_target:
            results.append(
                await self._safe_tool_command(
                    "lint",
                    "run_lint",
                    {"paths": req.lint_target},
                )
            )

        if req.pytest_target:
            results.append(
                await self._safe_tool_command(
                    "pytest",
                    "run_pytest",
                    {"paths": req.pytest_target},
                )
            )

        return results

    async def _safe_tool_command(self, step: str, tool_name: str, args: Dict[str, Any]) -> ValidationResult:
        try:
            result = await self.worker.call_tool(tool_name, args)
            return self._normalize_command_result(step, result)
        except Exception as exc:
            return ValidationResult(
                success=False,
                step=step,
                stderr=str(exc),
            )

    def _normalize_worker_result(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        text = self._extract_text_field(raw)

        status: Status = "success"
        reason = "Worker completed successfully."

        lower = text.lower()
        if "strict validation failed" in lower:
            status = "fail"
            reason = "Worker output failed strict validation."
        elif "escalate:" in text:
            status = "escalate"
            reason = text.split("ESCALATE:", 1)[1].strip() or "Worker requested escalation."
        elif re.search(r"\bsuccess:\s*false\b", lower):
            status = "fail"
            reason = "Worker reported unsuccessful execution."

        return {
            "status": status,
            "reason": reason,
            "output": text,
            "raw": raw,
        }

    def _normalize_command_result(self, step: str, raw: Dict[str, Any]) -> ValidationResult:
        if isinstance(raw, dict):
            success = bool(raw.get("success", False))
            exit_code = raw.get("exit_code")
            stdout = str(raw.get("stdout", ""))
            stderr = str(raw.get("stderr", ""))
            return ValidationResult(
                success=success,
                step=step,
                exit_code=exit_code if isinstance(exit_code, int) else None,
                stdout=stdout,
                stderr=stderr,
                metadata=raw,
            )

        return ValidationResult(
            success=False,
            step=step,
            stderr=f"Unexpected command result shape: {type(raw)!r}",
        )

    def _extract_text_field(self, raw: Dict[str, Any]) -> str:
        """
        Extract a primary text field from a worker result.

        The worker may return responses with various field names depending on the
        tool used. This method now recognizes 'source' and 'patch_text' in addition
        to common keys like 'text', 'output', 'content', and 'result'. If none
        of these fields contain a string, the entire payload is JSON-dumped.
        """
        if isinstance(raw, dict):
            for key in ("text", "output", "content", "result", "source", "patch_text"):
                value = raw.get(key)
                if isinstance(value, str):
                    return value
        return json.dumps(raw, ensure_ascii=False, indent=2)

    def _extract_patch_or_code(self, output: str) -> Optional[str]:
        diff_match = re.search(r"```diff\s*(.*?)```", output, flags=re.DOTALL | re.IGNORECASE)
        if diff_match:
            return diff_match.group(1).strip()

        code_match = re.search(r"```(?:python|py|text)?\s*(.*?)```", output, flags=re.DOTALL | re.IGNORECASE)
        if code_match:
            return code_match.group(1).strip()

        section_match = re.search(r"5\.\s*Code(?: block)?\s*(.*)", output, flags=re.DOTALL | re.IGNORECASE)
        if section_match:
            return section_match.group(1).strip()

        return None

    def _build_failure_bundle(self, req: TaskRequest, failed_validations: Sequence[ValidationResult]) -> str:
        parts = [
            "[DETERMINISTIC VALIDATION FAILURES]",
            f"Task type: {req.task_type}",
            f"Task: {req.task}",
        ]
        for item in failed_validations:
            parts.append(f"[{item.step}] success={item.success} exit_code={item.exit_code}")
            if item.stdout:
                parts.append("STDOUT:")
                parts.append(item.stdout[:4000])
            if item.stderr:
                parts.append("STDERR:")
                parts.append(item.stderr[:4000])
        return "\n".join(parts)

    def _merge_contexts(self, original: str, extra: str) -> str:
        merged = (original + "\n\n" + extra).strip()
        if len(merged) > self.config.max_context_chars_local:
            return merged[-self.config.max_context_chars_local :]
        return merged

    def _count_functions(self, text: str) -> int:
        return len(re.findall(r"^\s*(?:async\s+def|def)\s+\w+\s*\(", text, flags=re.MULTILINE))