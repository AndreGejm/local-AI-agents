"""
llm_orchestrator.py

Builds prompts and runs LLM phases via the local Ollama model.
Enforces per-phase token budgets defined in config.PHASE_INPUT_BUDGETS.
Prompt truncation always preserves the system instruction and task header.

Python 3.14 compatible.
"""
from __future__ import annotations

import re
import time
import json
from dataclasses import dataclass
from typing import List, Optional, Sequence

from config import (
    PHASE_DEFINITIONS,
    PHASE_INPUT_BUDGETS,
    MAX_PROMPT_CHARS_PER_PHASE,
    DEFAULT_TIMEOUT_SEC,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY_SEC,
)
import llm_client


@dataclass(frozen=True)
class PhaseResult:
    phase_name: str
    success: bool
    output: str
    duration_sec: float
    validation_passed: bool
    raw_output: Optional[str] = None


@dataclass(frozen=True)
class PipelineResult:
    success: bool
    final_state: str
    final_output: str
    duration_sec: float
    phase_results: Sequence[PhaseResult]


# ---------------------------------------------------------------------------
# Marker validation
# ---------------------------------------------------------------------------

def validate_markers(output: str, markers: List[str]) -> List[str]:
    output_lower = output.lower()
    return [m for m in markers if m.lower() not in output_lower]


# ---------------------------------------------------------------------------
# Prior-output compression
# ---------------------------------------------------------------------------

def _compress_prior_output(text: str, max_chars: int) -> str:
    """
    Extract only numbered-heading sections from a prior phase output.
    This avoids re-sending full prose when only the summary matters.
    Falls back to head-truncation (not tail) if no structure is found.
    """
    sections = re.findall(r"(\d+\.\s+[^\n]+(?:\n(?!\d+\.\s)[^\n]*)*)", text)
    if sections:
        compressed = "\n".join(s.strip() for s in sections)
        return compressed[:max_chars]
    # Fallback: keep the beginning (system output tends to be most structured at the top).
    return text[:max_chars]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_phase_prompt(
    phase_name: str,
    task_description: str,
    base_context: Optional[str],
    prior_results: Optional[List[PhaseResult]] = None,
) -> str:
    """
    Assemble a prompt for the given phase, respecting PHASE_INPUT_BUDGETS.

    Budget allocation (in priority order, never truncated):
      1. System instruction + task header
      2. Prior phase summaries (compressed)
      3. Base context (truncated last if needed)

    This ensures the system instruction and task are always preserved.
    """
    if prior_results is None:
        prior_results = []

    phase_def = PHASE_DEFINITIONS[phase_name]
    system_prompt = phase_def["system_prompt"]
    budget = PHASE_INPUT_BUDGETS.get(phase_name, {"base_context": 20000, "prior_outputs": 0})

    # Immutable header — always kept intact.
    header = f"System: {system_prompt}\n\nTask: {task_description}\n\n"

    # Prior phase summaries — compressed to headings only, capped by budget slot.
    prior_budget = budget["prior_outputs"]
    prior_section = ""
    if prior_results and prior_budget > 0:
        per_phase = max(400, prior_budget // max(len(prior_results), 1))
        parts = [
            f"--- {r.phase_name} ---\n{_compress_prior_output(r.output, per_phase)}"
            for r in prior_results
        ]
        combined = "\n".join(parts)
        prior_section = f"Prior phase results:\n{combined[:prior_budget]}\n\n"

    # Base context — capped by budget slot, then trimmed further if total is still over limit.
    ctx_budget = budget["base_context"]
    ctx_text = (base_context or "")[:ctx_budget]
    context_section = f"Context:\n{ctx_text}\n\n" if ctx_text else ""

    prompt = header + context_section + prior_section

    # Final safety cap: if still over limit, shrink context only (header + prior are protected).
    max_chars = MAX_PROMPT_CHARS_PER_PHASE.get(phase_name, 32000)
    if len(prompt) > max_chars:
        fixed = header + prior_section
        remaining = max_chars - len(fixed)
        if remaining > 0 and base_context:
            context_section = f"Context:\n{base_context[:remaining]}\n\n"
        else:
            context_section = ""
        prompt = header + context_section + prior_section

    return prompt


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------

async def run_phase(
    phase_name: str,
    task_description: str,
    base_context: Optional[str] = None,
    prior_phase_results: Optional[List[PhaseResult]] = None,  # never use [] as default
    strict: bool = True,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
) -> PhaseResult:
    if prior_phase_results is None:
        prior_phase_results = []

    start_t = time.monotonic()
    prompt = build_phase_prompt(phase_name, task_description, base_context, prior_phase_results)

    response = await llm_client.run_ollama_api(prompt, phase_name, timeout_sec)
    output = response.get("response", "").strip()

    if not output:
        return PhaseResult(
            phase_name=phase_name,
            success=False,
            output="Empty output from model.",
            duration_sec=time.monotonic() - start_t,
            validation_passed=False,
        )

    phase_def = PHASE_DEFINITIONS[phase_name]
    missing = validate_markers(output, phase_def.get("required_markers", []))

    success = True
    if strict and missing:
        success = False
        output = f"[STRICT VALIDATION FAILED] Missing: {', '.join(missing)}\n\n" + output

    # Hard-fail on explicit pipeline_failed / rejected state in final decision.
    if phase_name == "fix_final_decision":
        lower = output.lower()
        if "pipeline_failed" in lower or "rejected" in lower:
            success = False

    return PhaseResult(
        phase_name=phase_name,
        success=success,
        output=output,
        duration_sec=time.monotonic() - start_t,
        validation_passed=(not missing),
        raw_output=output,
    )


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

async def run_pipeline(
    mode: str,
    task_description: str,
    context: Optional[str],
    phase_names: Sequence[str],
    strict: bool = True,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
) -> PipelineResult:
    start = time.monotonic()
    phase_results: List[PhaseResult] = []

    for phase_name in phase_names:
        res = await run_phase(
            phase_name,
            task_description,
            context,
            phase_results,
            strict,
            timeout_sec,
        )
        phase_results.append(res)
        if not res.success:
            break

    success = all(r.success for r in phase_results)
    final_output = phase_results[-1].output if phase_results else "No phases run."

    return PipelineResult(
        success=success,
        final_state="accepted" if success else "failed",
        final_output=final_output,
        duration_sec=time.monotonic() - start,
        phase_results=phase_results,
    )
