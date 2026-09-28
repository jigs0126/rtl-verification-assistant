"""
RTL Workflow

What it is: the orchestrators for the three-step design flow -
RTLGenerator (spec -> RTL), PlanTestbenchGenerator (RTL + verification
plan -> testbench), SimulationDebugger (RTL + testbench + simulator
console log -> root causes and fixes) - plus parse_console_log(), a
deterministic (no LLM) extractor of pass/fail facts from a log.

Why required: this is the human-in-the-loop "generate -> simulate in
YOUR tool -> paste the log back -> debug" loop. The app never runs a
simulator itself; the user's simulator is the source of truth, and the
log they paste back is the evidence the debugger reasons over.

Input: user-pasted specification / plan / RTL / testbench / log text.
Output: RTLGenResult, PlanTestbenchResult, SimDebugResult.

How it connects: LLMClient (app/llm/client.py) + workflow_prompts.py +
response_parser.py, rendered by the "RTL Workflow" tab in app/main.py.

What would happen if it were removed: the app could only work on the
pre-indexed ALU project, not on a design the user brings.
"""

from __future__ import annotations

import re

from app.llm.client import LLMClient, LLMError
from app.llm.response_parser import extract_code_block, parse_labeled_sections, parse_numbered_list
from app.llm.workflow_prompts import (
    FAIL_MARKER,
    PASS_MARKER,
    build_plan_tb_prompt,
    build_rtl_gen_prompt,
    build_sim_debug_prompt,
)
from app.schemas.workflow import (
    LogSummary,
    PlanTestbenchRequest,
    PlanTestbenchResult,
    RTLGenRequest,
    RTLGenResult,
    SimDebugRequest,
    SimDebugResult,
)

_RTL_NOT_VERIFIED = "Generated RTL — not compiled, linted or simulation-verified."
_TB_NOT_VERIFIED = "Generated testbench — not compiled or simulation-verified."
_DEBUG_NOT_CONFIRMED = "Root causes and fixes are hypotheses until the fix is re-simulated."

# Log longer than this is trimmed (head + tail) before going to the LLM.
_MAX_LOG_CHARS = 20000

# Generic simulator error signatures (Questa/ModelSim, VCS, Xcelium, Verilator,
# Icarus, Vivado xsim). Case-insensitive, matched per line.
_ERROR_RE = re.compile(
    r"(\*\*\s*error|\berror[-:\s\[]|syntax error|fatal[:\s]|undeclared|"
    r"not declared|unresolved|cannot find|elaboration failed|compilation failed|"
    r"%error|assertion .*failed|\bxmvlog: \*e|\bxmelab: \*e)",
    re.IGNORECASE,
)


def parse_console_log(log: str) -> LogSummary:
    """Deterministically extract facts from a simulator console log:
    [PASS]/[FAIL] counts, the failing lines, and any error lines. Never
    calls the LLM, so these counts are facts, not guesses."""
    passed = failed = 0
    failing: list[str] = []
    errors: list[str] = []
    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue
        if PASS_MARKER in line:
            passed += 1
        elif FAIL_MARKER in line:
            failed += 1
            failing.append(line)
        elif _ERROR_RE.search(line):
            errors.append(line)
    return LogSummary(
        passed=passed,
        failed=failed,
        failing_lines=failing[:50],
        error_lines=errors[:30],
        has_result_markers=(passed + failed) > 0,
    )


def _trim_log(log: str) -> str:
    if len(log) <= _MAX_LOG_CHARS:
        return log
    half = _MAX_LOG_CHARS // 2
    return log[:half] + "\n... [log trimmed] ...\n" + log[-half:]


def _require(value: str, name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} must not be empty")


def _ensure(warnings: list[str], required: str) -> list[str]:
    return warnings if required in warnings else warnings + [required]


class RTLGenerator:
    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    def generate(self, request: RTLGenRequest) -> RTLGenResult:
        _require(request.specification, "specification")
        system, user = build_rtl_gen_prompt(request.specification, request.module_name)
        text = self._llm.complete(user, system=system, max_tokens=4000)
        s = parse_labeled_sections(
            text,
            ["MODULE_NAME", "INTERFACE", "DESIGN_DECISIONS", "ASSUMPTIONS", "WARNINGS", "RTL_CODE"],
        )
        return RTLGenResult(
            rtl_code=extract_code_block(s["RTL_CODE"]),
            module_name=s["MODULE_NAME"].strip().splitlines()[0].strip() if s["MODULE_NAME"].strip() else "",
            interface_summary=parse_numbered_list(s["INTERFACE"]),
            design_decisions=parse_numbered_list(s["DESIGN_DECISIONS"]),
            assumptions=parse_numbered_list(s["ASSUMPTIONS"]),
            warnings=_ensure(parse_numbered_list(s["WARNINGS"]), _RTL_NOT_VERIFIED),
        )


class PlanTestbenchGenerator:
    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    def generate(self, request: PlanTestbenchRequest) -> PlanTestbenchResult:
        _require(request.rtl_code, "rtl_code")
        _require(request.verification_plan, "verification_plan")
        system, user = build_plan_tb_prompt(request.rtl_code, request.verification_plan, request.module_name)
        text = self._llm.complete(user, system=system, max_tokens=6000)
        s = parse_labeled_sections(
            text, ["TEST_SCENARIOS", "PLAN_COVERAGE", "ASSUMPTIONS", "WARNINGS", "TESTBENCH_CODE"]
        )
        return PlanTestbenchResult(
            testbench_code=extract_code_block(s["TESTBENCH_CODE"]),
            test_scenarios=parse_numbered_list(s["TEST_SCENARIOS"]),
            plan_coverage=parse_numbered_list(s["PLAN_COVERAGE"]),
            assumptions=parse_numbered_list(s["ASSUMPTIONS"]),
            warnings=_ensure(parse_numbered_list(s["WARNINGS"]), _TB_NOT_VERIFIED),
        )


class SimulationDebugger:
    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    def analyze(self, request: SimDebugRequest) -> SimDebugResult:
        _require(request.console_log, "console_log")
        _require(request.rtl_code, "rtl_code")
        _require(request.testbench_code, "testbench_code")

        summary = parse_console_log(request.console_log)
        system, user = build_sim_debug_prompt(
            request.rtl_code,
            request.testbench_code,
            _trim_log(request.console_log),
            summary,
            request.verification_plan,
        )
        text = self._llm.complete(user, system=system, max_tokens=6000)
        s = parse_labeled_sections(
            text,
            [
                "SUMMARY",
                "ROOT_CAUSES",
                "SUGGESTED_FIXES",
                "CORRECTED_CODE_TARGET",
                "CORRECTED_CODE",
                "CONFIDENCE",
            ],
        )

        target = s["CORRECTED_CODE_TARGET"].strip().upper()
        target = target if target in ("RTL", "TESTBENCH") else ""
        code_section = s["CORRECTED_CODE"].strip()
        corrected = ""
        if target and not code_section.lower().startswith("none"):
            try:
                corrected = extract_code_block(code_section)
            except LLMError:
                corrected = ""  # a missing patch must not discard the analysis
        if not corrected:
            target = ""

        warnings = [_DEBUG_NOT_CONFIRMED]
        if not summary.has_result_markers and not summary.error_lines:
            warnings.append(
                f"No {PASS_MARKER}/{FAIL_MARKER} lines or error lines were found in the log; "
                "the analysis is based on limited evidence. Make sure the full console output was pasted."
            )

        return SimDebugResult(
            log_summary=summary,
            summary=s["SUMMARY"],
            root_causes=parse_numbered_list(s["ROOT_CAUSES"]),
            suggested_fixes=parse_numbered_list(s["SUGGESTED_FIXES"]),
            corrected_code=corrected,
            corrected_code_target=target,
            confidence=s["CONFIDENCE"],
            warnings=warnings,
        )
