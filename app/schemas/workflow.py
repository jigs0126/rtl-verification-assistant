"""
RTL Workflow Schemas

What they are: typed request/result shapes for the three-step design flow
- Step 1: specification  -> RTL code
- Step 2: RTL + verification plan -> testbench code
- Step 3: RTL + testbench + simulator console log -> debug analysis + fixes

Why required: same reason as schemas/verification.py -- an LLM's raw text
must be parsed into validated models so a malformed response fails loudly
instead of rendering garbage in the UI.

Unlike the RAG-based tabs, this flow is grounded in what the user pastes
in (their spec, their plan, their log), not in the indexed knowledge base.
"""

from __future__ import annotations

from pydantic import BaseModel


# ---- Step 1: spec -> RTL ---------------------------------------------------

class RTLGenRequest(BaseModel):
    specification: str
    module_name: str = ""  # optional; LLM chooses one if blank


class RTLGenResult(BaseModel):
    """rtl_code is LLM-produced text -- NOT compiled, linted or simulated."""

    rtl_code: str
    module_name: str
    interface_summary: list[str]
    design_decisions: list[str]
    assumptions: list[str]
    warnings: list[str]


# ---- Step 2: RTL + plan -> testbench ---------------------------------------

class PlanTestbenchRequest(BaseModel):
    rtl_code: str
    verification_plan: str
    module_name: str = ""


class PlanTestbenchResult(BaseModel):
    """testbench_code is LLM-produced text -- NOT compiled or simulated."""

    testbench_code: str
    test_scenarios: list[str]
    plan_coverage: list[str]      # which plan items are covered / not covered
    assumptions: list[str]
    warnings: list[str]


# ---- Step 3: simulator log -> debug + fixes --------------------------------

class LogSummary(BaseModel):
    """Facts extracted deterministically from the console log (no LLM)."""

    passed: int
    failed: int
    failing_lines: list[str]
    error_lines: list[str]       # compile / elaboration / runtime errors
    has_result_markers: bool     # False if the log had no PASS/FAIL lines at all

    @property
    def all_passed(self) -> bool:
        return self.has_result_markers and self.failed == 0 and not self.error_lines


class SimDebugRequest(BaseModel):
    rtl_code: str
    testbench_code: str
    console_log: str
    verification_plan: str = ""


class SimDebugResult(BaseModel):
    """Root causes and fixes are hypotheses until re-simulated."""

    log_summary: LogSummary
    summary: str
    root_causes: list[str]        # each labeled [RTL] / [TESTBENCH] / [UNKNOWN]
    suggested_fixes: list[str]
    corrected_code: str           # "" when the model proposed no patch
    corrected_code_target: str    # "RTL" | "TESTBENCH" | ""
    confidence: str
    warnings: list[str]
