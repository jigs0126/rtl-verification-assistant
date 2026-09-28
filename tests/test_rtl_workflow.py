import pytest

from app.llm.client import FakeLLMClient, LLMError
from app.schemas.workflow import (
    PlanTestbenchRequest,
    RTLGenRequest,
    SimDebugRequest,
)
from app.verification.rtl_workflow import (
    PlanTestbenchGenerator,
    RTLGenerator,
    SimulationDebugger,
    parse_console_log,
)

RTL_RESPONSE = """\
MODULE_NAME:
counter

INTERFACE:
1. clk | input | 1 | clock
2. count | output | 8 | value

DESIGN_DECISIONS:
1. Synchronous reset

ASSUMPTIONS:
None.

WARNINGS:
None.

RTL_CODE:
```systemverilog
module counter(input logic clk, input logic rst, output logic [7:0] count);
  always_ff @(posedge clk) count <= rst ? '0 : count + 1;
endmodule
```
"""

TB_RESPONSE = """\
TEST_SCENARIOS:
1. reset_clears

PLAN_COVERAGE:
1. reset -> covered

ASSUMPTIONS:
None.

WARNINGS:
None.

TESTBENCH_CODE:
```systemverilog
module tb; endmodule
```
"""

DEBUG_RESPONSE = """\
SUMMARY:
One test failed.

ROOT_CAUSES:
1. [RTL] wraparound not handled in `count`

SUGGESTED_FIXES:
1. [RTL] change count + 1 to saturate

CORRECTED_CODE_TARGET:
RTL

CORRECTED_CODE:
```systemverilog
module counter; endmodule
```

CONFIDENCE:
medium - only one failing test
"""

LOG = """\
# [PASS] reset_clears
# [PASS] count_up
# [FAIL] wrap: expected=0 observed=256
# ** Error: (vsim-3) something odd
# SUMMARY: 2 passed, 1 failed
"""


# ---- log parsing (deterministic, no LLM) ----

def test_parse_console_log_counts_and_lines():
    s = parse_console_log(LOG)
    assert s.passed == 2
    assert s.failed == 1
    assert s.failing_lines == ["# [FAIL] wrap: expected=0 observed=256"]
    assert any("Error" in e for e in s.error_lines)
    assert s.has_result_markers
    assert not s.all_passed  # a failure and an error line


def test_parse_console_log_all_pass():
    s = parse_console_log("[PASS] a\n[PASS] b\nSUMMARY: 2 passed, 0 failed\nErrors: 0, Warnings: 0")
    assert s.passed == 2 and s.failed == 0
    assert s.error_lines == []  # "Errors: 0" must not count as an error line
    assert s.all_passed


def test_parse_console_log_without_markers_or_errors():
    s = parse_console_log("simulation ended at 100 ns")
    assert not s.has_result_markers
    assert not s.all_passed  # no evidence of a pass


def test_parse_console_log_detects_compile_error():
    s = parse_console_log("** Error: tb.sv(12): (vlog-2730) Undefined variable: foo.")
    assert s.error_lines and not s.has_result_markers


# ---- Step 1 ----

def test_rtl_generator_parses_result_and_adds_warning():
    res = RTLGenerator(FakeLLMClient(fixed_response=RTL_RESPONSE)).generate(
        RTLGenRequest(specification="8-bit counter")
    )
    assert res.module_name == "counter"
    assert "module counter" in res.rtl_code
    assert len(res.interface_summary) == 2
    assert any("not compiled" in w for w in res.warnings)


def test_rtl_generator_rejects_empty_spec():
    with pytest.raises(ValueError):
        RTLGenerator(FakeLLMClient(fixed_response=RTL_RESPONSE)).generate(RTLGenRequest(specification="  "))


def test_rtl_generator_malformed_response_raises():
    with pytest.raises(LLMError):
        RTLGenerator(FakeLLMClient(fixed_response="nope")).generate(RTLGenRequest(specification="x"))


def test_rtl_prompt_uses_requested_module_name():
    seen = {}

    def fn(prompt):
        return RTL_RESPONSE

    class Spy(FakeLLMClient):
        def complete(self, prompt, system=None, max_tokens=2048):
            seen["system"] = system
            return super().complete(prompt, system, max_tokens)

    RTLGenerator(Spy(response_fn=fn)).generate(RTLGenRequest(specification="x", module_name="my_ctr"))
    assert '"my_ctr"' in seen["system"]


# ---- Step 2 ----

def test_plan_testbench_generator_parses_result():
    res = PlanTestbenchGenerator(FakeLLMClient(fixed_response=TB_RESPONSE)).generate(
        PlanTestbenchRequest(rtl_code="module counter; endmodule", verification_plan="check reset")
    )
    assert "module tb" in res.testbench_code
    assert res.test_scenarios == ["reset_clears"]
    assert res.plan_coverage
    assert any("not compiled" in w for w in res.warnings)


def test_plan_testbench_requires_rtl_and_plan():
    gen = PlanTestbenchGenerator(FakeLLMClient(fixed_response=TB_RESPONSE))
    with pytest.raises(ValueError):
        gen.generate(PlanTestbenchRequest(rtl_code="", verification_plan="p"))
    with pytest.raises(ValueError):
        gen.generate(PlanTestbenchRequest(rtl_code="module m; endmodule", verification_plan=" "))


# ---- Step 3 ----

def _debug_req(log=LOG):
    return SimDebugRequest(rtl_code="module counter; endmodule", testbench_code="module tb; endmodule", console_log=log)


def test_sim_debugger_returns_fixes_and_patch():
    res = SimulationDebugger(FakeLLMClient(fixed_response=DEBUG_RESPONSE)).analyze(_debug_req())
    assert res.log_summary.failed == 1
    assert res.root_causes[0].startswith("[RTL]")
    assert res.suggested_fixes
    assert res.corrected_code_target == "RTL"
    assert "module counter" in res.corrected_code
    assert any("hypotheses" in w for w in res.warnings)


def test_sim_debugger_log_facts_are_sent_to_llm():
    seen = {}

    def fn(prompt):
        seen["prompt"] = prompt
        return DEBUG_RESPONSE

    SimulationDebugger(FakeLLMClient(response_fn=fn)).analyze(_debug_req())
    assert "tests failed (from [FAIL] lines): 1" in seen["prompt"]
    assert "wrap: expected=0 observed=256" in seen["prompt"]


def test_sim_debugger_no_patch_when_none():
    resp = DEBUG_RESPONSE.replace("CORRECTED_CODE_TARGET:\nRTL", "CORRECTED_CODE_TARGET:\nNone").replace(
        "```systemverilog\nmodule counter; endmodule\n```", "None."
    )
    res = SimulationDebugger(FakeLLMClient(fixed_response=resp)).analyze(_debug_req())
    assert res.corrected_code == "" and res.corrected_code_target == ""


def test_sim_debugger_bad_patch_does_not_discard_analysis():
    resp = DEBUG_RESPONSE.replace("```systemverilog\nmodule counter; endmodule\n```", "see above")
    res = SimulationDebugger(FakeLLMClient(fixed_response=resp)).analyze(_debug_req())
    assert res.corrected_code == ""
    assert res.suggested_fixes


def test_sim_debugger_warns_when_log_has_no_evidence():
    res = SimulationDebugger(FakeLLMClient(fixed_response=DEBUG_RESPONSE)).analyze(_debug_req("sim done"))
    assert any("limited evidence" in w for w in res.warnings)


def test_sim_debugger_rejects_empty_log():
    with pytest.raises(ValueError):
        SimulationDebugger(FakeLLMClient(fixed_response=DEBUG_RESPONSE)).analyze(_debug_req(" "))


def test_sim_debugger_trims_huge_logs():
    seen = {}

    def fn(prompt):
        seen["n"] = len(prompt)
        return DEBUG_RESPONSE

    SimulationDebugger(FakeLLMClient(response_fn=fn)).analyze(_debug_req("[PASS] x\n" * 20000))
    assert seen["n"] < 40000
