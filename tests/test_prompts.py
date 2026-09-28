from app.llm.prompts import build_debug_prompt, build_testbench_prompt
from app.schemas.models import Chunk, RetrievedChunk, RetrievedContext


def _context(with_results: bool = True) -> RetrievedContext:
    if not with_results:
        return RetrievedContext(query="a query", results=[], top_k=4)
    chunk = Chunk(
        chunk_id="alu.sv::always_block[alu_control]::0::abc",
        text="case (alu_control) ALU_SUB: result = a - b; endcase",
        source="alu.sv", file_type="systemverilog", project="alu",
        section="always_block[alu_control]", module="alu",
    )
    return RetrievedContext(
        query="a query", top_k=4,
        results=[RetrievedChunk(chunk=chunk, score=0.12, rank=1)],
    )


def test_testbench_prompt_contains_module_and_requirement():
    system, user = build_testbench_prompt("alu", "cover all operations", _context())
    assert "alu" in system
    assert "cover all operations" in user


def test_testbench_prompt_includes_retrieved_source_and_text():
    system, user = build_testbench_prompt("alu", "cover SUB", _context())
    assert "alu.sv" in user
    assert "ALU_SUB: result = a - b" in user


def test_testbench_prompt_states_not_invent_ports_rule():
    system, _ = build_testbench_prompt("alu", "req", _context())
    assert "do not invent ports" in system.lower()


def test_testbench_prompt_defines_output_format_sections():
    system, _ = build_testbench_prompt("alu", "req", _context())
    for label in ["TEST_SCENARIOS:", "ASSUMPTIONS:", "WARNINGS:", "TESTBENCH_CODE:"]:
        assert label in system


def test_testbench_prompt_handles_empty_context_explicitly():
    _, user = build_testbench_prompt("alu", "req", _context(with_results=False))
    assert "No relevant project context was retrieved" in user


def test_debug_prompt_contains_module_and_failure_report():
    system, user = build_debug_prompt("alu", "SUB gave wrong result", _context())
    assert "alu" in system
    assert "SUB gave wrong result" in user


def test_debug_prompt_requires_fact_hypothesis_unknown_labeling():
    system, _ = build_debug_prompt("alu", "report", _context())
    assert "FACT" in system
    assert "HYPOTHESIS" in system
    assert "UNKNOWN" in system


def test_debug_prompt_forbids_confirming_bugs_without_evidence():
    system, _ = build_debug_prompt("alu", "report", _context())
    assert "confirmed" in system.lower() or "confirm" in system.lower()


def test_debug_prompt_defines_output_format_sections():
    system, _ = build_debug_prompt("alu", "report", _context())
    for label in [
        "FAILURE_SUMMARY:", "EXPECTED_BEHAVIOR:", "OBSERVED_BEHAVIOR:",
        "POSSIBLE_ROOT_CAUSES:", "EVIDENCE:", "CONFIDENCE:",
    ]:
        assert label in system
