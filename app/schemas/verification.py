"""
Verification Schemas

What they are: typed request/result shapes for the two application-level
capabilities — testbench generation and debug analysis.

Why required: an LLM's raw text response is not something the UI should
render directly, or something a malformed response should be allowed to
silently corrupt. Parsing the LLM's structured-text response into these
Pydantic models means a genuinely malformed response fails loudly (a
validation error) instead of quietly displaying garbage (§23's
"malformed LLM response" error-handling requirement).

Input: these are populated by verification/testbench_generator.py and
verification/debug_analyzer.py after parsing an LLMClient response.
Output: consumed by the Streamlit UI (Phase 10) to render each field in
its own section.

How they connect: TestbenchRequest/DebugRequest go in; the corresponding
Result schema comes out, carrying retrieved_sources so the UI can always
show "what was this grounded in?" alongside the generated artifact.
"""

from __future__ import annotations

from pydantic import BaseModel



class TestbenchRequest(BaseModel):
    """Input to TestbenchGenerator.generate()."""

    module: str
    requirement: str


class TestbenchResult(BaseModel):
    """
    Structured output of testbench generation.

    IMPORTANT: testbench_code being present here means the LLM produced
    SystemVerilog text — it does NOT mean the testbench has been
    simulated, compiled, or verified in any way. The UI must always
    display this alongside the "Generated artifact — not
    simulation-verified" warning (see §13), regardless of what
    `warnings` itself contains.
    """

    testbench_code: str
    test_scenarios: list[str]
    assumptions: list[str]
    warnings: list[str]
    retrieved_sources: list[str]


class DebugRequest(BaseModel):
    """Input to DebugAnalyzer.analyze()."""

    module: str
    failure_report: str


class DebugResult(BaseModel):
    """
    Structured output of debug analysis.

    IMPORTANT: possible_root_causes are hypotheses, not conclusions —
    see §16/§24. This schema deliberately keeps `evidence` (facts
    supporting the interpretation) and `possible_root_causes`
    (unconfirmed explanations) as separate fields rather than one mixed
    list, so the UI cannot accidentally present a hypothesis as a fact
    just by rendering fields in the same style.
    """

    failure_summary: str
    expected_behavior: str
    observed_behavior: str
    interpretation: str
    relevant_modules: list[str]
    relevant_signals: list[str]
    possible_root_causes: list[str]
    evidence: list[str]
    recommended_investigation: list[str]
    confidence: str
    retrieved_sources: list[str]
