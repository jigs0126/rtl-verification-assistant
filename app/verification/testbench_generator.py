"""
Testbench Generator

What it is: the application-level orchestrator for capability #1
(SystemVerilog testbench generation). Ties together retrieval, prompt
construction, the LLM call, and response parsing into one
generate(TestbenchRequest) -> TestbenchResult call.

Why required: this is where "natural-language requirement" actually
becomes "grounded SystemVerilog testbench" — the concrete implementation
of the flow described in §13 of the project spec.

Input: TestbenchRequest (module, requirement).
Output: TestbenchResult (testbench_code, test_scenarios, assumptions,
warnings, retrieved_sources).

How it connects: Retriever (Phase 6) -> prompts.build_testbench_prompt
(Phase 7) -> LLMClient.complete (Phase 7) ->
response_parser (this phase) -> TestbenchResult, which the Streamlit UI
(Phase 10) renders.

What would happen if it were removed: there would be no single place
implementing "requirement -> testbench" — the UI would have to call
retrieval/prompts/LLM/parsing directly itself, duplicating this logic if
a second entry point (e.g. a CLI) were ever added.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.llm.client import LLMClient
from app.llm.prompts import build_testbench_prompt
from app.llm.response_parser import extract_code_block, parse_labeled_sections, parse_numbered_list
from app.retrieval.retriever import Retriever
from app.schemas.verification import TestbenchRequest, TestbenchResult
from app.utils.logging import log_query

_REQUIRED_SECTIONS = ["TEST_SCENARIOS", "ASSUMPTIONS", "WARNINGS", "TESTBENCH_CODE"]

# Always appended, regardless of what the LLM itself said in WARNINGS —
# per §13, the UI/result must never omit this regardless of model output.
_NOT_VERIFIED_WARNING = "Generated artifact — not simulation-verified."


class TestbenchGenerator:
    def __init__(
        self,
        retriever: Retriever,
        llm_client: LLMClient,
        top_k: int = 5,
        log_path: Optional[Path] = None,
    ):
        self._retriever = retriever
        self._llm = llm_client
        self._top_k = top_k
        self._log_path = log_path

    def generate(self, request: TestbenchRequest) -> TestbenchResult:
        if not request.requirement or not request.requirement.strip():
            raise ValueError("requirement must not be empty")

        query = f"{request.module}: {request.requirement}"
        context = self._retriever.query(query, top_k=self._top_k)

        try:
            system_prompt, user_prompt = build_testbench_prompt(request.module, request.requirement, context)
            response_text = self._llm.complete(user_prompt, system=system_prompt, max_tokens=3000)

            sections = parse_labeled_sections(response_text, _REQUIRED_SECTIONS)

            testbench_code = extract_code_block(sections["TESTBENCH_CODE"])
            test_scenarios = parse_numbered_list(sections["TEST_SCENARIOS"])
            assumptions = parse_numbered_list(sections["ASSUMPTIONS"])
            warnings = parse_numbered_list(sections["WARNINGS"])

            if _NOT_VERIFIED_WARNING not in warnings:
                warnings.append(_NOT_VERIFIED_WARNING)

            retrieved_sources = sorted({r.chunk.source for r in context.results})

            result = TestbenchResult(
                testbench_code=testbench_code,
                test_scenarios=test_scenarios,
                assumptions=assumptions,
                warnings=warnings,
                retrieved_sources=retrieved_sources,
            )
        except Exception as exc:
            if self._log_path is not None:
                model_name = getattr(self._llm, "_model", "unknown")
                log_query(self._log_path, "testbench", query, context, str(model_name), f"error: {exc}")
            raise

        if self._log_path is not None:
            model_name = getattr(self._llm, "_model", "unknown")
            log_query(self._log_path, "testbench", query, context, str(model_name), "success")

        return result
