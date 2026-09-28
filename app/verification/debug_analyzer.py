"""
Debug Analyzer

What it is: the application-level orchestrator for capability #2 (RTL
debug/failure analysis). Ties together retrieval, prompt construction,
the LLM call, and response parsing into one
analyze(DebugRequest) -> DebugResult call.

Why required: this is where a supplied failure report becomes a
structured investigation — the concrete implementation of §15/§16's
FACT/HYPOTHESIS/UNKNOWN debugging flow.

Input: DebugRequest (module, failure_report).
Output: DebugResult (failure_summary, expected/observed_behavior,
interpretation, relevant_modules/signals, possible_root_causes,
evidence, recommended_investigation, confidence, retrieved_sources).

How it connects: Retriever (Phase 6) -> prompts.build_debug_prompt
(Phase 7) -> LLMClient.complete (Phase 7) -> response_parser (this
phase) -> DebugResult, which the Streamlit UI (Phase 10) renders with
possible_root_causes and evidence shown as visually distinct sections
(never merged into one list) per §16.

What would happen if it were removed: there would be no single place
implementing "failure report -> structured debug analysis" — and
critically, no single place enforcing that hypotheses and facts stay in
separate fields of the returned schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.llm.client import LLMClient
from app.llm.prompts import build_debug_prompt
from app.llm.response_parser import parse_comma_list, parse_labeled_sections, parse_numbered_list
from app.retrieval.retriever import Retriever
from app.schemas.verification import DebugRequest, DebugResult
from app.utils.logging import log_query

_REQUIRED_SECTIONS = [
    "FAILURE_SUMMARY",
    "EXPECTED_BEHAVIOR",
    "OBSERVED_BEHAVIOR",
    "INTERPRETATION",
    "RELEVANT_MODULES",
    "RELEVANT_SIGNALS",
    "POSSIBLE_ROOT_CAUSES",
    "EVIDENCE",
    "RECOMMENDED_INVESTIGATION",
    "CONFIDENCE",
]


class DebugAnalyzer:
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

    def analyze(self, request: DebugRequest) -> DebugResult:
        if not request.failure_report or not request.failure_report.strip():
            raise ValueError("failure_report must not be empty")

        # Retrieve using the failure report itself as the query — its
        # operation name, module, and symptom description are what
        # should surface relevant RTL/spec/historical-bug chunks.
        query = f"{request.module}: {request.failure_report}"
        context = self._retriever.query(query, top_k=self._top_k)

        try:
            system_prompt, user_prompt = build_debug_prompt(request.module, request.failure_report, context)
            response_text = self._llm.complete(user_prompt, system=system_prompt, max_tokens=3000)

            sections = parse_labeled_sections(response_text, _REQUIRED_SECTIONS)
            retrieved_sources = sorted({r.chunk.source for r in context.results})

            result = DebugResult(
                failure_summary=sections["FAILURE_SUMMARY"],
                expected_behavior=sections["EXPECTED_BEHAVIOR"],
                observed_behavior=sections["OBSERVED_BEHAVIOR"],
                interpretation=sections["INTERPRETATION"],
                relevant_modules=parse_comma_list(sections["RELEVANT_MODULES"]),
                relevant_signals=parse_comma_list(sections["RELEVANT_SIGNALS"]),
                possible_root_causes=parse_numbered_list(sections["POSSIBLE_ROOT_CAUSES"]),
                evidence=parse_numbered_list(sections["EVIDENCE"]),
                recommended_investigation=parse_numbered_list(sections["RECOMMENDED_INVESTIGATION"]),
                confidence=sections["CONFIDENCE"],
                retrieved_sources=retrieved_sources,
            )
        except Exception as exc:
            if self._log_path is not None:
                model_name = getattr(self._llm, "_model", "unknown")
                log_query(self._log_path, "debug", query, context, str(model_name), f"error: {exc}")
            raise

        if self._log_path is not None:
            model_name = getattr(self._llm, "_model", "unknown")
            log_query(self._log_path, "debug", query, context, str(model_name), "success")

        return result
