"""
Streamlit UI

What it is: the human-in-the-loop front end — three tabs (Testbench
Generator, Debug Assistant, Knowledge Base) over the TestbenchGenerator,
DebugAnalyzer, and Retriever built in app/context.py.

Why required: this is what makes the RAG pipeline inspectable rather
than a black box — per §10/§19-22, retrieved chunks (with their
distances) must always be visible, indexing must be an explicit action
(never silent), and every generated artifact must carry its
not-simulation-verified warning and its retrieved sources.

Input: user interaction (module/requirement text, failure report text,
button clicks) via Streamlit widgets.
Output: rendered UI — retrieved context, generated testbench / debug
analysis, knowledge base status.

How it connects: calls app.context.build_app_context() once per session
(cached), then AppContext.testbench_generator.generate(),
.debug_analyzer.analyze(), and .retriever.query() directly.

What would happen if it were removed: there would be no way to actually
use the system interactively — it would only be usable via Python calls
in tests/scripts, with no way to see retrieved chunks or generated
artifacts without writing code.

Run with: streamlit run app/main.py  (or `python run.py`)
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.config.settings import Settings
from app.context import AppError, build_app_context
from app.llm.client import LLMError
from app.retrieval.indexer import index_project
from app.schemas.verification import DebugRequest, TestbenchRequest
from app.schemas.workflow import PlanTestbenchRequest, RTLGenRequest, SimDebugRequest

st.set_page_config(page_title="RAG-Based RTL Verification Assistant", layout="wide")


@st.cache_resource
def _get_context():
    settings = Settings()
    return settings, build_app_context(settings)


def _render_retrieved_context(context) -> None:
    """Shared rendering for retrieved chunks — used by all three tabs.
    This is the "retrieval inspection" view §19/§22 both require: every
    retrieved chunk with its source, section, and similarity distance,
    so it's visible that the system actually retrieved project-specific
    context rather than sending the raw prompt straight to the LLM."""
    if context is None or context.is_empty:
        st.warning(
            "No relevant project context was retrieved. The assistant will not "
            "invent project-specific details. Add relevant RTL/specification to "
            "the knowledge base or refine the query."
        )
        return
    st.caption(f"Retrieved {len(context.results)} chunk(s) for query: \"{context.query}\"")
    for result in context.results:
        chunk = result.chunk
        lang = "systemverilog" if chunk.file_type in ("systemverilog", "verilog") else "text"
        with st.expander(f"#{result.rank}  {chunk.source} — {chunk.section}  (distance: {result.score:.4f})"):
            st.code(chunk.text, language=lang)


def main() -> None:
    st.title("RAG-Based RTL Verification Assistant")
    st.caption(
        "Human-in-the-loop. Generated testbenches are not simulation-verified. "
        "Debug findings are hypotheses unless explicitly marked as facts."
    )

    try:
        settings, ctx = _get_context()
    except AppError as exc:
        st.error(str(exc))
        st.stop()
        return

    with st.sidebar:
        st.header("Project")
        st.text(f"Documents path: {settings.documents_path}")
        st.text(f"Indexed chunks: {ctx.vector_store.count()}")
        if ctx.using_fallback_embeddings:
            st.warning(
                "Using the deterministic hashing fallback embedding model "
                "(sentence-transformers not available). Retrieval quality is "
                "lexical, not semantic — see README Limitations."
            )
        if ctx.using_fake_llm:
            st.warning("No LLM API key configured — generation will show a clear error, not fake output.")

        st.header("RAG Settings")
        top_k = st.slider("top_k", min_value=1, max_value=10, value=settings.top_k)

        st.header("Indexing")
        if st.button("Re-index Knowledge Base"):
            with st.spinner("Indexing project..."):
                count = index_project(
                    Path(settings.documents_path),
                    ctx.vector_store,
                    ctx.embedding_model,
                    project_name="alu",
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                )
            st.success(f"Indexed {count} chunk(s).")
            st.cache_resource.clear()

    tab_workflow, tab_testbench, tab_debug, tab_kb = st.tabs(
        ["RTL Workflow", "Testbench Generator", "Debug Assistant", "Knowledge Base"]
    )

    with tab_workflow:
        _render_workflow_tab(ctx)

    with tab_testbench:
        _render_testbench_tab(ctx, top_k)

    with tab_debug:
        _render_debug_tab(ctx, top_k)

    with tab_kb:
        _render_knowledge_base_tab(ctx, settings)


def _render_workflow_tab(ctx) -> None:
    """Spec -> RTL -> (plan) -> testbench -> [user simulates] -> log -> fixes.

    Each step's output lives in st.session_state so the next step can use
    it, and the user can still edit the RTL / testbench text areas before
    moving on (e.g. after applying a suggested fix and re-simulating).
    """
    st.subheader("RTL Workflow")
    st.caption(
        "1) spec \u2192 RTL   2) verification plan \u2192 testbench   "
        "3) run it in your own simulator, paste the console log \u2192 debug + fixes. "
        "Nothing here is compiled or simulated by this app."
    )

    # ---- Step 1: spec -> RTL ------------------------------------------------
    st.markdown("### Step 1 \u2014 Specification \u2192 RTL")
    spec = st.text_area(
        "Specification / example",
        height=180,
        placeholder="e.g. 8-bit up/down counter with synchronous active-high reset, enable, and load...",
        key="wf_spec",
    )
    module_name = st.text_input("Module name (optional)", key="wf_module_name")

    if st.button("Generate RTL", key="wf_gen_rtl"):
        try:
            with st.spinner("Generating RTL..."):
                res = ctx.rtl_generator.generate(RTLGenRequest(specification=spec, module_name=module_name))
        except (LLMError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.session_state["wf_rtl_result"] = res
            st.session_state["wf_rtl"] = res.rtl_code  # editable copy for later steps
            st.session_state.pop("wf_tb_result", None)
            st.session_state.pop("wf_debug_result", None)

    rtl_res = st.session_state.get("wf_rtl_result")
    if rtl_res:
        st.markdown("**Interface**")
        for item in rtl_res.interface_summary:
            st.markdown(f"- {item}")
        if rtl_res.design_decisions:
            st.markdown("**Design decisions**")
            for item in rtl_res.design_decisions:
                st.markdown(f"- {item}")
        if rtl_res.assumptions:
            st.markdown("**Assumptions**")
            for item in rtl_res.assumptions:
                st.markdown(f"- {item}")
        for w in rtl_res.warnings:
            st.markdown(f"- \u26a0\ufe0f {w}")

    # ---- Step 2: plan -> testbench ------------------------------------------
    st.divider()
    st.markdown("### Step 2 \u2014 Verification plan \u2192 testbench")
    st.text_area(
        "RTL under test (generated above, or paste your own)",
        height=250,
        key="wf_rtl",
    )
    plan = st.text_area(
        "Verification plan",
        height=180,
        placeholder="List the scenarios/corner cases to verify, e.g. reset behaviour, load, wraparound...",
        key="wf_plan",
    )

    if st.button("Generate Testbench", key="wf_gen_tb"):
        try:
            with st.spinner("Generating testbench..."):
                res = ctx.plan_tb_generator.generate(
                    PlanTestbenchRequest(
                        rtl_code=st.session_state.get("wf_rtl", ""),
                        verification_plan=plan,
                        module_name=module_name,
                    )
                )
        except (LLMError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.session_state["wf_tb_result"] = res
            st.session_state["wf_tb"] = res.testbench_code
            st.session_state.pop("wf_debug_result", None)

    tb_res = st.session_state.get("wf_tb_result")
    if tb_res:
        st.markdown("**Test scenarios**")
        for item in tb_res.test_scenarios:
            st.markdown(f"- {item}")
        st.markdown("**Verification-plan coverage**")
        for item in tb_res.plan_coverage:
            st.markdown(f"- {item}")
        if tb_res.assumptions:
            st.markdown("**Assumptions**")
            for item in tb_res.assumptions:
                st.markdown(f"- {item}")
        for w in tb_res.warnings:
            st.markdown(f"- \u26a0\ufe0f {w}")

    # ---- Step 3: simulator log -> debug + fixes -----------------------------
    st.divider()
    st.markdown("### Step 3 \u2014 Simulator console \u2192 debug and fixes")
    st.text_area(
        "Testbench (generated above, or paste the one you simulated)",
        height=250,
        key="wf_tb",
    )
    st.download_button(
        "Download RTL (.sv)", data=st.session_state.get("wf_rtl", ""), file_name="design.sv",
        disabled=not st.session_state.get("wf_rtl"), key="wf_dl_rtl",
    )
    st.download_button(
        "Download testbench (.sv)", data=st.session_state.get("wf_tb", ""), file_name="tb.sv",
        disabled=not st.session_state.get("wf_tb"), key="wf_dl_tb",
    )
    log = st.text_area(
        "Simulator console output (pass or fail \u2014 paste the whole log)",
        height=250,
        placeholder="# [PASS] add_basic\n# [FAIL] sub_underflow: expected=... observed=...\n# SUMMARY: 9 passed, 1 failed",
        key="wf_log",
    )

    if st.button("Debug Simulation", key="wf_debug"):
        try:
            with st.spinner("Analyzing simulation log..."):
                res = ctx.sim_debugger.analyze(
                    SimDebugRequest(
                        rtl_code=st.session_state.get("wf_rtl", ""),
                        testbench_code=st.session_state.get("wf_tb", ""),
                        console_log=log,
                        verification_plan=plan,
                    )
                )
        except (LLMError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.session_state["wf_debug_result"] = res

    dbg = st.session_state.get("wf_debug_result")
    if dbg:
        ls = dbg.log_summary
        c1, c2, c3 = st.columns(3)
        c1.metric("Passed (from log)", ls.passed)
        c2.metric("Failed (from log)", ls.failed)
        c3.metric("Error lines", len(ls.error_lines))
        if ls.all_passed:
            st.success("The log shows all tests passing.")
        for line in ls.error_lines:
            st.code(line)

        st.markdown("**Summary**")
        st.write(dbg.summary)

        st.markdown("**Possible root causes (hypotheses \u2014 unconfirmed)**")
        for item in dbg.root_causes:
            st.markdown(f"- {item}")

        st.markdown("**Suggested fixes**")
        for item in dbg.suggested_fixes:
            st.markdown(f"- {item}")

        if dbg.corrected_code:
            st.markdown(f"**Proposed corrected {dbg.corrected_code_target}** (review before use)")
            st.code(dbg.corrected_code, language="systemverilog")
            st.caption(
                "To try it: copy it into the matching box above, re-run your simulator, "
                "and paste the new log here."
            )

        st.markdown("**Confidence**")
        st.write(dbg.confidence)
        for w in dbg.warnings:
            st.markdown(f"- \u26a0\ufe0f {w}")


def _render_testbench_tab(ctx, top_k: int) -> None:
    st.subheader("Testbench Generator")
    module = st.text_input("Module", value="alu", key="tb_module")
    requirement = st.text_area(
        "Requirement",
        placeholder=(
            "Generate a SystemVerilog testbench for the ALU covering all "
            "operations and important boundary cases."
        ),
        key="tb_requirement",
    )

    if st.button("Generate Testbench"):
        if not requirement or not requirement.strip():
            st.error("Requirement must not be empty.")
            return

        context = None
        try:
            with st.spinner("Retrieving context and generating..."):
                context = ctx.retriever.query(f"{module}: {requirement}", top_k=top_k)
                request = TestbenchRequest(module=module, requirement=requirement)
                result = ctx.testbench_generator.generate(request)
        except LLMError as exc:
            st.error(f"LLM request failed: {exc}")
            st.subheader("Retrieved Context")
            _render_retrieved_context(context)
            return
        except ValueError as exc:
            st.error(str(exc))
            return

        st.subheader("Retrieved Context")
        _render_retrieved_context(context)

        st.subheader("Generated Testbench")
        st.code(result.testbench_code, language="systemverilog")

        st.subheader("Test Scenarios")
        for s in result.test_scenarios:
            st.markdown(f"- {s}")

        st.subheader("Assumptions")
        for a in result.assumptions:
            st.markdown(f"- {a}")

        st.subheader("Warnings")
        for w in result.warnings:
            st.markdown(f"- \u26a0\ufe0f {w}")

        st.caption(f"Sources used: {', '.join(result.retrieved_sources) or 'none'}")


def _render_debug_tab(ctx, top_k: int) -> None:
    st.subheader("Debug Assistant")
    module = st.text_input("Module", value="alu", key="debug_module")
    failure_report = st.text_area(
        "Failure report",
        height=200,
        placeholder="Test: alu_sub_01\nOperation: SUB\nA=20\nB=10\nExpected=10\nObserved=4294967286",
        key="debug_report",
    )

    if st.button("Analyze Failure"):
        if not failure_report or not failure_report.strip():
            st.error("Failure report must not be empty.")
            return

        context = None
        try:
            with st.spinner("Retrieving context and analyzing..."):
                context = ctx.retriever.query(f"{module}: {failure_report}", top_k=top_k)
                request = DebugRequest(module=module, failure_report=failure_report)
                result = ctx.debug_analyzer.analyze(request)
        except LLMError as exc:
            st.error(f"LLM request failed: {exc}")
            st.subheader("Retrieved Context")
            _render_retrieved_context(context)
            return
        except ValueError as exc:
            st.error(str(exc))
            return

        st.subheader("Retrieved Context")
        _render_retrieved_context(context)

        st.subheader("Failure Summary")
        st.write(result.failure_summary)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Expected Behavior")
            st.write(result.expected_behavior)
        with col2:
            st.subheader("Observed Behavior")
            st.write(result.observed_behavior)

        st.subheader("Interpretation")
        st.write(result.interpretation)

        st.subheader("Relevant RTL")
        st.markdown(f"**Modules:** {', '.join(result.relevant_modules) or 'none identified'}")
        st.markdown(f"**Signals:** {', '.join(result.relevant_signals) or 'none identified'}")

        st.subheader("Possible Root Causes (hypotheses — unconfirmed)")
        for c in result.possible_root_causes:
            st.markdown(f"- {c}")

        st.subheader("Evidence (facts)")
        for e in result.evidence:
            st.markdown(f"- {e}")

        st.subheader("Recommended Investigation")
        for step in result.recommended_investigation:
            st.markdown(f"- {step}")

        st.subheader("Confidence")
        st.write(result.confidence)

        st.caption(f"Sources used: {', '.join(result.retrieved_sources) or 'none'}")


def _render_knowledge_base_tab(ctx, settings) -> None:
    st.subheader("Knowledge Base")
    ids = ctx.vector_store.list_ids()
    st.metric("Indexed chunks", len(ids))

    by_source: dict[str, int] = {}
    for chunk_id in ids:
        source = chunk_id.split("::")[0]
        by_source[source] = by_source.get(source, 0) + 1

    if by_source:
        st.table({"file": list(by_source.keys()), "chunks": list(by_source.values())})
    else:
        st.info("Knowledge base is empty. Use 'Re-index Knowledge Base' in the sidebar.")

    st.subheader("Try a retrieval query")
    query = st.text_input("Query", placeholder="How is subtraction implemented in the ALU?")
    if st.button("Search"):
        if not query or not query.strip():
            st.error("Query must not be empty.")
        else:
            try:
                context = ctx.retriever.query(query, top_k=5)
                _render_retrieved_context(context)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Retrieval failed: {exc}")

    st.subheader("Recent query log")
    from app.utils.logging import read_recent_queries

    records = read_recent_queries(settings.log_path, limit=10)
    if records:
        st.table(records)
    else:
        st.info("No queries logged yet.")


if __name__ == "__main__":
    main()
