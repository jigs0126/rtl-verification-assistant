"""
End-to-End Integration Tests

Unlike the per-module unit tests (test_loader.py, test_chunking.py,
test_embedder.py, test_vector_store.py, test_retriever.py,
test_prompts.py, test_response_parser.py, test_testbench_generator.py,
test_debug_analyzer.py — each of which tests one layer in isolation,
often with a fake for its neighbor), these tests run the REAL pipeline
end-to-end against the real ALU project files:

    corpus -> ingestion -> embeddings -> ChromaDB -> query -> retrieval
    -> prompt construction -> (LLM) -> structured result

The only faked component is the LLM call itself (FakeLLMClient) — this
sandbox has no LLM API key (see README "Limitations"). Everything
before that point (loading real files, parsing real RTL, chunking,
embedding with whichever EmbeddingModel get_embedding_model() resolves
to, persisting to a real temporary ChromaDB, and querying it back) is
the genuine implementation, not a mock.
"""

from __future__ import annotations

from pathlib import Path

from app.embeddings.embedder import DeterministicHashingEmbeddingModel
from app.llm.client import FakeLLMClient
from app.retrieval.indexer import index_project
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore
from app.schemas.verification import DebugRequest, TestbenchRequest
from app.verification.debug_analyzer import DebugAnalyzer
from app.verification.testbench_generator import TestbenchGenerator

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_FAKE_TESTBENCH_RESPONSE = """\
TEST_SCENARIOS:
1. SUB with a > b
2. SUB with a == b
3. SUB with a < b (underflow)

ASSUMPTIONS:
1. alu_control is driven per docs/operation_table.md encoding.

WARNINGS:
1. Not simulation-verified.

TESTBENCH_CODE:
```systemverilog
module tb_alu;
  logic [31:0] a, b, result;
  logic [2:0] alu_control;
  alu dut(.a(a), .b(b), .alu_control(alu_control), .result(result));
  initial begin
    a = 20; b = 10; alu_control = 3'b001; #1;
    $display("result=%0d", result);
  end
endmodule
```"""

_FAKE_DEBUG_RESPONSE = """\
FAILURE_SUMMARY:
SUB operation produced an unexpected large unsigned result.

EXPECTED_BEHAVIOR:
Per docs/alu_spec.md, a=20 b=10 SUB should yield 10.

OBSERVED_BEHAVIOR:
Result was 4294967286 (0xFFFFFFF6), i.e. -10 under two's complement.

INTERPRETATION:
The observed value matches -10, consistent with a computing b - a rather than a - b.

RELEVANT_MODULES:
alu

RELEVANT_SIGNALS:
a, b, alu_control, result

POSSIBLE_ROOT_CAUSES:
1. [HYPOTHESIS] Operand ordering swapped at the testbench-to-DUT connection.
2. [HYPOTHESIS] alu_control decoding selects the wrong operation.

EVIDENCE:
1. [FACT] 4294967286 equals 0xFFFFFFF6, the two's-complement encoding of -10.
2. [FACT] -10 equals b - a for the supplied a=20, b=10.

RECOMMENDED_INVESTIGATION:
1. Inspect the testbench DUT instantiation for swapped .a/.b connections.
2. Confirm alu_control decodes to the SUB encoding in operation_table.md.

CONFIDENCE:
medium - the arithmetic match to b - a is a strong lexical clue but not a captured waveform."""


def _build_retriever(tmp_path: Path) -> Retriever:
    """Real ingest -> embed -> persist -> retriever, against the actual
    ALU project files on disk, using a fresh temp vector store per test
    (so tests never interfere with each other's index)."""
    model = DeterministicHashingEmbeddingModel(dimension=128)
    store = VectorStore(path=tmp_path / "vector_store")
    count = index_project(PROJECT_ROOT, store, model, project_name="alu", chunk_size=500, chunk_overlap=80)
    assert count > 0
    return Retriever(store, model, default_top_k=4)


# ------------------------------------------------------- ingestion path -


def test_full_ingestion_to_chromadb_indexes_real_corpus(tmp_path):
    model = DeterministicHashingEmbeddingModel(dimension=64)
    store = VectorStore(path=tmp_path / "vs")
    count = index_project(PROJECT_ROOT, store, model, project_name="alu")

    assert count == 33  # matches the corpus size established in Phase 2/3
    assert store.count() == 33
    assert store.collection_metadata()["model_name"] == model.model_name


def test_reindexing_is_idempotent_no_duplicates(tmp_path):
    model = DeterministicHashingEmbeddingModel(dimension=64)
    store = VectorStore(path=tmp_path / "vs")
    index_project(PROJECT_ROOT, store, model, project_name="alu")
    index_project(PROJECT_ROOT, store, model, project_name="alu")  # run twice
    assert store.count() == 33  # upsert by chunk_id, not duplicated


# --------------------------------------------------------- retrieval path


def test_retrieval_returns_relevant_chunks_for_sub_query(tmp_path):
    retriever = _build_retriever(tmp_path)
    context = retriever.query("How is subtraction implemented in the ALU?", top_k=5)

    assert not context.is_empty
    sources = {r.chunk.source for r in context.results}
    # At minimum the RTL itself and the spec should be reachable via this query.
    assert "alu.sv" in sources or "alu_spec.md" in sources or "operation_table.md" in sources


def test_retrieval_results_are_ranked_nearest_first(tmp_path):
    retriever = _build_retriever(tmp_path)
    context = retriever.query("ALU AND operation bitwise", top_k=5)
    ranks = [r.rank for r in context.results]
    assert ranks == sorted(ranks)


# ------------------------------------------------------ prompt assembly -


def test_prompt_construction_includes_retrieved_context(tmp_path):
    from app.llm.prompts import build_testbench_prompt

    retriever = _build_retriever(tmp_path)
    context = retriever.query("SUB operation testbench", top_k=3)
    system_prompt, user_prompt = build_testbench_prompt("alu", "cover all SUB corner cases", context)

    assert "alu" in system_prompt
    assert "PROJECT CONTEXT" in user_prompt
    assert "USER REQUEST" in user_prompt
    for result in context.results:
        assert result.chunk.source in user_prompt


# ------------------------------------------------- testbench end-to-end -


def test_testbench_generator_end_to_end_with_fake_llm(tmp_path):
    retriever = _build_retriever(tmp_path)
    llm = FakeLLMClient(fixed_response=_FAKE_TESTBENCH_RESPONSE)
    generator = TestbenchGenerator(retriever, llm, top_k=4, log_path=tmp_path / "logs")

    result = generator.generate(TestbenchRequest(module="alu", requirement="Cover all SUB corner cases"))

    assert "module tb_alu" in result.testbench_code
    assert len(result.test_scenarios) == 3
    assert "Generated artifact — not simulation-verified." in result.warnings
    assert len(result.retrieved_sources) > 0

    # The query was logged.
    from app.utils.logging import read_recent_queries
    records = read_recent_queries(tmp_path / "logs")
    assert records[-1]["mode"] == "testbench"
    assert records[-1]["status"] == "success"


# ----------------------------------------------------- debug end-to-end -


def test_debug_analyzer_end_to_end_with_fake_llm(tmp_path):
    retriever = _build_retriever(tmp_path)
    llm = FakeLLMClient(fixed_response=_FAKE_DEBUG_RESPONSE)
    analyzer = DebugAnalyzer(retriever, llm, top_k=4, log_path=tmp_path / "logs")

    failure_report = (
        "Test: alu_sub_01\nOperation: SUB\nA=20\nB=10\nExpected=10\nObserved=4294967286 (0xFFFFFFF6)"
    )
    result = analyzer.analyze(DebugRequest(module="alu", failure_report=failure_report))

    assert "SUB" in result.failure_summary or "sub" in result.failure_summary.lower()
    assert len(result.possible_root_causes) == 2
    assert all("[HYPOTHESIS]" in c for c in result.possible_root_causes)
    assert len(result.evidence) == 2
    assert all("[FACT]" in e for e in result.evidence)
    assert result.confidence.startswith("medium")
    assert len(result.retrieved_sources) > 0


def test_debug_analyzer_rejects_empty_failure_report(tmp_path):
    retriever = _build_retriever(tmp_path)
    llm = FakeLLMClient(fixed_response=_FAKE_DEBUG_RESPONSE)
    analyzer = DebugAnalyzer(retriever, llm)

    import pytest
    with pytest.raises(ValueError):
        analyzer.analyze(DebugRequest(module="alu", failure_report="   "))
