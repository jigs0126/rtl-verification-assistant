import pytest

from app.embeddings.embedder import DeterministicHashingEmbeddingModel, embed_chunks
from app.llm.client import FakeLLMClient, LLMError
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore
from app.schemas.models import Chunk
from app.schemas.verification import DebugRequest
from app.verification.debug_analyzer import DebugAnalyzer

FAKE_RESPONSE = """\
FAILURE_SUMMARY:
SUB operation produced a result inconsistent with A - B for A=20, B=10.

EXPECTED_BEHAVIOR:
Per alu_spec.md, SUB should compute a - b, giving 10.

OBSERVED_BEHAVIOR:
Result was 4294967286 (0xFFFFFFF6), which is -10 under 32-bit two's complement.

INTERPRETATION:
The observed value equals B - A rather than A - B, which is consistent \
with an operand-ordering issue, but this is not confirmed by the \
supplied evidence alone.

RELEVANT_MODULES:
alu

RELEVANT_SIGNALS:
a, b, alu_control, result

POSSIBLE_ROOT_CAUSES:
1. [HYPOTHESIS] Operand ordering swapped at the ALU input connections
2. [HYPOTHESIS] Testbench driver applied a/b in the wrong order

EVIDENCE:
1. [FACT] 4294967286 equals 0xFFFFFFF6, which is -10 in 32-bit two's complement
2. [FACT] -10 equals B - A for the supplied inputs A=20, B=10

RECOMMENDED_INVESTIGATION:
1. Inspect the testbench driver's port connections for a/b ordering
2. Add a waveform capture on the next run to confirm operand values at the DUT boundary

CONFIDENCE:
medium - the arithmetic identity is solid, but the root cause is not yet confirmed.
"""


def _retriever(tmp_path) -> Retriever:
    store = VectorStore(tmp_path / "vector_store", collection_name="debug_test")
    model = DeterministicHashingEmbeddingModel()
    chunks = [
        Chunk(
            chunk_id="alu.sv::always_block[alu_control]::0::x",
            text="ALU_SUB: result = a - b;",
            source="alu.sv", file_type="systemverilog", project="alu",
            section="always_block[alu_control]", module="alu",
        ),
        Chunk(
            chunk_id="alu_sub_failure.md::intro::0::y",
            text="Previous SUB failure: operand ordering suspected, unconfirmed.",
            source="alu_sub_failure.md", file_type="markdown", project="alu",
            section="Introduction", module=None,
        ),
    ]
    store.add_embeddings(embed_chunks(chunks, model))
    return Retriever(store, model, default_top_k=4)


def test_analyze_returns_parsed_debug_result(tmp_path):
    retriever = _retriever(tmp_path)
    llm = FakeLLMClient(fixed_response=FAKE_RESPONSE)
    analyzer = DebugAnalyzer(retriever, llm)

    request = DebugRequest(
        module="alu",
        failure_report=(
            "Test: alu_sub_01\nOperation: SUB\nA=20\nB=10\nExpected=10\nObserved=4294967286"
        ),
    )
    result = analyzer.analyze(request)

    assert "SUB operation" in result.failure_summary
    assert "alu" in result.relevant_modules
    assert "result" in result.relevant_signals
    assert len(result.possible_root_causes) == 2
    assert len(result.evidence) == 2
    assert "medium" in result.confidence.lower()
    assert "alu.sv" in result.retrieved_sources


def test_analyze_keeps_evidence_and_hypotheses_in_separate_fields(tmp_path):
    retriever = _retriever(tmp_path)
    llm = FakeLLMClient(fixed_response=FAKE_RESPONSE)
    analyzer = DebugAnalyzer(retriever, llm)

    result = analyzer.analyze(DebugRequest(module="alu", failure_report="SUB mismatch"))

    # Evidence and root causes must never be merged into one list.
    assert result.evidence != result.possible_root_causes
    assert all("[FACT]" in e for e in result.evidence)
    assert all("[HYPOTHESIS]" in c for c in result.possible_root_causes)


def test_analyze_rejects_empty_failure_report(tmp_path):
    retriever = _retriever(tmp_path)
    llm = FakeLLMClient(fixed_response=FAKE_RESPONSE)
    analyzer = DebugAnalyzer(retriever, llm)

    with pytest.raises(ValueError):
        analyzer.analyze(DebugRequest(module="alu", failure_report=""))


def test_analyze_raises_llm_error_on_malformed_response(tmp_path):
    retriever = _retriever(tmp_path)
    llm = FakeLLMClient(fixed_response="not structured at all")
    analyzer = DebugAnalyzer(retriever, llm)

    with pytest.raises(LLMError):
        analyzer.analyze(DebugRequest(module="alu", failure_report="SUB mismatch"))
