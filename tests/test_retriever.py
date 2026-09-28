import pytest

from app.embeddings.embedder import DeterministicHashingEmbeddingModel, embed_chunks
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore
from app.schemas.models import Chunk


def _chunk(text: str, chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, text=text, source="alu.sv", file_type="systemverilog",
        project="alu", section="always_block[alu_control]", module="alu",
    )


def _populated_retriever(tmp_path) -> Retriever:
    store = VectorStore(tmp_path / "vector_store", collection_name="retriever_test")
    model = DeterministicHashingEmbeddingModel()
    chunks = [
        _chunk("SUB operation performs subtraction a minus b", "c_sub"),
        _chunk("ADD operation performs addition a plus b", "c_add"),
        _chunk("verification plan corner cases for the ALU", "c_plan"),
    ]
    store.add_embeddings(embed_chunks(chunks, model))
    return Retriever(store, model, default_top_k=2)


def test_query_returns_retrieved_context_with_query_text(tmp_path):
    retriever = _populated_retriever(tmp_path)
    ctx = retriever.query("How does subtraction work?")
    assert ctx.query == "How does subtraction work?"
    assert not ctx.is_empty


def test_query_respects_default_top_k(tmp_path):
    retriever = _populated_retriever(tmp_path)  # default_top_k=2
    ctx = retriever.query("ALU operation")
    assert ctx.top_k == 2
    assert len(ctx.results) == 2


def test_query_respects_explicit_top_k_override(tmp_path):
    retriever = _populated_retriever(tmp_path)
    ctx = retriever.query("ALU operation", top_k=3)
    assert ctx.top_k == 3
    assert len(ctx.results) == 3


def test_query_results_are_ranked_1_indexed_in_order(tmp_path):
    retriever = _populated_retriever(tmp_path)
    ctx = retriever.query("SUB operation performs subtraction a minus b", top_k=3)
    ranks = [r.rank for r in ctx.results]
    assert ranks == [1, 2, 3]
    # exact-text match should be the top result
    assert ctx.results[0].chunk.chunk_id == "c_sub"


def test_query_rejects_empty_string(tmp_path):
    retriever = _populated_retriever(tmp_path)
    with pytest.raises(ValueError):
        retriever.query("")


def test_query_rejects_whitespace_only_string(tmp_path):
    retriever = _populated_retriever(tmp_path)
    with pytest.raises(ValueError):
        retriever.query("   ")


def test_query_on_empty_store_returns_empty_context(tmp_path):
    store = VectorStore(tmp_path / "vector_store", collection_name="unindexed")
    model = DeterministicHashingEmbeddingModel()
    retriever = Retriever(store, model)

    ctx = retriever.query("anything at all")
    assert ctx.is_empty
    assert ctx.results == []
