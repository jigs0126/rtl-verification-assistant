import math
import sys

import pytest

from app.embeddings.embedder import (
    DeterministicHashingEmbeddingModel,
    assert_compatible_embeddings,
    embed_chunks,
    get_embedding_model,
)
from app.schemas.models import Chunk

# All tests here run against DeterministicHashingEmbeddingModel, the
# network-free fallback — see the module docstring in embedder.py for
# why the real SentenceTransformerEmbeddingModel can't be exercised in
# this sandbox. Both implementations satisfy the same EmbeddingModel
# contract, which is what these tests actually verify.


def _model(dimension: int = 384) -> DeterministicHashingEmbeddingModel:
    return DeterministicHashingEmbeddingModel(dimension=dimension)


# --------------------------------------------------------------- init ---


def test_model_initializes_with_default_dimension():
    model = DeterministicHashingEmbeddingModel()
    assert model.dimension == 384


def test_model_initializes_with_custom_dimension():
    model = DeterministicHashingEmbeddingModel(dimension=128)
    assert model.dimension == 128


def test_model_rejects_non_positive_dimension():
    with pytest.raises(ValueError):
        DeterministicHashingEmbeddingModel(dimension=0)
    with pytest.raises(ValueError):
        DeterministicHashingEmbeddingModel(dimension=-5)


def test_model_name_is_set():
    model = DeterministicHashingEmbeddingModel(model_name="test-model")
    assert model.model_name == "test-model"


# ---------------------------------------------------------- dimension ---


def test_embedding_dimension_matches_declared_dimension():
    model = _model(dimension=256)
    vector = model.embed_text("ALU_SUB: result = a - b;")
    assert len(vector) == 256
    assert model.dimension == 256


def test_embedding_dimension_consistent_across_different_texts():
    model = _model()
    v1 = model.embed_text("short text")
    v2 = model.embed_text(
        "a much longer piece of text describing the ALU SUB operation "
        "and its two's-complement wraparound behavior in detail"
    )
    assert len(v1) == len(v2) == model.dimension


# --------------------------------------------------------- determinism --


def test_embedding_is_deterministic_for_repeated_calls():
    model = _model()
    text = "ALU_SUB: result = a - b;"
    v1 = model.embed_text(text)
    v2 = model.embed_text(text)
    assert v1 == v2


def test_embedding_differs_for_different_text():
    model = _model()
    v1 = model.embed_text("ADD operation")
    v2 = model.embed_text("SUB operation")
    assert v1 != v2


def test_embedding_is_unit_normalized():
    model = _model()
    vector = model.embed_text("normalize me please")
    norm = math.sqrt(sum(x * x for x in vector))
    assert math.isclose(norm, 1.0, rel_tol=1e-6)


# ------------------------------------------------------------- batching -


def test_embed_batch_matches_individual_embed_calls():
    model = _model()
    texts = ["ADD operation", "SUB operation", "AND operation"]
    batch_vectors = model.embed_batch(texts)
    individual_vectors = [model.embed_text(t) for t in texts]
    assert batch_vectors == individual_vectors


def test_embed_batch_returns_one_vector_per_text():
    model = _model()
    texts = ["one", "two", "three", "four"]
    vectors = model.embed_batch(texts)
    assert len(vectors) == 4
    assert all(len(v) == model.dimension for v in vectors)


# ------------------------------------------------------ empty handling --


def test_embed_text_rejects_empty_string():
    model = _model()
    with pytest.raises(ValueError):
        model.embed_text("")


def test_embed_text_rejects_whitespace_only_string():
    model = _model()
    with pytest.raises(ValueError):
        model.embed_text("   \n\t  ")


def test_embed_batch_rejects_empty_list():
    model = _model()
    with pytest.raises(ValueError):
        model.embed_batch([])


# ------------------------------------------------------------- factory --


def test_get_embedding_model_returns_an_embedding_model():
    model = get_embedding_model()
    assert hasattr(model, "embed_text")
    assert hasattr(model, "embed_batch")
    assert model.dimension > 0


def test_get_embedding_model_warns_when_falling_back(monkeypatch):
    # Simulate sentence-transformers being unavailable (setting the entry in
    # sys.modules to None makes `import sentence_transformers` raise
    # ImportError), so this test behaves the same whether or not the package
    # is actually installed on the machine running it.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.warns(UserWarning, match="DeterministicHashingEmbeddingModel"):
        model = get_embedding_model("all-MiniLM-L6-v2")
    assert model.dimension == 384


# --------------------------------------------------------- embed_chunks -


def _chunk(text: str, chunk_id: str = "c1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id, text=text, source="alu.sv", file_type="systemverilog",
        project="alu", section="always_block[alu_control]", module="alu",
    )


def test_embed_chunks_produces_one_embedded_chunk_per_chunk():
    model = _model()
    chunks = [_chunk("ADD operation", "c1"), _chunk("SUB operation", "c2")]
    embedded = embed_chunks(chunks, model)

    assert len(embedded) == 2
    assert embedded[0].chunk.chunk_id == "c1"
    assert embedded[1].chunk.chunk_id == "c2"
    assert all(e.dimension == model.dimension for e in embedded)
    assert all(e.model_name == model.model_name for e in embedded)
    assert all(len(e.embedding) == model.dimension for e in embedded)


def test_embed_chunks_rejects_empty_chunk_list():
    model = _model()
    with pytest.raises(ValueError):
        embed_chunks([], model)


# ------------------------------------------------ model incompatibility -


def test_same_dimension_does_not_imply_same_model_name():
    # Two backends can share a dimension (both 384) while being
    # completely different, non-comparable vector spaces. model_name is
    # the actual identity signal — this test pins that distinction down.
    hashing_model = DeterministicHashingEmbeddingModel(dimension=384)
    assert hashing_model.dimension == 384
    assert hashing_model.model_name != "all-MiniLM-L6-v2"
    assert hashing_model.model_name == "deterministic-hashing-v1"


def test_assert_compatible_embeddings_allows_single_model():
    model = _model()
    chunks = [_chunk("ADD operation", "c1"), _chunk("SUB operation", "c2")]
    embedded = embed_chunks(chunks, model)
    assert_compatible_embeddings(embedded)  # should not raise


def test_assert_compatible_embeddings_allows_empty_list():
    assert_compatible_embeddings([])  # should not raise


def test_assert_compatible_embeddings_rejects_mixed_model_names():
    model_a = DeterministicHashingEmbeddingModel(dimension=384, model_name="model-a")
    model_b = DeterministicHashingEmbeddingModel(dimension=384, model_name="model-b")

    chunk_a = embed_chunks([_chunk("ADD operation", "c1")], model_a)[0]
    chunk_b = embed_chunks([_chunk("SUB operation", "c2")], model_b)[0]

    # Same dimension (384 == 384) but different model_name — must be
    # rejected even though the vectors are structurally compatible
    # (same length), because they are NOT semantically comparable.
    assert chunk_a.dimension == chunk_b.dimension
    assert chunk_a.model_name != chunk_b.model_name

    with pytest.raises(ValueError, match="Cannot mix embeddings"):
        assert_compatible_embeddings([chunk_a, chunk_b])
