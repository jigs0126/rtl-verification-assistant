import pytest

from app.embeddings.embedder import DeterministicHashingEmbeddingModel, embed_chunks
from app.retrieval.vector_store import ModelMismatchError, VectorStore
from app.schemas.models import Chunk


def _chunk(text: str, chunk_id: str, source: str = "alu.sv", section: str = "always_block[alu_control]") -> Chunk:
    return Chunk(
        chunk_id=chunk_id, text=text, source=source, file_type="systemverilog",
        project="alu", section=section, module="alu",
    )


def _store(tmp_path, name="test_collection") -> VectorStore:
    return VectorStore(tmp_path / "vector_store", collection_name=name)


def test_new_store_is_empty(tmp_path):
    store = _store(tmp_path)
    assert store.count() == 0
    assert store.collection_metadata() is None
    assert store.list_ids() == []


def test_add_embeddings_persists_chunks(tmp_path):
    store = _store(tmp_path)
    model = DeterministicHashingEmbeddingModel()
    chunks = [_chunk("ADD operation", "c1"), _chunk("SUB operation", "c2")]
    embedded = embed_chunks(chunks, model)

    written = store.add_embeddings(embedded)
    assert written == 2
    assert store.count() == 2
    assert set(store.list_ids()) == {"c1", "c2"}


def test_add_embeddings_rejects_empty_list(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        store.add_embeddings([])


def test_reindexing_same_chunk_ids_does_not_duplicate(tmp_path):
    store = _store(tmp_path)
    model = DeterministicHashingEmbeddingModel()
    chunks = [_chunk("ADD operation", "c1"), _chunk("SUB operation", "c2")]
    embedded = embed_chunks(chunks, model)

    store.add_embeddings(embedded)
    store.add_embeddings(embedded)  # re-index the same corpus

    assert store.count() == 2  # not 4


def test_reindexing_updates_existing_record(tmp_path):
    store = _store(tmp_path)
    model = DeterministicHashingEmbeddingModel()

    store.add_embeddings(embed_chunks([_chunk("old text", "c1")], model))
    store.add_embeddings(embed_chunks([_chunk("new text", "c1")], model))

    assert store.count() == 1
    results = store.query_by_embedding(model.embed_text("new text"), model.model_name, top_k=1)
    assert results[0]["text"] == "new text"


def test_collection_records_model_name_and_dimension(tmp_path):
    store = _store(tmp_path)
    model = DeterministicHashingEmbeddingModel(dimension=128, model_name="test-model-v1")
    store.add_embeddings(embed_chunks([_chunk("text", "c1")], model))

    meta = store.collection_metadata()
    assert meta["model_name"] == "test-model-v1"
    assert meta["dimension"] == 128


def test_adding_different_model_to_existing_collection_raises(tmp_path):
    store = _store(tmp_path)
    model_a = DeterministicHashingEmbeddingModel(model_name="model-a")
    model_b = DeterministicHashingEmbeddingModel(model_name="model-b")

    store.add_embeddings(embed_chunks([_chunk("text a", "c1")], model_a))

    with pytest.raises(ModelMismatchError):
        store.add_embeddings(embed_chunks([_chunk("text b", "c2")], model_b))

    assert store.count() == 1  # the mismatched batch was never written


def test_query_returns_nearest_first(tmp_path):
    store = _store(tmp_path)
    model = DeterministicHashingEmbeddingModel()
    chunks = [
        _chunk("ADD operation performs addition", "c_add"),
        _chunk("SUB operation performs subtraction", "c_sub"),
        _chunk("completely unrelated verification plan text", "c_other"),
    ]
    store.add_embeddings(embed_chunks(chunks, model))

    query_vector = model.embed_text("SUB operation performs subtraction")
    results = store.query_by_embedding(query_vector, model.model_name, top_k=3)

    assert len(results) == 3
    assert results[0]["chunk_id"] == "c_sub"  # exact text match should rank first
    assert results[0]["distance"] <= results[1]["distance"] <= results[2]["distance"]


def test_query_on_empty_collection_returns_empty_list(tmp_path):
    store = _store(tmp_path)
    model = DeterministicHashingEmbeddingModel()
    results = store.query_by_embedding(model.embed_text("anything"), model.model_name, top_k=3)
    assert results == []


def test_query_with_mismatched_model_raises(tmp_path):
    store = _store(tmp_path)
    model_a = DeterministicHashingEmbeddingModel(model_name="model-a")
    model_b = DeterministicHashingEmbeddingModel(model_name="model-b")
    store.add_embeddings(embed_chunks([_chunk("text", "c1")], model_a))

    with pytest.raises(ModelMismatchError):
        store.query_by_embedding(model_b.embed_text("text"), model_b.model_name, top_k=1)


def test_query_rejects_non_positive_top_k(tmp_path):
    store = _store(tmp_path)
    model = DeterministicHashingEmbeddingModel()
    store.add_embeddings(embed_chunks([_chunk("text", "c1")], model))
    with pytest.raises(ValueError):
        store.query_by_embedding(model.embed_text("text"), model.model_name, top_k=0)


def test_reset_clears_the_collection(tmp_path):
    store = _store(tmp_path)
    model = DeterministicHashingEmbeddingModel()
    store.add_embeddings(embed_chunks([_chunk("text", "c1")], model))
    assert store.count() == 1

    store.reset()
    assert store.count() == 0
    assert store.collection_metadata() is None


def test_reset_on_nonexistent_collection_does_not_raise(tmp_path):
    store = _store(tmp_path)
    store.reset()  # should be a no-op, not an error
