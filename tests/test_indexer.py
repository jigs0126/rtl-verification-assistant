from pathlib import Path

from app.embeddings.embedder import DeterministicHashingEmbeddingModel
from app.retrieval.indexer import index_project
from app.retrieval.vector_store import VectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_index_project_indexes_the_real_alu_corpus(tmp_path):
    store = VectorStore(tmp_path / "vector_store", collection_name="alu_test")
    model = DeterministicHashingEmbeddingModel()

    count = index_project(PROJECT_ROOT, store, model, project_name="alu")

    assert count > 0
    assert store.count() == count


def test_index_project_is_idempotent_on_repeated_calls(tmp_path):
    store = VectorStore(tmp_path / "vector_store", collection_name="alu_test")
    model = DeterministicHashingEmbeddingModel()

    first = index_project(PROJECT_ROOT, store, model, project_name="alu")
    second = index_project(PROJECT_ROOT, store, model, project_name="alu")

    assert first == second
    assert store.count() == first  # not doubled


def test_index_project_returns_zero_for_empty_corpus(tmp_path):
    empty_project = tmp_path / "empty_project"
    empty_project.mkdir()
    store = VectorStore(tmp_path / "vector_store", collection_name="empty_test")
    model = DeterministicHashingEmbeddingModel()

    count = index_project(empty_project, store, model, project_name="nothing")

    assert count == 0
    assert store.count() == 0
