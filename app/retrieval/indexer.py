"""
Indexer

What it is: the single function that performs an explicit "index this
project" run: ingest (Phase 2) -> embed (Phase 3) -> persist (Phase 4/5).

Why required: §22 of the project spec requires indexing to be an
explicit, visible action (e.g. a "Re-index Knowledge Base" button in the
eventual UI), never something that silently happens on every app start.
Keeping that whole sequence in one function gives the UI (Phase 10) and
any test/CLI usage a single, obvious call site for that action.

Input: project_root, project_name, an EmbeddingModel, a VectorStore, and
the configured chunk_size/chunk_overlap.
Output: the number of chunks indexed (int); the durable output is
whatever the VectorStore persisted to disk.

How it connects to the next component: retriever.py (Phase 6) reads from
the same VectorStore this function wrote to, using the same
EmbeddingModel to embed incoming queries.

What would happen if it were removed: the UI (or a test) would have to
manually call ingest_project(), then embed_chunks(), then
vector_store.add_embeddings() in the right order itself every time.
"""

from __future__ import annotations

from pathlib import Path

from app.embeddings.embedder import EmbeddingModel, embed_chunks
from app.ingestion.pipeline import ingest_project
from app.retrieval.vector_store import VectorStore


def index_project(
    project_root: Path,
    vector_store: VectorStore,
    embedding_model: EmbeddingModel,
    project_name: str = "alu",
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> int:
    """Run the full ingest -> embed -> persist sequence. Returns chunk count indexed."""
    chunks = ingest_project(
        project_root,
        project_name=project_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        return 0

    embedded_chunks = embed_chunks(chunks, embedding_model)
    return vector_store.add_embeddings(embedded_chunks)
