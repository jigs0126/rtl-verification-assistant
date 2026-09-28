"""
Vector Store

What it is: a thin wrapper around a local, persistent ChromaDB collection
that stores each EmbeddedChunk's vector, text, and metadata to disk.

Why required: Phase 3's embeddings only exist in memory for as long as a
script runs — recomputing every embedding on every application start
would be slow and pointless, since the underlying RTL/docs rarely
change between runs. A persistent vector store lets ingestion +
embedding happen once (an explicit "index" step, per §22 of the
project spec — never silently re-run on every app start) and be reused
across every later query.

Input: EmbeddedChunk objects (from app.embeddings.embedder.embed_chunks).
Output: nothing returned from storage calls — the durable output is the
on-disk Chroma database under settings.vector_db_path. Read access is
intentionally minimal in this phase (count/list/metadata only); turning
a natural-language query into a ranked list of relevant chunks is
retriever.py's job (Phase 6), which will call this store's low-level
`query_by_embedding` once it exists.

How it connects to the next component: retriever.py (Phase 6) will embed
an incoming query with the same EmbeddingModel used at indexing time,
then search this store for the top_k nearest vectors.

What would happen if it were removed: there would be nowhere to persist
embeddings — every query would require re-ingesting and re-embedding the
entire project from scratch, defeating the point of building an index.

------------------------------------------------------------------------
On mixing embedding models (see app/embeddings/embedder.py for the full
explanation): this store records model_name and dimension as
collection-level metadata the first time a collection is created, and
every subsequent add_embeddings() call is checked against it. Adding
vectors from a different embedding model into a collection that already
holds vectors from another model raises ValueError immediately — same
dimension is not sufficient grounds to consider two models' vectors
comparable.
------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import chromadb

from app.embeddings.embedder import assert_compatible_embeddings
from app.schemas.models import EmbeddedChunk

DEFAULT_COLLECTION_NAME = "rtl_project_chunks"


class ModelMismatchError(ValueError):
    """Raised when embeddings from a different embedding model are added
    to a collection that already holds vectors from another model."""


class VectorStore:
    """Persistent, local Chroma-backed store for EmbeddedChunk vectors."""

    def __init__(self, path: Path, collection_name: str = DEFAULT_COLLECTION_NAME):
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._path))
        self._collection_name = collection_name

    # ----------------------------------------------------------- setup --

    def _get_or_create_collection(self, model_name: str, dimension: int):
        """
        Get the collection, creating it (with model_name/dimension
        recorded as metadata) if it doesn't exist yet. If it already
        exists, verify the recorded model_name matches — Chroma's
        get_or_create does not update metadata on an existing
        collection, so the check has to happen explicitly here.
        """
        collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={
                "model_name": model_name,
                "dimension": dimension,
                # cosine distance is the right choice since our vectors
                # are unit-normalized (both embedding backends normalize).
                "hnsw:space": "cosine",
            },
        )
        stored_model_name = (collection.metadata or {}).get("model_name")
        if stored_model_name is not None and stored_model_name != model_name:
            raise ModelMismatchError(
                f"Collection '{self._collection_name}' already contains embeddings "
                f"from model '{stored_model_name}'. Cannot add embeddings from "
                f"'{model_name}' into the same collection — use a different "
                "collection_name, or reset() this one first, to avoid mixing "
                "vectors from different, non-comparable embedding spaces."
            )
        return collection

    # ------------------------------------------------------------ write --

    def add_embeddings(self, embedded_chunks: list[EmbeddedChunk]) -> int:
        """
        Persist a batch of EmbeddedChunks. Upserts by chunk_id, so
        re-indexing the same source file (same chunk_id) updates its
        vector/text/metadata in place rather than duplicating it.
        Returns the number of chunks written.
        """
        if not embedded_chunks:
            raise ValueError("add_embeddings received an empty list")

        # Guards against mixing models *within this batch* (see embedder.py).
        assert_compatible_embeddings(embedded_chunks)

        model_name = embedded_chunks[0].model_name
        dimension = embedded_chunks[0].dimension
        collection = self._get_or_create_collection(model_name, dimension)

        collection.upsert(
            ids=[e.chunk.chunk_id for e in embedded_chunks],
            embeddings=[e.embedding for e in embedded_chunks],
            documents=[e.chunk.text for e in embedded_chunks],
            metadatas=[e.chunk.metadata() for e in embedded_chunks],
        )
        return len(embedded_chunks)

    # ------------------------------------------------------------- read --

    def count(self) -> int:
        """Number of chunks currently stored (0 if the collection doesn't exist yet)."""
        try:
            collection = self._client.get_collection(name=self._collection_name)
        except Exception:
            return 0
        return collection.count()

    def collection_metadata(self) -> Optional[dict]:
        """The model_name/dimension recorded for this collection, or None if empty/unindexed."""
        try:
            collection = self._client.get_collection(name=self._collection_name)
        except Exception:
            return None
        return dict(collection.metadata or {})

    def list_ids(self) -> list[str]:
        """All chunk_ids currently stored — used by the Knowledge Base UI (Phase 10)."""
        try:
            collection = self._client.get_collection(name=self._collection_name)
        except Exception:
            return []
        result = collection.get(include=[])  # ids are always returned
        return list(result.get("ids", []))

    # ------------------------------------------------------------ query --

    def query_by_embedding(self, query_embedding: list[float], model_name: str, top_k: int = 4) -> list[dict]:
        """
        Similarity search by a pre-computed query embedding. model_name
        must match the model this collection was indexed with — this is
        what stops a caller from accidentally embedding a query with one
        model and searching vectors produced by another (same dimension
        does not imply comparable vectors; see embedder.py).

        Returns a list of dicts (nearest first):
            {"chunk_id": str, "text": str, "metadata": dict, "distance": float}
        Empty list if the collection doesn't exist yet or is empty.

        `distance` is Chroma's cosine distance (1 - cosine_similarity):
        0.0 = identical direction, 2.0 = opposite. Lower is more similar.
        Returned as-is rather than converted into a fabricated
        "similarity %", so the UI shows a real, documented number.
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        try:
            collection = self._client.get_collection(name=self._collection_name)
        except Exception:
            return []

        stored_model_name = (collection.metadata or {}).get("model_name")
        if stored_model_name is not None and stored_model_name != model_name:
            raise ModelMismatchError(
                f"Query embedding came from model '{model_name}', but collection "
                f"'{self._collection_name}' was indexed with '{stored_model_name}'. "
                "Use the same embedding model for indexing and querying."
            )

        count = collection.count()
        if count == 0:
            return []

        result = collection.query(query_embeddings=[query_embedding], n_results=min(top_k, count))
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        return [
            {"chunk_id": ids[i], "text": documents[i], "metadata": metadatas[i], "distance": distances[i]}
            for i in range(len(ids))
        ]

    # ------------------------------------------------------------ reset --

    def reset(self) -> None:
        """
        Delete the collection entirely. Used by the explicit
        "Re-index Knowledge Base" action (§22) — indexing must never
        silently happen on every app start, and re-indexing from a clean
        slate (rather than upserting stale entries forever) needs an
        explicit way to clear out anything removed from the source project.
        """
        try:
            self._client.delete_collection(name=self._collection_name)
        except Exception:
            pass
