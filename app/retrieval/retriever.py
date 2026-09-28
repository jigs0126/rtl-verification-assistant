"""
Retriever

What it is: turns a natural-language query string into a ranked list of
relevant chunks, by embedding the query with the same EmbeddingModel used
at indexing time and searching the VectorStore for the nearest vectors.

Why required: this is the actual "R" in RAG. Without it, an LLM prompt
could only ever be grounded in whatever the caller manually pasted in —
there would be no automatic way to find which pieces of the project
corpus are relevant to a given request.

Input: a query string, top_k.
Output: RetrievedContext (query + ranked list[RetrievedChunk]).

How it connects to the next component: prompts.py (Phase 7) takes a
RetrievedContext and formats it into the "PROJECT CONTEXT" section of
the LLM prompt.

What would happen if it were removed: TestbenchGenerator/DebugAnalyzer
would have no way to fetch relevant RTL/docs automatically — the system
would degrade into "whatever context the user happens to paste in
manually", which is not RAG.

------------------------------------------------------------------------
ON KEYWORD SEARCH VS. SEMANTIC RETRIEVAL:

Keyword search matches literal substrings/tokens: searching "subtraction"
would NOT match a chunk that only contains "SUB operation" or "a - b",
because none of those share the literal token "subtraction".

Semantic retrieval instead compares MEANING, via embedding vectors: the
query is embedded the same way the corpus was, and chunks are ranked by
vector distance. A well-trained embedding model places "How is
subtraction implemented?" close to chunks about SUB/`a - b`/two's
complement, even though none of those chunks contain the word
"subtraction" verbatim. This is what makes RAG retrieval meaningfully
better than grep for this project's use case: verification engineers ask
questions in natural language, not RTL identifier syntax.

(Caveat carried over from Phase 3: this benefit is only real when the
underlying vectors come from a trained semantic model, e.g.
all-MiniLM-L6-v2 — see embedder.py. The deterministic hashing fallback
this sandbox actually runs on is lexical/character-overlap based, closer
to keyword search than true semantic retrieval; the retrieval mechanism
built here is correct and model-agnostic, but its retrieval *quality* in
this environment is limited by which backend is active.)
------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Optional

from app.embeddings.embedder import EmbeddingModel
from app.retrieval.vector_store import VectorStore
from app.schemas.models import Chunk, RetrievedChunk, RetrievedContext


class Retriever:
    """Combines an EmbeddingModel and a VectorStore into query() -> RetrievedContext."""

    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel, default_top_k: int = 4):
        self._store = vector_store
        self._model = embedding_model
        self._default_top_k = default_top_k

    def query(self, query_text: str, top_k: Optional[int] = None) -> RetrievedContext:
        """
        Embed query_text and return the top_k most similar chunks as a
        RetrievedContext. Raises ValueError on an empty/whitespace-only
        query (per §23's "empty user query" error-handling requirement)
        rather than silently returning nothing or embedding garbage.
        """
        if not query_text or not query_text.strip():
            raise ValueError("Query text must not be empty")

        k = top_k if top_k is not None else self._default_top_k
        query_vector = self._model.embed_text(query_text)

        raw_results = self._store.query_by_embedding(
            query_embedding=query_vector, model_name=self._model.model_name, top_k=k
        )

        results = [
            RetrievedChunk(
                chunk=self._store_record_to_chunk(record),
                score=record["distance"],
                rank=i + 1,
            )
            for i, record in enumerate(raw_results)
        ]
        return RetrievedContext(query=query_text, results=results, top_k=k)

    @staticmethod
    def _store_record_to_chunk(record: dict) -> Chunk:
        meta = record["metadata"]
        return Chunk(
            chunk_id=record["chunk_id"],
            text=record["text"],
            source=meta["source"],
            file_type=meta["file_type"],
            project=meta["project"],
            section=meta["section"],
            module=meta.get("module") or None,
        )
