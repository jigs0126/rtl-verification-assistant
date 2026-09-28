"""
Ingestion Schemas

What they are: the three data shapes that flow through the ingestion
pipeline — RawDocument (a file as read from disk), ParsedSection (a
structurally meaningful piece of that file), and Chunk (a retrieval-sized,
metadata-tagged unit ready for embedding).

Why required: loader.py, parser.py, chunker.py, and metadata.py need a
shared, typed contract between them so each stage doesn't have to guess
what shape the previous stage produced.

Input/Output: these are pure data containers — no logic lives here.

How they connect: RawDocument → (parser.py) → ParsedSection →
(chunker.py + metadata.py) → Chunk → (embedder.py, Phase 4) → embedding.

If removed: each ingestion module would pass around raw dicts or tuples,
with no validation and no single place to see the pipeline's data model.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class RawDocument(BaseModel):
    """A single file read from disk, before any parsing."""

    path: str
    source: str          # filename only, e.g. "alu_spec.md"
    file_type: str        # "systemverilog" | "verilog" | "markdown" | "text"
    project: str
    content: str


class ParsedSection(BaseModel):
    """A structurally meaningful piece of a document, before chunking."""

    source: str
    file_type: str
    project: str
    section: str           # e.g. "always_block[alu_control]", "## Overview"
    text: str
    module: Optional[str] = None


class Chunk(BaseModel):
    """A retrieval-ready unit of text with full source metadata."""

    chunk_id: str
    text: str
    source: str
    file_type: str
    project: str
    section: str
    module: Optional[str] = None

    def metadata(self) -> dict:
        """Metadata dict in the shape vector_store.py will persist (Phase 5)."""
        return {
            "source": self.source,
            "file_type": self.file_type,
            "project": self.project,
            "section": self.section,
            "chunk_id": self.chunk_id,
            "module": self.module or "",
        }


class EmbeddedChunk(BaseModel):
    """A Chunk plus its embedding vector — the unit vector_store.py (Phase 5)
    will persist. Kept separate from Chunk itself so ingestion code never
    has to know anything about embeddings, and embedding code never has
    to know anything about vector storage.

    IMPORTANT: `dimension` alone does not indicate compatibility between
    two EmbeddedChunks. Two different embedding models can happen to
    produce vectors of the same length while living in entirely
    unrelated vector spaces (e.g. a 384-dim hashing vector vs. a 384-dim
    all-MiniLM-L6-v2 vector). `model_name` is the actual identity of the
    space a vector lives in — only EmbeddedChunks with the same
    model_name are meaningfully comparable. See
    app.embeddings.embedder.assert_compatible_embeddings(), which the
    vector store uses to enforce this before persisting or querying.
    """

    model_config = ConfigDict(protected_namespaces=())

    chunk: Chunk
    embedding: list[float]
    dimension: int
    model_name: str


class RetrievedChunk(BaseModel):
    """A Chunk returned by a similarity search, with its relevance score.

    What it is: the output unit of retriever.py's query() — everything a
    caller (prompt builder, UI) needs to both use the chunk's text and
    show the user *why* it was retrieved.

    score: ChromaDB's cosine *distance* (lower = more similar, 0.0 =
    identical) — see retriever.py's docstring for why we surface the raw
    distance rather than inventing a fake "similarity percentage".
    """

    chunk: Chunk
    score: float
    rank: int


class RetrievedContext(BaseModel):
    """The full result of one retrieval call: the query plus what came
    back for it. Kept as its own object (rather than a bare
    list[RetrievedChunk]) so prompts.py and the UI can show the query
    alongside its results, and so "nothing relevant was retrieved" is a
    real, inspectable value (empty `results`) rather than silently
    disappearing."""

    query: str
    results: list[RetrievedChunk]
    top_k: int

    @property
    def is_empty(self) -> bool:
        return len(self.results) == 0
