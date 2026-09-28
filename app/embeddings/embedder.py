"""
Embedding Layer

What it is: converts chunk text into numerical vectors ("embeddings") that
represent the text's semantic content, behind a single EmbeddingModel
interface so the rest of the application never depends on a specific
embedding library directly.

    text
      |
      v
  embedding model
      |
      v
   vector (list[float])

The vector represents semantic information about the text. To be precise
about what does what: the embedding MODEL produces these vectors; the
VECTOR DATABASE (Phase 5) only stores and searches them by similarity — it
does not itself "understand" anything.

Why required: an LLM prompt can only be grounded in retrieved chunks if
those chunks can be ranked by relevance to a query, and ranking by
relevance is exactly what comparing embedding vectors (e.g. cosine
similarity) does.

Input: chunk text (str) or a batch of chunk texts (list[str]).
Output: a fixed-length vector (list[float]) per input text — same length
for every text produced by a given model, which is what "dimension" means.

How it connects to the next component: vector_store.py (Phase 5) will
call an EmbeddingModel to embed both stored chunks and incoming queries,
then compare vectors to find the top_k most similar chunks.

What would happen if it were removed: there would be nothing to search
by meaning — retrieval would have to fall back to exact keyword matching,
which is exactly what RAG is meant to improve on (see docs discussion in
prompts.py, Phase 7, on keyword vs. semantic search).

------------------------------------------------------------------------
IMPORTANT ENVIRONMENT NOTE (read this before running the tests):

This module ships two EmbeddingModel implementations:

1. SentenceTransformerEmbeddingModel — the real, production implementation.
   Wraps sentence-transformers' `all-MiniLM-L6-v2` (or any other
   sentence-transformers model named in settings.embedding_model). This is
   what the project is *supposed* to run on, and what you should use when
   running this locally on your own machine.

2. DeterministicHashingEmbeddingModel — a network-free, dependency-light
   fallback. It builds a vector via feature hashing over word and
   character-trigram tokens, then L2-normalizes it. It produces a
   deterministic, fixed-dimension vector for any text with NO external
   model weights and NO network access required — but it is NOT a
   semantic embedding: it will not capture meaning the way a trained
   transformer does. It exists purely so the embedding *interface* and
   *pipeline* can be exercised and tested in network-restricted
   environments (like the sandbox this project was built in, which
   cannot reach huggingface.co or download.pytorch.org to fetch real
   model weights).

get_embedding_model() below picks #1 when sentence-transformers is
importable, and falls back to #2 (with a clear warning) otherwise. All
unit tests in tests/test_embedder.py run against the fallback, since
that's what's actually executable without network access — but they
test the EmbeddingModel *contract* (dimension consistency, determinism,
batching, empty-input handling, initialization), which both
implementations must satisfy identically. When you run this on your own
machine with `pip install sentence-transformers`, the exact same
pipeline code will transparently use the real model instead — nothing
else in the codebase needs to change.
------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import math
import warnings
from abc import ABC, abstractmethod
from typing import Optional

from app.schemas.models import Chunk, EmbeddedChunk

# Known output dimensions for common sentence-transformers models, so
# SentenceTransformerEmbeddingModel can report `.dimension` without
# having to load the model (which would require network access) just to
# ask it its own output size.
KNOWN_MODEL_DIMENSIONS = {
    "all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "sentence-transformers/all-mpnet-base-v2": 768,
}


class EmbeddingModel(ABC):
    """Interface every embedding backend must implement."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Length of every vector this model produces."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier for which model produced the vectors (stored in metadata)."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single string. Raises ValueError on empty/whitespace-only text."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple strings. The default implementation simply calls
        embed_text per item; concrete backends may override this to use
        a real batched call (sentence-transformers' .encode() accepts a
        list directly and is meaningfully faster in batch).
        """
        if not texts:
            raise ValueError("embed_batch received an empty list of texts")
        return [self.embed_text(t) for t in texts]


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    """
    Real, production embedding backend: wraps
    sentence_transformers.SentenceTransformer. The model is loaded lazily
    (on first embed call, not at construction) and cached at the instance
    level, so constructing this object is cheap and network-free —
    network access is only needed the first time you actually embed
    something (to download the model weights, cached by
    sentence-transformers under ~/.cache after that).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None  # loaded lazily
        self._dimension: Optional[int] = KNOWN_MODEL_DIMENSIONS.get(model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only when the dep is missing
            raise ImportError(
                "sentence-transformers is not installed. Install it with "
                "`pip install sentence-transformers` to use "
                "SentenceTransformerEmbeddingModel, or use "
                "DeterministicHashingEmbeddingModel / get_embedding_model() "
                "for a network-free fallback."
            ) from exc
        self._model = SentenceTransformer(self._model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._ensure_loaded()
        return self._dimension  # type: ignore[return-value]

    def embed_text(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Cannot embed empty or whitespace-only text")
        self._ensure_loaded()
        vector = self._model.encode(text, convert_to_numpy=True)  # type: ignore[union-attr]
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("embed_batch received an empty list of texts")
        if any(not t or not t.strip() for t in texts):
            raise ValueError("Cannot embed empty or whitespace-only text in a batch")
        self._ensure_loaded()
        vectors = self._model.encode(texts, convert_to_numpy=True)  # type: ignore[union-attr]
        return [v.tolist() for v in vectors]


class DeterministicHashingEmbeddingModel(EmbeddingModel):
    """
    Network-free fallback embedding backend, for tests/development ONLY.

    *** NOT a substitute for a real embedding model. Do not use this as ***
    *** the retrieval backend. ***

    Its vectors happen to default to 384 dimensions — the same as
    all-MiniLM-L6-v2 — purely so the two backends are drop-in compatible
    at the *interface* level (same vector length, same EmbeddingModel
    API). That shared dimension does NOT make the vectors semantically
    equivalent or comparable: a hashing vector for "SUB operation" and a
    real MiniLM vector for "SUB operation" are not the same point in any
    shared space, and cosine similarity between a hashing vector and a
    MiniLM vector is meaningless — they were produced by unrelated
    mathematical processes. Two embeddings are only meaningfully
    comparable when they came from the *same* embedding model (see
    `model_name`, and `assert_compatible_embeddings()` below, which the
    vector store uses to guard against exactly this mistake).

    Produces a deterministic, unit-length vector via feature hashing:

        text -> tokens (words + character trigrams)
             -> each token hashed to an index in [0, dimension)
             -> that index incremented (or decremented, by hash sign)
             -> vector L2-normalized

    This is a real, well-known technique (the "hashing trick", used e.g.
    in scikit-learn's HashingVectorizer) — it is deterministic and fast,
    but it captures lexical/character overlap, not learned semantic
    similarity. "SUB operation" and "subtract two operands" will NOT be
    placed near each other by this model the way a trained
    sentence-transformer is designed to place them, because it has no
    learned notion of synonymy or meaning — only shared substrings.
    """

    def __init__(self, dimension: int = 384, model_name: str = "deterministic-hashing-v1"):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self._model_name = model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def _tokens(self, text: str) -> list[str]:
        words = text.lower().split()
        trigrams = [text[i : i + 3] for i in range(max(len(text) - 2, 0))]
        return words + trigrams

    def embed_text(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Cannot embed empty or whitespace-only text")

        vector = [0.0] * self._dimension
        for token in self._tokens(text):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self._dimension
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingModel:
    """
    Factory: returns the real SentenceTransformerEmbeddingModel when
    sentence-transformers is importable, otherwise falls back to
    DeterministicHashingEmbeddingModel with a clear warning. This is the
    function the rest of the application (embed_chunks below, and later
    vector_store.py) should call, rather than constructing a specific
    EmbeddingModel subclass directly.
    """
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        warnings.warn(
            "sentence-transformers is not installed (or its model weights "
            "are not reachable from this environment). Falling back to "
            "DeterministicHashingEmbeddingModel, which is NOT a semantic "
            "embedding model. Install sentence-transformers and ensure "
            "network access to huggingface.co to use the real model.",
            stacklevel=2,
        )
        return DeterministicHashingEmbeddingModel(
            dimension=KNOWN_MODEL_DIMENSIONS.get(model_name, 384)
        )
    return SentenceTransformerEmbeddingModel(model_name=model_name)


def embed_chunks(chunks: list[Chunk], model: EmbeddingModel) -> list[EmbeddedChunk]:
    """
    Embed a list of Chunks (from the Phase 2 ingestion pipeline) using
    the given EmbeddingModel, returning EmbeddedChunk objects ready for
    vector_store.py (Phase 5) to persist. Uses embed_batch for efficiency
    rather than embedding one chunk at a time.
    """
    if not chunks:
        raise ValueError("embed_chunks received an empty list of chunks")

    vectors = model.embed_batch([c.text for c in chunks])
    return [
        EmbeddedChunk(
            chunk=chunk,
            embedding=vector,
            dimension=model.dimension,
            model_name=model.model_name,
        )
        for chunk, vector in zip(chunks, vectors)
    ]


def assert_compatible_embeddings(embedded_chunks: list[EmbeddedChunk]) -> None:
    """
    Guard against silently mixing vectors from different embedding
    backends. Same dimension does NOT mean compatible — a
    DeterministicHashingEmbeddingModel vector and a
    SentenceTransformerEmbeddingModel vector can both be length 384 while
    living in completely unrelated vector spaces (see
    DeterministicHashingEmbeddingModel's docstring). The only thing that
    actually guarantees comparability is having come from the exact same
    model_name.

    Raises ValueError if embedded_chunks contains vectors from more than
    one distinct model_name. vector_store.py (Phase 5) calls this before
    persisting new embeddings alongside any already-stored ones, and
    persists model_name as collection-level metadata so a later run
    using a different embedding model is caught immediately rather than
    silently corrupting retrieval quality.
    """
    if not embedded_chunks:
        return
    model_names = {e.model_name for e in embedded_chunks}
    if len(model_names) > 1:
        raise ValueError(
            "Cannot mix embeddings from different models in the same "
            f"batch/store: found {sorted(model_names)}. Vectors are only "
            "comparable when produced by the same embedding model, "
            "regardless of whether their dimensions happen to match."
        )
