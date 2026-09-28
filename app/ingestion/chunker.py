"""
Chunker

What it is: splits a ParsedSection's text into retrieval-sized chunks,
using configurable chunk_size / chunk_overlap (from settings.py).

Why required: even a structurally sound section can be too large to
embed as one meaningful retrieval unit, or too large to include wholesale
in an LLM prompt. Chunking creates smaller, overlapping units so that
relevant content isn't lost at an arbitrary cut point, while keeping
each chunk small enough for precise top_k retrieval.

Input: a ParsedSection's text, plus chunk_size and chunk_overlap.
Output: list[str] — raw chunk texts (metadata.py attaches IDs next).

How it connects to the next component: metadata.py wraps each chunk text
with source/section/chunk_id info to produce Chunk objects; embedder.py
(Phase 4) will embed those Chunk objects next.

What would happen if it were removed: whole sections would have to be
embedded and retrieved as single units, which defeats the purpose of
top_k retrieval — a query would always pull back an entire section
(sometimes an entire file) instead of the specific relevant passage.
"""

from __future__ import annotations

from app.schemas.models import ParsedSection


def chunk_section(section: ParsedSection, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split section.text into chunks of at most chunk_size characters, with
    chunk_overlap characters of overlap between consecutive chunks. When a
    chunk boundary would fall mid-line, it is pulled back to the nearest
    preceding newline (as long as that doesn't shrink the chunk below half
    of chunk_size), so chunks don't split a single RTL/markdown line in two.
    """
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = section.text.strip("\n")
    if not text.strip():
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            newline_pos = text.rfind("\n", start, end)
            if newline_pos != -1 and newline_pos > start + (chunk_size // 2):
                end = newline_pos
        chunk_text = text[start:end].strip("\n")
        if chunk_text:
            chunks.append(chunk_text)
        if end >= length:
            break
        start = end - chunk_overlap

    return chunks
