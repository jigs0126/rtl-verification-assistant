"""
Metadata

What it is: builds the final Chunk objects — chunk text plus a
deterministic chunk_id and structured metadata — from a ParsedSection and
the chunk texts chunker.py produced for it.

Why required: retrieval is only useful if a retrieved chunk can be traced
back to where it came from. Without attached metadata (source file,
section, module, chunk_id), a retrieved chunk is anonymous text and the
UI/LLM prompt could not cite "Source: alu_spec.md" as required by §12/§31.

Input: a ParsedSection, plus the list[str] chunk texts chunker.py
produced for that section.
Output: list[Chunk]

How it connects to the next component: vector_store.py (Phase 5) will
store each Chunk's text, embedding, and chunk.metadata() dict together.

What would happen if it were removed: chunks would have no source
attribution, breaking every "show retrieved sources" requirement
downstream (UI, prompt construction, hallucination-control citations).
"""

from __future__ import annotations

import hashlib

from app.schemas.models import Chunk, ParsedSection


def _make_chunk_id(source: str, section: str, index: int) -> str:
    """
    Deterministic, human-readable chunk_id: <source>::<section-slug>::<index>::<hash>.
    The trailing hash disambiguates sections that produce identical slugs
    (e.g. two markdown headings with the same text in different files
    would still differ by source, but two same-named sections within one
    file are covered by the hash on the full source:section:index string).
    """
    digest = hashlib.md5(f"{source}:{section}:{index}".encode("utf-8")).hexdigest()[:8]
    section_slug = section.lower().replace(" ", "_").replace("/", "-")[:40]
    return f"{source}::{section_slug}::{index}::{digest}"


def build_chunks(section: ParsedSection, chunk_texts: list[str]) -> list[Chunk]:
    """Wrap each chunk text with a chunk_id and the section's metadata."""
    chunks: list[Chunk] = []
    for index, text in enumerate(chunk_texts):
        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(section.source, section.section, index),
                text=text,
                source=section.source,
                file_type=section.file_type,
                project=section.project,
                section=section.section,
                module=section.module,
            )
        )
    return chunks
