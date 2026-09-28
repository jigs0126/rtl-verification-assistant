"""
Ingestion Pipeline

What it is: the single function that runs a project's files through the
full ingestion sequence: load -> parse -> chunk -> attach metadata.

Why required: loader.py, parser.py, chunker.py, and metadata.py are each
independently testable, but something has to call them in order. Keeping
that orchestration in one small function (rather than inline in Phase 5's
vector-store indexing code, or in the Streamlit UI) keeps ingestion
usable and testable on its own, before any embedding/storage code exists.

Input: project_root path, project_name label, chunk_size, chunk_overlap.
Output: list[Chunk], ready to be embedded and stored (Phase 4/5).

How it connects to the next component: Phase 5's vector_store indexing
step will call ingest_project(...) and embed each returned Chunk's text.

What would happen if it were removed: Phase 5 would have to know how to
call loader/parser/chunker/metadata in the right order itself, coupling
storage code to ingestion internals.
"""

from __future__ import annotations

from pathlib import Path

from app.ingestion.chunker import chunk_section
from app.ingestion.loader import load_documents
from app.ingestion.metadata import build_chunks
from app.ingestion.parser import parse_document
from app.schemas.models import Chunk


def ingest_project(
    project_root: Path,
    project_name: str = "alu",
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Run the full ingestion sequence and return every chunk, in order."""
    chunks: list[Chunk] = []
    for doc in load_documents(project_root, project_name=project_name):
        for section in parse_document(doc):
            chunk_texts = chunk_section(section, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks.extend(build_chunks(section, chunk_texts))
    return chunks
