"""
Document Loader

What it is: reads files out of the project's rtl/, docs/, bugs/, and
examples/ directories and turns each into a RawDocument.

Why required: everything downstream (parsing, chunking, embedding) needs
a uniform starting point regardless of whether the underlying file is
SystemVerilog, Verilog, Markdown, or plain text.

Input: a project root directory (settings.documents_path in later phases).
Output: list[RawDocument].

How it connects to the next component: parser.py consumes these
RawDocument objects next, choosing an RTL-aware or markdown-aware parse
strategy based on `file_type`.

What would happen if it were removed: nothing downstream would have any
file contents to work with — there would be no ingestion pipeline at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from app.schemas.models import RawDocument

# Extension -> file_type label used throughout the pipeline.
SUPPORTED_EXTENSIONS = {
    ".sv": "systemverilog",
    ".v": "verilog",
    ".md": "markdown",
    ".txt": "text",
}

DEFAULT_SUBDIRS = ("rtl", "docs", "bugs", "examples")


def file_type_for(path: Path) -> Optional[str]:
    """Return the ingestion file_type for a path, or None if unsupported."""
    return SUPPORTED_EXTENSIONS.get(path.suffix.lower())


def load_documents(
    project_root: Path,
    project_name: str = "alu",
    subdirs: Iterable[str] = DEFAULT_SUBDIRS,
) -> list[RawDocument]:
    """
    Walk the given subdirectories of project_root and load every
    supported file into a RawDocument. Unsupported extensions are
    skipped silently (this matches §23's "unsupported file types" error
    handling — skipping is not an error, it's expected for e.g. .gitkeep).

    Files are returned sorted by (subdir, filename) for reproducible
    ordering, which makes chunk_id generation deterministic run to run.
    """
    project_root = Path(project_root)
    documents: list[RawDocument] = []

    for subdir in subdirs:
        directory = project_root / subdir
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            file_type = file_type_for(path)
            if file_type is None:
                continue
            content = path.read_text(encoding="utf-8")
            documents.append(
                RawDocument(
                    path=str(path),
                    source=path.name,
                    file_type=file_type,
                    project=project_name,
                    content=content,
                )
            )
    return documents
