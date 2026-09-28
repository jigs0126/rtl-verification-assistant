import pytest

from app.ingestion.chunker import chunk_section
from app.schemas.models import ParsedSection


def _section(text: str) -> ParsedSection:
    return ParsedSection(
        source="alu_spec.md", file_type="markdown", project="alu", section="Corner Cases", text=text
    )


def test_short_text_returns_single_chunk():
    section = _section("short text under the chunk size")
    chunks = chunk_section(section, chunk_size=800, chunk_overlap=120)
    assert chunks == ["short text under the chunk size"]


def test_long_text_is_split_into_multiple_chunks():
    text = "\n".join(f"line {i} of the corner case discussion" for i in range(100))
    section = _section(text)
    chunks = chunk_section(section, chunk_size=200, chunk_overlap=40)

    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_chunks_respect_overlap_by_sharing_content():
    text = "\n".join(f"line {i} of the corner case discussion" for i in range(100))
    section = _section(text)
    chunks = chunk_section(section, chunk_size=200, chunk_overlap=40)

    # Consecutive chunks should share at least some trailing/leading text
    # because of the overlap window.
    first_tail = chunks[0][-20:]
    assert any(first_tail[:10] in chunks[1] for _ in [0]) or chunks[1].startswith(chunks[0][-40:][:10])


def test_empty_section_returns_no_chunks():
    section = _section("   \n\n   ")
    assert chunk_section(section, chunk_size=100, chunk_overlap=10) == []


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    section = _section("x" * 500)
    with pytest.raises(ValueError):
        chunk_section(section, chunk_size=100, chunk_overlap=100)


def test_negative_overlap_rejected():
    section = _section("x" * 500)
    with pytest.raises(ValueError):
        chunk_section(section, chunk_size=100, chunk_overlap=-1)


def test_all_text_is_covered_no_gaps():
    # Every line should appear in at least one chunk.
    lines = [f"UNIQUE_LINE_{i}" for i in range(50)]
    text = "\n".join(lines)
    section = _section(text)
    chunks = chunk_section(section, chunk_size=150, chunk_overlap=30)

    combined = "\n".join(chunks)
    for line in lines:
        assert line in combined
