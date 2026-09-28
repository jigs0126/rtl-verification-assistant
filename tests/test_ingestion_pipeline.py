from pathlib import Path

from app.ingestion.pipeline import ingest_project

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_ingest_project_produces_chunks_from_real_files():
    chunks = ingest_project(PROJECT_ROOT, project_name="alu", chunk_size=800, chunk_overlap=120)
    assert len(chunks) > 0

    sources = {c.source for c in chunks}
    assert "alu.sv" in sources
    assert "alu_spec.md" in sources


def test_ingest_project_alu_rtl_chunks_are_structurally_labeled():
    chunks = ingest_project(PROJECT_ROOT, project_name="alu", chunk_size=800, chunk_overlap=120)
    alu_chunks = [c for c in chunks if c.source == "alu.sv"]
    sections = {c.section for c in alu_chunks}

    assert "module_declaration" in sections
    assert any(s.startswith("always_block") for s in sections)
    assert all(c.module == "alu" for c in alu_chunks)


def test_ingest_project_chunk_ids_are_all_unique():
    chunks = ingest_project(PROJECT_ROOT, project_name="alu", chunk_size=800, chunk_overlap=120)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_ingest_project_respects_chunk_size():
    chunks = ingest_project(PROJECT_ROOT, project_name="alu", chunk_size=300, chunk_overlap=50)
    assert all(len(c.text) <= 300 for c in chunks)
