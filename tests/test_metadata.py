from app.ingestion.metadata import build_chunks
from app.schemas.models import Chunk, ParsedSection


def _section() -> ParsedSection:
    return ParsedSection(
        source="alu_spec.md",
        file_type="markdown",
        project="alu",
        section="Corner Cases",
        text="full section text",
        module=None,
    )


def test_build_chunks_produces_one_chunk_per_text():
    section = _section()
    chunks = build_chunks(section, ["chunk one", "chunk two"])
    assert len(chunks) == 2
    assert chunks[0].text == "chunk one"
    assert chunks[1].text == "chunk two"


def test_chunk_ids_are_unique_and_deterministic():
    section = _section()
    chunks_a = build_chunks(section, ["chunk one", "chunk two"])
    chunks_b = build_chunks(section, ["chunk one", "chunk two"])

    ids_a = [c.chunk_id for c in chunks_a]
    assert len(set(ids_a)) == len(ids_a)  # unique within a run
    assert ids_a == [c.chunk_id for c in chunks_b]  # deterministic across runs


def test_chunk_id_contains_source_and_index():
    section = _section()
    chunks = build_chunks(section, ["only chunk"])
    assert chunks[0].chunk_id.startswith("alu_spec.md::corner_cases::0::")


def test_chunk_carries_full_metadata():
    section = ParsedSection(
        source="alu.sv", file_type="systemverilog", project="alu", section="always_block[alu_control]",
        text="case (alu_control) ... endcase", module="alu",
    )
    chunks = build_chunks(section, [section.text])
    chunk = chunks[0]

    metadata = chunk.metadata()
    assert metadata["source"] == "alu.sv"
    assert metadata["file_type"] == "systemverilog"
    assert metadata["project"] == "alu"
    assert metadata["section"] == "always_block[alu_control]"
    assert metadata["module"] == "alu"
    assert metadata["chunk_id"] == chunk.chunk_id


def test_chunk_schema_defaults_module_to_none():
    chunk = Chunk(
        chunk_id="x", text="t", source="s", file_type="markdown", project="alu", section="sec"
    )
    assert chunk.module is None
    assert chunk.metadata()["module"] == ""
