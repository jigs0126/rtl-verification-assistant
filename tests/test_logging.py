import json
from pathlib import Path

from app.utils.logging import log_query, read_recent_queries
from app.schemas.models import Chunk, RetrievedChunk, RetrievedContext


def _context() -> RetrievedContext:
    chunk = Chunk(
        chunk_id="alu.sv::x::0::abc", text="case (alu_control) ...", source="alu.sv",
        file_type="systemverilog", project="alu", section="always_block[alu_control]", module="alu",
    )
    return RetrievedContext(query="SUB operation", results=[RetrievedChunk(chunk=chunk, score=0.12, rank=1)], top_k=1)


def test_log_query_writes_one_json_line(tmp_path):
    log_query(tmp_path, "testbench", "generate SUB testbench", _context(), "claude-sonnet-4-6", "success")
    log_file = tmp_path / "query_log.jsonl"
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["mode"] == "testbench"
    assert record["query"] == "generate SUB testbench"
    assert record["retrieved_sources"] == ["alu.sv"]
    assert record["llm_model"] == "claude-sonnet-4-6"
    assert record["status"] == "success"
    assert "timestamp" in record


def test_log_query_never_includes_api_key_field():
    # The record shape itself has no field capable of holding a key —
    # this pins that contract down so a future edit can't accidentally
    # add one.
    import inspect
    from app.utils import logging as logging_module
    source = inspect.getsource(logging_module.log_query)
    assert "api_key" not in source.lower()


def test_log_query_handles_none_context(tmp_path):
    log_query(tmp_path, "debug", "some query", None, "claude-sonnet-4-6", "error: empty corpus")
    records = read_recent_queries(tmp_path)
    assert len(records) == 1
    assert records[0]["retrieved_sources"] == []
    assert records[0]["status"] == "error: empty corpus"


def test_read_recent_queries_returns_empty_when_no_log(tmp_path):
    assert read_recent_queries(tmp_path) == []


def test_read_recent_queries_respects_limit(tmp_path):
    for i in range(5):
        log_query(tmp_path, "testbench", f"query {i}", _context(), "claude-sonnet-4-6", "success")
    records = read_recent_queries(tmp_path, limit=2)
    assert len(records) == 2
    assert records[-1]["query"] == "query 4"


def test_log_query_never_raises_on_bad_path(tmp_path):
    # A path that can't be created (a regular file blocking the directory)
    # must not raise -- logging failures must never break the caller.
    # Uses pytest's tmp_path so it works on Windows, Linux and macOS.
    blocking_file = tmp_path / "not_a_directory_marker_for_test"
    blocking_file.write_text("x")
    log_query(blocking_file / "logs", "testbench", "q", None, "model", "success")
