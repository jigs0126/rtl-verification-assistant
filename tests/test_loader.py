from pathlib import Path

from app.ingestion.loader import load_documents

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_documents_finds_expected_files():
    docs = load_documents(PROJECT_ROOT, project_name="alu")
    sources = {d.source for d in docs}

    assert "alu.sv" in sources
    assert "alu_spec.md" in sources
    assert "operation_table.md" in sources
    assert "verification_plan.md" in sources
    assert "interface_description.md" in sources
    assert "alu_sub_failure.md" in sources
    assert "example_failure_report.md" in sources


def test_load_documents_assigns_correct_file_types():
    docs = load_documents(PROJECT_ROOT, project_name="alu")
    by_source = {d.source: d for d in docs}

    assert by_source["alu.sv"].file_type == "systemverilog"
    assert by_source["alu_spec.md"].file_type == "markdown"


def test_load_documents_skips_unsupported_extensions(tmp_path):
    rtl_dir = tmp_path / "rtl"
    rtl_dir.mkdir()
    (rtl_dir / "alu.sv").write_text("module alu; endmodule", encoding="utf-8")
    (rtl_dir / "notes.docx").write_text("binary-ish content", encoding="utf-8")

    docs = load_documents(tmp_path, project_name="alu")

    sources = {d.source for d in docs}
    assert "alu.sv" in sources
    assert "notes.docx" not in sources


def test_load_documents_handles_missing_subdir(tmp_path):
    # No rtl/docs/bugs/examples directories exist at all.
    docs = load_documents(tmp_path, project_name="alu")
    assert docs == []


def test_load_documents_sets_project_name():
    docs = load_documents(PROJECT_ROOT, project_name="rv_cordix")
    assert all(d.project == "rv_cordix" for d in docs)
