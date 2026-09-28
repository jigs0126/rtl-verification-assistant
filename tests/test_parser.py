from app.ingestion.parser import parse_document
from app.schemas.models import RawDocument

ALU_RTL = """\
// header comment line 1
// header comment line 2
module alu (
    input  logic [31:0] a,
    input  logic [31:0] b,
    input  logic [2:0]  alu_control,
    output logic [31:0] result,
    output logic        zero
);

    localparam logic [2:0] ALU_ADD = 3'b000;
    localparam logic [2:0] ALU_SUB = 3'b001;

    always_comb begin
        case (alu_control)
            ALU_ADD: result = a + b;
            ALU_SUB: result = a - b;
            default: result = 32'hDEADBEEF;
        endcase
    end

    assign zero = (result == 32'd0);

endmodule
"""


def _rtl_doc(content: str = ALU_RTL) -> RawDocument:
    return RawDocument(
        path="rtl/alu.sv",
        source="alu.sv",
        file_type="systemverilog",
        project="alu",
        content=content,
    )


def test_rtl_parser_extracts_module_name():
    sections = parse_document(_rtl_doc())
    assert all(s.module == "alu" for s in sections)


def test_rtl_parser_produces_expected_section_labels():
    sections = parse_document(_rtl_doc())
    labels = [s.section for s in sections]

    assert "header_comments" in labels
    assert "module_declaration" in labels
    assert "parameter_declarations" in labels
    assert any(l.startswith("always_block") for l in labels)
    assert "continuous_assignments" in labels
    assert "endmodule" in labels


def test_rtl_parser_module_declaration_contains_full_port_list():
    sections = parse_document(_rtl_doc())
    decl = next(s for s in sections if s.section == "module_declaration")

    assert "input  logic [31:0] a" in decl.text
    assert "output logic        zero" in decl.text
    assert decl.text.strip().endswith(");")


def test_rtl_parser_always_block_labeled_with_case_target():
    sections = parse_document(_rtl_doc())
    always_sections = [s for s in sections if s.section.startswith("always_block")]

    assert len(always_sections) == 1
    assert always_sections[0].section == "always_block[alu_control]"
    assert "endcase" in always_sections[0].text


def test_rtl_parser_does_not_split_case_statement_across_sections():
    sections = parse_document(_rtl_doc())
    always_section = next(s for s in sections if s.section.startswith("always_block"))

    assert "ALU_ADD: result = a + b;" in always_section.text
    assert "ALU_SUB: result = a - b;" in always_section.text
    assert "default: result = 32'hDEADBEEF;" in always_section.text


def test_markdown_parser_splits_on_headings():
    doc = RawDocument(
        path="docs/alu_spec.md",
        source="alu_spec.md",
        file_type="markdown",
        project="alu",
        content="# ALU Specification\n\nIntro text.\n\n## Ports\n\nPort table here.\n\n## Corner Cases\n\nSome cases.\n",
    )
    sections = parse_document(doc)
    headings = [s.section for s in sections]

    assert "ALU Specification" in headings
    assert "Ports" in headings
    assert "Corner Cases" in headings

    ports_section = next(s for s in sections if s.section == "Ports")
    assert "Port table here." in ports_section.text


def test_text_parser_returns_single_section():
    doc = RawDocument(
        path="notes.txt", source="notes.txt", file_type="text", project="alu", content="plain notes"
    )
    sections = parse_document(doc)
    assert len(sections) == 1
    assert sections[0].text == "plain notes"
