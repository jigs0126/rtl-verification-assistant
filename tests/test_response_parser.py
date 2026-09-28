import pytest

from app.llm.client import LLMError
from app.llm.response_parser import (
    extract_code_block,
    parse_comma_list,
    parse_labeled_sections,
    parse_numbered_list,
)

SAMPLE_RESPONSE = """\
TEST_SCENARIOS:
1. ADD normal values
2. SUB underflow

ASSUMPTIONS:
1. Operands are 32-bit unsigned

WARNINGS:
1. Generated artifact — not simulation-verified.

TESTBENCH_CODE:
```systemverilog
module tb;
  initial $display("hi");
endmodule
```
"""


def test_parse_labeled_sections_extracts_all_sections():
    sections = parse_labeled_sections(
        SAMPLE_RESPONSE, ["TEST_SCENARIOS", "ASSUMPTIONS", "WARNINGS", "TESTBENCH_CODE"]
    )
    assert "1. ADD normal values" in sections["TEST_SCENARIOS"]
    assert "module tb;" in sections["TESTBENCH_CODE"]


def test_parse_labeled_sections_raises_on_missing_required_section():
    with pytest.raises(LLMError):
        parse_labeled_sections(SAMPLE_RESPONSE, ["TEST_SCENARIOS", "NONEXISTENT_SECTION"])


def test_parse_labeled_sections_raises_on_unstructured_text():
    with pytest.raises(LLMError):
        parse_labeled_sections("just some plain text with no headers", ["TEST_SCENARIOS"])


def test_parse_numbered_list_extracts_items():
    items = parse_numbered_list("1. first item\n2. second item\n3. third item")
    assert items == ["first item", "second item", "third item"]


def test_parse_numbered_list_handles_none_response():
    assert parse_numbered_list("None.") == []
    assert parse_numbered_list("None identified.") == []


def test_parse_numbered_list_handles_empty_string():
    assert parse_numbered_list("") == []
    assert parse_numbered_list("   ") == []


def test_parse_numbered_list_falls_back_to_lines_without_numbers():
    items = parse_numbered_list("first line\nsecond line")
    assert items == ["first line", "second line"]


def test_extract_code_block_returns_code_contents():
    code = extract_code_block("```systemverilog\nmodule tb;\nendmodule\n```")
    assert code == "module tb;\nendmodule"


def test_extract_code_block_raises_when_no_fence_present():
    with pytest.raises(LLMError):
        extract_code_block("no code block here at all")


def test_parse_comma_list_extracts_items():
    assert parse_comma_list("a, b, c") == ["a", "b", "c"]


def test_parse_comma_list_handles_none_identified():
    assert parse_comma_list("None identified.") == []
