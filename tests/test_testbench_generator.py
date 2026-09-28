import pytest

from app.embeddings.embedder import DeterministicHashingEmbeddingModel, embed_chunks
from app.llm.client import FakeLLMClient, LLMError
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore
from app.schemas.models import Chunk
from app.schemas.verification import TestbenchRequest
from app.verification.testbench_generator import TestbenchGenerator

FAKE_RESPONSE = """\
TEST_SCENARIOS:
1. ADD normal values
2. SUB underflow wraps to two's complement

ASSUMPTIONS:
1. Operands are treated as 32-bit unsigned bit vectors

WARNINGS:
1. Generated artifact — not simulation-verified.

TESTBENCH_CODE:
```systemverilog
module alu_tb;
  logic [31:0] a, b, result;
  logic [2:0] alu_control;
  logic zero;

  alu dut (.a(a), .b(b), .alu_control(alu_control), .result(result), .zero(zero));

  initial begin
    a = 32'd20; b = 32'd10; alu_control = 3'b001; #1;
    $display("result=%0d", result);
  end
endmodule
```
"""


def _retriever(tmp_path) -> Retriever:
    store = VectorStore(tmp_path / "vector_store", collection_name="tb_gen_test")
    model = DeterministicHashingEmbeddingModel()
    chunk = Chunk(
        chunk_id="alu.sv::always_block[alu_control]::0::x",
        text="case (alu_control) ALU_SUB: result = a - b; endcase",
        source="alu.sv", file_type="systemverilog", project="alu",
        section="always_block[alu_control]", module="alu",
    )
    store.add_embeddings(embed_chunks([chunk], model))
    return Retriever(store, model, default_top_k=4)


def test_generate_returns_parsed_testbench_result(tmp_path):
    retriever = _retriever(tmp_path)
    llm = FakeLLMClient(fixed_response=FAKE_RESPONSE)
    generator = TestbenchGenerator(retriever, llm)

    result = generator.generate(TestbenchRequest(module="alu", requirement="cover SUB underflow"))

    assert "module alu_tb;" in result.testbench_code
    assert "ADD normal values" in result.test_scenarios
    assert "SUB underflow wraps to two's complement" in result.test_scenarios
    assert any("32-bit unsigned" in a for a in result.assumptions)
    assert "alu.sv" in result.retrieved_sources


def test_generate_always_includes_not_verified_warning_even_if_llm_omits_it(tmp_path):
    retriever = _retriever(tmp_path)
    response_without_warning = FAKE_RESPONSE.replace(
        "WARNINGS:\n1. Generated artifact — not simulation-verified.", "WARNINGS:\nNone."
    )
    llm = FakeLLMClient(fixed_response=response_without_warning)
    generator = TestbenchGenerator(retriever, llm)

    result = generator.generate(TestbenchRequest(module="alu", requirement="cover ADD"))
    assert "Generated artifact — not simulation-verified." in result.warnings


def test_generate_rejects_empty_requirement(tmp_path):
    retriever = _retriever(tmp_path)
    llm = FakeLLMClient(fixed_response=FAKE_RESPONSE)
    generator = TestbenchGenerator(retriever, llm)

    with pytest.raises(ValueError):
        generator.generate(TestbenchRequest(module="alu", requirement=""))


def test_generate_raises_llm_error_on_malformed_response(tmp_path):
    retriever = _retriever(tmp_path)
    llm = FakeLLMClient(fixed_response="not a structured response at all")
    generator = TestbenchGenerator(retriever, llm)

    with pytest.raises(LLMError):
        generator.generate(TestbenchRequest(module="alu", requirement="cover ADD"))


def test_generate_raises_when_code_block_missing(tmp_path):
    retriever = _retriever(tmp_path)
    broken_response = FAKE_RESPONSE.split("TESTBENCH_CODE:")[0] + "TESTBENCH_CODE:\nno code here"
    llm = FakeLLMClient(fixed_response=broken_response)
    generator = TestbenchGenerator(retriever, llm)

    with pytest.raises(LLMError):
        generator.generate(TestbenchRequest(module="alu", requirement="cover ADD"))
