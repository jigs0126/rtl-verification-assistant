# ALU Interface Description

## Module Declaration

```systemverilog
module alu (
    input  logic [31:0] a,
    input  logic [31:0] b,
    input  logic [2:0]  alu_control,
    output logic [31:0] result,
    output logic        zero
);
```

## Port-by-Port Description

- **`a` (input, 32 bits)** — First operand. Presented as a raw bit vector;
  interpretation (signed/unsigned) is determined by the operation and by the
  caller, not by the ALU itself.
- **`b` (input, 32 bits)** — Second operand. Same interpretation rules as `a`.
- **`alu_control` (input, 3 bits)** — Selects which operation the ALU
  performs this cycle. Encoding is defined in `operation_table.md`.
- **`result` (output, 32 bits)** — Combinational output of the selected
  operation on `a` and `b`.
- **`zero` (output, 1 bit)** — Convenience flag, asserted whenever `result`
  is exactly `32'd0`. Useful for branch-comparison use cases (e.g. `BEQ` in a
  processor datapath), even though this standalone ALU project does not
  exercise that use case directly.

## Timing

The ALU is fully combinational (`always_comb`). There is no clock or reset
port. `result` and `zero` reflect the current values of `a`, `b`, and
`alu_control` with only gate propagation delay — no registered latency.

## Instantiation Example

```systemverilog
alu u_alu (
    .a           (operand_a),
    .b           (operand_b),
    .alu_control (opcode_select),
    .result      (alu_result),
    .zero        (alu_zero)
);
```
