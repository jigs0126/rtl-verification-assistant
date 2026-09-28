# ALU Specification

## Overview

The ALU (`alu.sv`) is a 32-bit combinational Arithmetic Logic Unit supporting
five operations: ADD, SUB, AND, OR, XOR. It is purely combinational — there is
no clock, and `result` updates immediately whenever `a`, `b`, or `alu_control`
change.

## Ports

| Signal        | Direction | Width | Description                              |
|---------------|-----------|-------|-------------------------------------------|
| `a`           | input     | 32    | Operand A                                 |
| `b`           | input     | 32    | Operand B                                 |
| `alu_control` | input     | 3     | Operation select code                     |
| `result`      | output    | 32    | Result of the selected operation          |
| `zero`        | output    | 1     | High when `result` is all zero            |

## Width and Signedness Assumptions

- All operands and results are 32 bits wide.
- The ALU itself performs no sign extension — it treats `a` and `b` as raw
  32-bit bit vectors during arithmetic (`+`, `-`) and bitwise (`&`, `|`, `^`)
  operations. In SystemVerilog, `logic` vectors default to unsigned
  interpretation, so `a - b` for `a < b` wraps around using standard 32-bit
  two's-complement arithmetic rather than producing a signed negative value.
- Callers that need signed semantics must interpret `result` as
  two's-complement themselves. For example, if `a = 20` and `b = 10` and the
  operation is SUB but the operand order is reversed at the caller level, the
  ALU computes `10 - 20 = 4294967286` (`0xFFFFFFF6`), which is `-10` under
  32-bit two's-complement interpretation. This is expected ALU behavior, not
  a bug in the ALU itself — see `verification_plan.md` for how this is tested.

## Operation Encoding

See `operation_table.md` for the authoritative encoding table. Any
`alu_control` value outside the five defined operations is undefined: the RTL
currently drives `result` to the sentinel value `32'hDEADBEEF` in that case,
purely so undefined-opcode behavior is visible in simulation waveforms rather
than silently holding a stale value.

## Corner Cases to Consider

- Operands equal to zero (`a = 0` or `b = 0`)
- Operands equal to `32'hFFFFFFFF` (all ones)
- ADD overflow (`a + b` exceeding 32 bits, wraps silently — no overflow flag
  is implemented in this version)
- SUB underflow (`a < b`, wraps to a large unsigned value / two's-complement
  negative value)
- `alu_control` value outside `000`–`100` (undefined opcode)
- `a == b` for SUB (result should be exactly zero, `zero` flag should assert)
