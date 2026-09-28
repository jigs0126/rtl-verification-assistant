# ALU Verification Plan

## Objective

Verify that `alu.sv` correctly implements all five defined operations
(ADD, SUB, AND, OR, XOR) across normal values and documented corner cases,
and that the `zero` flag asserts correctly.

## Required Test Scenarios

### ADD
1. Normal values (e.g. `a = 15`, `b = 27`)
2. `a = 0` or `b = 0` (identity case)
3. Operands near `32'hFFFFFFFF` to exercise wraparound (overflow is not
   flagged in this version — result simply wraps per unsigned 32-bit
   arithmetic)

### SUB
1. Normal values where `a > b` (positive result)
2. `a == b` (result must be exactly zero; `zero` flag must assert)
3. `a < b` — underflow/wraparound case. Result is the 32-bit two's-complement
   representation of the negative value. Example: `a = 10`, `b = 20` →
   `result = 4294967286` (`0xFFFFFFF6`, i.e. `-10`).

### AND / OR / XOR
1. Alternating bit patterns (e.g. `32'hAAAAAAAA` vs `32'h55555555`)
2. All-zero operand paired with a non-zero operand
3. All-one operand (`32'hFFFFFFFF`) paired with a normal value
4. Identical operands (useful for XOR — result must be exactly zero)

### Undefined Opcode
1. Drive `alu_control` to a value in `101`–`111` and confirm `result` reads
   the sentinel `32'hDEADBEEF`.

## Out of Scope for This Verification Plan

- No overflow/carry flag exists in this ALU version — do not write tests
  expecting one.
- No timing/setup-hold checks — the ALU is combinational with no clock.
- No formal verification — this plan targets simulation-based
  (directed + corner-case) testing only.
