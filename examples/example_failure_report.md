Test Name:
alu_sub_01

Module:
alu

Operation:
SUB

Inputs:
A = 20
B = 10

Expected Result:
10

Observed Result:
4294967286
0xFFFFFFF6

Failure Type:
Result mismatch

Additional Context:
Directed test from verification_plan.md, SUB scenario "normal values where
a > b". Testbench drove alu_control to 3'b001 (SUB per operation_table.md).

Initial Engineer Observation:
Observed value matches -10 under 32-bit two's-complement interpretation.
Not yet determined whether this is an operand-ordering issue, a control
decoding issue, or a testbench connectivity issue.

Status:
Unconfirmed
