Test Name:
alu_sub_failure_01

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
Testbench applied A = 20, B = 10 with alu_control set to the SUB encoding
(3'b001) per docs/operation_table.md. Expected a straightforward positive
subtraction result of 10. Observed result instead corresponds to -10 under
32-bit two's-complement interpretation, consistent with the ALU having
computed B - A rather than A - B.

Initial Engineer Observation:
Result value 0xFFFFFFF6 is exactly what (B - A) would produce, which suggests
possible operand ordering at the ALU input connections or in the testbench
driver itself — not yet confirmed which side is at fault. No waveform
capture attached to this report; investigation was based on final values
only.

Status:
Unconfirmed
