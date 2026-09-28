# ALU Operation Table

This table is the authoritative source for `alu_control` encoding. It must
match `rtl/alu.sv` exactly — if the RTL changes, update this table in the
same commit.

| `alu_control` (binary) | `alu_control` (decimal) | Operation | Description            |
|-------------------------|--------------------------|-----------|-------------------------|
| `000`                    | 0                        | ADD       | `result = a + b`        |
| `001`                    | 1                        | SUB       | `result = a - b`        |
| `010`                    | 2                        | AND       | `result = a & b`        |
| `011`                    | 3                        | OR        | `result = a \| b`       |
| `100`                    | 4                        | XOR       | `result = a ^ b`        |
| `101`–`111`               | 5–7                      | UNDEFINED | `result = 32'hDEADBEEF` |

## Notes

- Only five of the eight possible 3-bit codes are defined operations.
- The undefined range (`101`–`111`) is not currently exercised by any
  verification plan requirement, but the ALU's defensive `default` case
  ensures it produces a recognizable sentinel value rather than X or a
  stale result, which makes undefined-opcode bugs easier to spot in
  simulation if this range is ever driven accidentally.
