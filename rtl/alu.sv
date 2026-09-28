// =============================================================================
// Module: alu
// Description: 32-bit Arithmetic Logic Unit supporting ADD, SUB, AND, OR, XOR.
//              Combinational design — result updates immediately with inputs.
// See: docs/alu_spec.md, docs/interface_description.md, docs/operation_table.md
// =============================================================================

module alu (
    input  logic [31:0] a,            // Operand A (unsigned/two's-complement, see spec)
    input  logic [31:0] b,            // Operand B (unsigned/two's-complement, see spec)
    input  logic [2:0]  alu_control,  // Operation select (see docs/operation_table.md)
    output logic [31:0] result,       // Operation result
    output logic        zero          // High when result == 32'd0
);

    // Operation encoding (must match docs/operation_table.md exactly)
    localparam logic [2:0] ALU_ADD = 3'b000;
    localparam logic [2:0] ALU_SUB = 3'b001;
    localparam logic [2:0] ALU_AND = 3'b010;
    localparam logic [2:0] ALU_OR  = 3'b011;
    localparam logic [2:0] ALU_XOR = 3'b100;

    always_comb begin
        case (alu_control)
            ALU_ADD: result = a + b;
            ALU_SUB: result = a - b;
            ALU_AND: result = a & b;
            ALU_OR:  result = a | b;
            ALU_XOR: result = a ^ b;
            default: result = 32'hDEADBEEF; // Undefined opcode sentinel — see spec §Corner Cases
        endcase
    end

    assign zero = (result == 32'd0);

endmodule
