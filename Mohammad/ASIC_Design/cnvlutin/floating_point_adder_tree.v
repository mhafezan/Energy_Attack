module floating_point_adder_tree (in01, in02, in03, in04, in05, in06, in07, in08, in09, in10, in11, in12, in13, in14, in15, in16, in_nbout, out_sum);

input [15:0] in01, in02, in03, in04, in05, in06, in07, in08, in09, in10, in11, in12, in13, in14, in15, in16, in_nbout;
output [15:0] out_sum;

wire [15:0] result_adder_11;
wire [15:0] result_adder_12;
wire [15:0] result_adder_13;
wire [15:0] result_adder_14;
wire [15:0] result_adder_15;
wire [15:0] result_adder_16;
wire [15:0] result_adder_17;
wire [15:0] result_adder_18;
wire [15:0] result_adder_21;
wire [15:0] result_adder_22;
wire [15:0] result_adder_23;
wire [15:0] result_adder_24;
wire [15:0] result_adder_31;
wire [15:0] result_adder_32;
wire [15:0] result_adder_41;

// Level 1
floating_point_16bit_adder adder_01 (.operand1(in01), .operand2(in02), .sum(result_adder_11));
floating_point_16bit_adder adder_02 (.operand1(in03), .operand2(in04), .sum(result_adder_12));
floating_point_16bit_adder adder_03 (.operand1(in05), .operand2(in06), .sum(result_adder_13));
floating_point_16bit_adder adder_04 (.operand1(in07), .operand2(in08), .sum(result_adder_14));
floating_point_16bit_adder adder_05 (.operand1(in09), .operand2(in10), .sum(result_adder_15));
floating_point_16bit_adder adder_06 (.operand1(in11), .operand2(in12), .sum(result_adder_16));
floating_point_16bit_adder adder_07 (.operand1(in13), .operand2(in14), .sum(result_adder_17));
floating_point_16bit_adder adder_08 (.operand1(in15), .operand2(in16), .sum(result_adder_18));

// Level 2
floating_point_16bit_adder adder_09 (.operand1(result_adder_11), .operand2(result_adder_12), .sum(result_adder_21));
floating_point_16bit_adder adder_10 (.operand1(result_adder_13), .operand2(result_adder_14), .sum(result_adder_22));
floating_point_16bit_adder adder_11 (.operand1(result_adder_15), .operand2(result_adder_16), .sum(result_adder_23));
floating_point_16bit_adder adder_12 (.operand1(result_adder_17), .operand2(result_adder_18), .sum(result_adder_24));

// Level 3
floating_point_16bit_adder adder_13 (.operand1(result_adder_21), .operand2(result_adder_22), .sum(result_adder_31));
floating_point_16bit_adder adder_14 (.operand1(result_adder_23), .operand2(result_adder_24), .sum(result_adder_32));

// Level 4
floating_point_16bit_adder adder_15 (.operand1(result_adder_31), .operand2(result_adder_32), .sum(result_adder_41));

// Level 5
floating_point_16bit_adder adder_16 (.operand1(result_adder_41), .operand2(in_nbout), .sum(out_sum));

endmodule