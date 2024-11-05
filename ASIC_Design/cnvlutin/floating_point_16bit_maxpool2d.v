/*-------------------------------------------------------------------------
Maxpool2D module considering inputs based on half-precision IEEE-754 statndard
---------------------------------------------------------------------------*/

module floating_point_16bit_maxpool2d(in1, in2, in3, in4, out);

// Inputs and Output
input [15:0] in1;
input [15:0] in2;
input [15:0] in3;
input [15:0] in4;

output reg [15:0] out;
reg [15:0] temp1;
reg [15:0] temp2;

wire [10:0] mantissa_1;
wire [10:0] mantissa_2;
wire [10:0] mantissa_3;
wire [10:0] mantissa_4;

// STEP 1: To equate the exponents and shift the mantissas according to the maximum exponent
multiple_fp_alignment step_1 (in1, in2, in3, in4, mantissa_1, mantissa_2, mantissa_3, mantissa_4);

// STEP 2: To find the maximum comparing mantissa (The maxpool2d is placed after ReLU in CNVLUTIN, so all numbers are positive)
always @(*) begin

	if(mantissa_1 <= mantissa_2)
		temp1 <= in2;
	else
		temp1 <= in1;

	if(mantissa_3 <= mantissa_4)
		temp2 <= in4;
	else
		temp2 <= in3;
		
	if(temp1 <= temp2)
		out <= temp2;
	else
		out <= temp1;
end

endmodule