module multiple_fp_alignment(in_A, in_B, in_C, in_D, mantissa_A_out, mantissa_B_out, mantissa_C_out, mantissa_D_out);

input [15:0] in_A;
input [15:0] in_B;
input [15:0] in_C;
input [15:0] in_D;

output reg [10:0] mantissa_A_out;
output reg [10:0] mantissa_B_out;
output reg [10:0] mantissa_C_out;
output reg [10:0] mantissa_D_out;

wire [4:0] exponent_A;
wire [4:0] exponent_B;
wire [4:0] exponent_C;
wire [4:0] exponent_D;
wire [3:0] selection_bitmap;

// To decompose the exponents
assign exponent_A = in_A[14:10];
assign exponent_B = in_B[14:10];	
assign exponent_C = in_C[14:10];
assign exponent_D = in_D[14:10];

exponent_comparator exp_comparison_1 (exponent_A, exponent_B, exponent_C, exponent_D, selection_bitmap);

always @ (*) begin
	
	// To decompose the mantissa (Adding leading 1 to input A and B to normalize them
	if(in_A == 0)
		mantissa_A_out = {1'b0, in_A[9:0]};
	else
		mantissa_A_out = {1'b1, in_A[9:0]};
	if(in_B == 0)
		mantissa_B_out = {1'b0, in_B[9:0]};
	else
		mantissa_B_out = {1'b1, in_B[9:0]};
	if(in_C == 0)
		mantissa_C_out = {1'b0, in_C[9:0]};
	else
		mantissa_C_out = {1'b1, in_C[9:0]};
	if(in_D == 0)
		mantissa_D_out = {1'b0, in_D[9:0]};
	else
		mantissa_D_out = {1'b1, in_D[9:0]};

	// Mantissa alignment of A,B,C,D when A is the greater one
	if(selection_bitmap == 4'b0001) begin
		mantissa_A_out = mantissa_A_out;
		mantissa_B_out = mantissa_B_out >> (exponent_A - exponent_B);
		mantissa_C_out = mantissa_C_out >> (exponent_A - exponent_C);
		mantissa_D_out = mantissa_D_out >> (exponent_A - exponent_D);
		end
	// Mantissa alignment of A,B,C,D when B is the greater one
	else if(selection_bitmap == 4'b0010) begin
		mantissa_B_out = mantissa_B_out;
		mantissa_A_out = mantissa_A_out >> (exponent_B - exponent_A);
		mantissa_C_out = mantissa_C_out >> (exponent_B - exponent_C);
		mantissa_D_out = mantissa_D_out >> (exponent_B - exponent_D);
		end
	// Mantissa alignment of A,B,C,D when C is the greater one
	else if(selection_bitmap == 4'b0100) begin
		mantissa_C_out = mantissa_C_out;
		mantissa_A_out = mantissa_A_out >> (exponent_C - exponent_A);
		mantissa_B_out = mantissa_B_out >> (exponent_C - exponent_B);
		mantissa_D_out = mantissa_D_out >> (exponent_C - exponent_D);
		end
	// Mantissa alignment of A,B,C,D when D is the greater one
	else if(selection_bitmap == 4'b1000) begin
		mantissa_D_out = mantissa_D_out;
		mantissa_A_out = mantissa_A_out >> (exponent_D - exponent_A);
		mantissa_B_out = mantissa_B_out >> (exponent_D - exponent_B);
		mantissa_C_out = mantissa_C_out >> (exponent_D - exponent_C);
		end
	// If all exponents are equal
	else begin
		mantissa_A_out = mantissa_A_out;
		mantissa_B_out = mantissa_B_out;
		mantissa_C_out = mantissa_C_out;
		mantissa_D_out = mantissa_D_out;
		end
end

endmodule