module Extraction(in_A, in_B, sign_A, sign_B, mantissa_A_out, mantissa_B_out, exponent_out);

input [15:0] in_A;
input [15:0] in_B;
output reg sign_A;
output reg sign_B;
output reg [10:0] mantissa_A_out;
output reg [10:0] mantissa_B_out;
output reg [4:0] exponent_out;

reg [4:0] exponent_A;
reg [4:0] exponent_B;

always @ (*) begin
	// To decompose the sign and exponents
	sign_A = in_A[15];
	sign_B = in_B[15];
	exponent_A = in_A[14:10];
	exponent_B = in_B[14:10];	
	
	// To decompose the mantissa (Adding leading 1 to input A and B to normalize them
	if(in_A == 0)
		mantissa_A_out = {1'b0, in_A[9:0]};
	else
		mantissa_A_out = {1'b1, in_A[9:0]};
	if(in_B == 0)
		mantissa_B_out = {1'b0, in_B[9:0]};
	else
		mantissa_B_out = {1'b1, in_B[9:0]};

	// Mantissa alignment of B when A is the greater one
	if(exponent_A > exponent_B) begin
		mantissa_A_out = mantissa_A_out;
		mantissa_B_out = mantissa_B_out >> (exponent_A - exponent_B);
		exponent_out = exponent_A;
		end
	// Mantissa alignment of A when B is the greater one
	else if(exponent_B > exponent_A) begin
		mantissa_B_out = mantissa_B_out;
		mantissa_A_out = mantissa_A_out >> (exponent_B - exponent_A);
		exponent_out = exponent_B;
		end
	// If both exponents are equal
	else begin
		mantissa_A_out = mantissa_A_out;
		mantissa_B_out = mantissa_B_out; 
			if ((sign_A != sign_B) && (mantissa_A_out == mantissa_B_out)) // And if both mantissa are equal but the signs are different
				exponent_out = 0;
			else
				exponent_out = exponent_A;
		end
	
end

endmodule