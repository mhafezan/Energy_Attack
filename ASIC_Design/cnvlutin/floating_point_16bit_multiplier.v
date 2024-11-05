// 16-bit floating-point multiplier based on half-precision IEEE-754 statndard

module floating_point_16bit_multiplier(operand1, operand2, result, zero, overflow, underflow, denormalized, NaN, infinity);

input [15:0] operand1, operand2; // 16-bit Floating-point inputs formated as half-precision IEEE-754 statndard
output [15:0] result; // 16-bit Floating-point multiplication output
output zero, overflow, underflow, denormalized, NaN, infinity; // Output Flags

wire [9:0] mantissa_result;  // To extract final mantissa part (10-bits)
wire [5:0] exponent_result;  // To extract final exponent part (extended to 6-bits for overflow/underflow checking)
wire sign_result;

wire s1, s2; // To decompose input signs
wire [9:0] m1, m2; // To decompose input mantissas
wire [5:0] e1, e2, sum_e1_e2; // Extended to 6-bits to carry overflow
wire [21:0] mantissa_product; // RAW production of mantissas (by considering one leading 1 for each mantissa, resulting to 22-bits output)

assign s1 = operand1[15];
assign e1 = {1'b0, operand1[14:10]};
assign m1 = operand1[9:0];

assign s2 = operand2[15];
assign e2 = {1'b0, operand2[14:10]};
assign m2 = operand2[9:0];

// STEP1: FLAGS
assign zero = (operand1[14:0]==15'd0 || operand2[14:0]==15'd0) ? 1'b1 : 1'b0;
assign NaN = (&operand1[14:10] & |operand1[9:0]) | (&operand2[14:10] & |operand2[9:0]);
assign infinity = (&operand1[14:10] & ~|operand1[9:0]) | (&operand2[14:10] & ~|operand2[9:0]);
assign overflow = ((exponent_result > 6'd30) && !NaN && !infinity) ? 1'b1 : 1'b0;
assign underflow = ((exponent_result < 6'd1) && !NaN && !infinity) ? 1'b1 : 1'b0;
assign denormalized = (exponent_result == 6'd0 && mantissa_result != 10'd0) ? 1'b1 : 1'b0;

// STEP2: Add the biased exponents of the two numbers, subtracting the bias from the sum to get the new biased exponent
assign sum_e1_e2 = (e1 + e2) - 6'd15;

// STEP3: Multiply the significands (Normalized version of the mantissas should be considered by adding leading 1 to each mantissa)
assign mantissa_product= {1'b1,m1} * {1'b1,m2};

// STEP4: Normalize the product if needed, shifting it right and incrementing the exponent
multplication_normalizer norm (.man_product(mantissa_product), .exp_sum(sum_e1_e2), .man_res(mantissa_result), .exp_res(exponent_result));
/*
always @(*) begin // Normalized
	if (mantissa_product[21]==1) begin
		mantissa_result = mantissa_product[20:11];
		exponent_result = sum_e1_e2 + 6'd1;
	end
	else begin // Not Normalized
		mantissa_result = mantissa_product[19:10];
		exponent_result = sum_e1_e2;
	end
end
*/

// STEP5: Set the sign of the product to positive if the signs of the original operands are the same; if they differ make the sign negative
assign sign_result = s1 ^ s2;

// STEP6: Truncate output to 16-bit considering flags' results
assign result = (zero==1'b1) ? (16'd0) :
					 ((NaN==1'b1) ? ({sign_result, 5'b11111, mantissa_result}) :
			  ((infinity==1'b1) ? ({sign_result, 5'b11111, 10'd0}) : ({sign_result, exponent_result[4:0], mantissa_result})));

endmodule