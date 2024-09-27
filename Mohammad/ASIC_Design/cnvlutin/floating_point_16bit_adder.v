/*-------------------------------------------------------------------------
16-bit floating-point multiplier based on half-precision IEEE-754 statndard
---------------------------------------------------------------------------*/
module floating_point_16bit_adder (operand1, operand2, sum, overflow, underflow, infinity, NaN);

	input [15:0] operand1;
	input [15:0] operand2;
	output [15:0] sum;
	output overflow, underflow, infinity, NaN;

	wire sign_A;
	wire sign_B;
	wire [10:0] mantissa_A;
	wire [10:0] mantissa_B;
	wire [4:0] exponent_shared;

	wire [11:0] mantissa_sum;
	wire sign_result;

	wire shift_direction;
	wire [3:0] shift_number;
	
	wire [5:0] exponent_result;
	wire [9:0] mantissa_result;
	
	// To set output Flags (i.e., overflow and underflow in exponent result, infinity and NaN in inputs)
	assign overflow = (exponent_result > 6'd30 && !NaN && !infinity) ? 1'b1 : 1'b0;
	assign underflow = (exponent_result < 6'd1 && !NaN && !infinity) ? 1'b1 : 1'b0;
	assign NaN = (&operand1[14:10] & |operand1[9:0]) | (&operand2[14:10] & |operand2[9:0]);
	assign infinity = (&operand1[14:10] & ~|operand1[9:0]) | (&operand2[14:10] & ~|operand2[9:0]);

   // Stage 1: To determine the number of required shifts, shift the mantissa, select the reference exponent, and return decomposition
	Extraction phase1 (operand1, operand2, sign_A, sign_B, mantissa_A, mantissa_B, exponent_shared);
   
	// Stage 2: To add two significands (based on the signs of operands)
	Addition phase2 (sign_A, sign_B, mantissa_A, mantissa_B, sign_result, mantissa_sum);
   
	// Stage 3:	To specify the amount of required shifts and the corresponding direction to normalize the result in the next step.
	Normalization1 phase3 (mantissa_sum, shift_direction, shift_number);
	
   // Stage 4: To normalize the results (by shifting the mantissa to the specified direction and specified amount)
	//          It adjusts the exponent accordingly
	Normalization2 phase4 (exponent_shared, mantissa_sum, shift_direction, shift_number, exponent_result, mantissa_result);
	
	// Stage 5: Framing and truncating the final result
	assign sum = (NaN==1'b1) ? ({sign_result, 5'b11111, mantissa_result}) :
			  ((infinity==1'b1) ? ({sign_result, 5'b11111, 10'd0}) : ({sign_result, exponent_result[4:0], mantissa_result}));

endmodule

/*----------------------------------------------------------
Reference: https://github.com/AkhilDotG/Floating-Point-Adder
------------------------------------------------------------*/