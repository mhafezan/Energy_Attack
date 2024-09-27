module Addition (sign_A, sign_B, mantissa_A, mantissa_B, sign_result, mantissa_sum);

input sign_A;
input sign_B;
input [10:0] mantissa_A;
input [10:0] mantissa_B;
output reg sign_result;
output reg [11:0] mantissa_sum;

always @(*) begin
	/* If signs are different, the sign of the larger number specifies the sign of the addition, and a subtraction determines the result
		of mantissa addition */
	if(sign_A != sign_B)
	begin
		if(mantissa_A > mantissa_B) begin
			mantissa_sum = mantissa_A - mantissa_B;
			sign_result = sign_A; // Assigning the sign of greater input, A
		end
		else  if(mantissa_B > mantissa_A) begin
			mantissa_sum = mantissa_B - mantissa_A;
			sign_result = sign_B ; // Assigning the sign of greater input, B
		end
		else begin // If mantissa_A == mantissa_B
			mantissa_sum = 0;
			sign_result = 0;		
		end
	end
	// If both signs are equal then add and take sign from any of the inputs
	else begin
		mantissa_sum = mantissa_A + mantissa_B;
		sign_result = sign_A;
	end
end

endmodule
