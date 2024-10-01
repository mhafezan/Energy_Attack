/* Combinational ReLU module that takes a 16-bit floating-point input in half-precision IEEE 754 format */

module floating_point_16bit_relu (relu_in, relu_out);

input [15:0] relu_in;
output reg [15:0] relu_out;

// Extracting the sign, exponent, and mantissa from the input
wire sign = relu_in [15];
wire [4:0] exponent = relu_in [14:10];
wire [9:0] mantissa = relu_in [9:0];

// Check if the input is negative
wire is_negative = (sign == 1'b1);

// Check if the number is Zero
wire is_zero = (exponent == 5'd0 && mantissa == 10'd0);

// If the input is negative, the output is 0
always @(*) begin
	if (is_negative || is_zero)
		relu_out = 16'b0;
	else
		// Otherwise, perform ReLU operation
		relu_out = {1'b0, exponent, mantissa};
end

endmodule