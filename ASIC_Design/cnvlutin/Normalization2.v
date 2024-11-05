module Normalization2 (exponent_in, mantissa_sum_in, shift_dir_in, shift_num_in, exponent_out, mantissa_out);

input [4:0] exponent_in;
input [11:0] mantissa_sum_in;
input shift_dir_in;
input [3:0] shift_num_in;
output reg [5:0] exponent_out;
output reg [9:0] mantissa_out;

reg [11:0] mantissa_sum_temp;

always @ (*) begin
	exponent_out = 0;
	mantissa_sum_temp = 0;
	if(shift_dir_in) begin // Right shifting the mantissa sum by N bits
		mantissa_sum_temp = mantissa_sum_in >> shift_num_in;
		exponent_out = exponent_in + shift_num_in;
		end
	else begin // Left shifting the mantissa sum by N bits
		mantissa_sum_temp = mantissa_sum_in << shift_num_in;
		exponent_out = exponent_in - shift_num_in;
		end
end

always @ (*) begin // Assigning final mantissa to output
 	mantissa_out = mantissa_sum_temp[9:0];
end

endmodule