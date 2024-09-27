module Normalization1 (mantissa_sum, shift_dir, shift_num);

input [11:0] mantissa_sum;
output reg shift_dir;
output reg [3:0] shift_num;

// 0 is assigned to dir for Left shift, and 1 is assigned for Right shift
always @(*) begin
	casex(mantissa_sum)
		12'b1xxxxxxxxxxx: // Overflow in the Mantissa is handled here by shifting to the right and by sacrificing precision.
			begin
			shift_dir = 1'b1;
			shift_num = 4'b0001;	
			end
		12'b01xxxxxxxxxx:
			begin
			shift_dir = 1'b0;
			shift_num  = 4'b0000;
			end
		12'b001xxxxxxxxx:
			begin
			shift_dir = 1'b0;
			shift_num   = 4'b0001;
			end
		12'b0001xxxxxxxx:
			begin
			shift_dir = 1'b0;
			shift_num   = 4'b0010;
			end
		12'b00001xxxxxxx:
			begin
			shift_dir = 1'b0;
			shift_num   = 4'b0011;
			end
		12'b000001xxxxxx:
			begin
			shift_dir = 1'b0;
			shift_num   = 4'b0100;
			end
		12'b0000001xxxxx:
			begin
			shift_dir = 1'b0;
			shift_num   = 4'b0101;
			end
		12'b00000001xxxx:
			begin
			shift_dir = 1'b0;
			shift_num   = 4'b0110;
			end
		12'b000000001xxx:
			begin
 			shift_dir = 1'b0;
			shift_num   = 4'b0111;
			end
		12'b0000000001xx:
			begin
			shift_dir = 1'b0;
			shift_num   = 4'b1000;
			end
		12'b00000000001x:
			begin
			shift_dir = 1'b0;
			shift_num   = 4'b1001;
			end
		12'b000000000001:
			begin
			shift_dir = 1'b0;
			shift_num   = 4'b1010;
			end
		default: // 12'b000000000000 which happens when (mantissa_A == mantissa_B == 0) or (+mantissa_A == -mantissa_B)
			begin
			shift_dir = 1'b0;
			shift_num = 4'b0000;
			end
	endcase

end

endmodule
