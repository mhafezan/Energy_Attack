module exponent_comparator (exp1, exp2, exp3, exp4, exp_selection_bitmap);

input [4:0] exp1;
input [4:0] exp2;
input [4:0] exp3;
input [4:0] exp4;

output reg [3:0] exp_selection_bitmap;

reg [3:0] exp_selection_bitmap_1 = 4'b0000;
reg [3:0] exp_selection_bitmap_2 = 4'b0000;

reg [4:0] temp1;
reg [4:0] temp2;

// To binary exploration of the maximum exponent
always @(*) begin

	if(exp1 <= exp2) begin
		temp1 = exp2;
		exp_selection_bitmap_1 = 4'b0010;
	end else begin
		temp1 = exp1;
		exp_selection_bitmap_1 = 4'b0001;
		end
		
	if(exp3 <= exp4) begin
		temp2 = exp4;
		exp_selection_bitmap_2 = 4'b1000;
	end else begin
		temp2 = exp3;
		exp_selection_bitmap_2 = 4'b0100;
		end
		
	if(temp1 <= temp2)
		exp_selection_bitmap = exp_selection_bitmap_2;
	else
		exp_selection_bitmap = exp_selection_bitmap_1;

end
	
endmodule