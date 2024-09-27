module multplication_normalizer (man_product, exp_sum, man_res, exp_res);

input [21:0] man_product;
input [5:0] exp_sum;
output reg [9:0] man_res;
output reg [5:0] exp_res;

always @(man_product, exp_sum) begin
		casex(man_product) 
			22'b1xxxxxxxxxxxxxxxxxxxxx:
			begin
			    man_res = man_product[20:11];
			    exp_res = exp_sum + 1;
			end
			22'b01xxxxxxxxxxxxxxxxxxxx:
			begin
			    man_res = man_product[19:10];
			    exp_res = exp_sum;
			end
			22'b001xxxxxxxxxxxxxxxxxxx:
			begin
			    man_res = man_product[18:9];
			    exp_res = exp_sum-1;
			end
			22'b0001xxxxxxxxxxxxxxxxxx:
			begin
			    man_res = man_product[17:8];
			    exp_res = exp_sum-2;
			end
			22'b00001xxxxxxxxxxxxxxxxx:
			begin
			    man_res = man_product[16:7];
			    exp_res = exp_sum-3;
			end
			22'b000001xxxxxxxxxxxxxxxx:
			begin
			    man_res = man_product[15:6];
			    exp_res = exp_sum-4;
			end
			22'b0000001xxxxxxxxxxxxxxx:
			begin
			    man_res = man_product[14:5];
			    exp_res = exp_sum-5;
			end
			22'b00000001xxxxxxxxxxxxxx:
			begin
			    man_res = man_product[13:4];
			    exp_res = exp_sum-6;
			end
			22'b000000001xxxxxxxxxxxxx:
			begin
			    man_res = man_product[12:3];
			    exp_res = exp_sum-7;
			end
			22'b0000000001xxxxxxxxxxxx:
			begin
			    man_res = man_product[11:2];
			    exp_res = exp_sum-8;
			end
			22'b00000000001xxxxxxxxxxx:
			begin
			    man_res = man_product[10:1];
			    exp_res = exp_sum-9;
			end
			22'b000000000001xxxxxxxxxx:
			begin
			    man_res = man_product[9:0];
			    exp_res = exp_sum-10;
			end
			22'b0000000000001xxxxxxxxx:
			begin
			    man_res = {man_product[8:0],1'b0};
			    exp_res = exp_sum-11;
			end
			22'b00000000000001xxxxxxxx:
			begin
			    man_res = {man_product[7:0],2'b00};
			    exp_res = exp_sum-12;
			end
			22'b000000000000001xxxxxxx:
			begin
			    man_res = {man_product[6:0],3'b000};
			    exp_res = exp_sum-13;
			end
			22'b0000000000000001xxxxxx:
			begin
			    man_res = {man_product[5:0],4'b0000};
			    exp_res = exp_sum-14;
			end
			22'b00000000000000001xxxxx:
			begin
			    man_res = {man_product[4:0],5'b0000};
			    exp_res = exp_sum-15;
			end
			22'b000000000000000001xxxx:
			begin
			    man_res = {man_product[3:0],6'b0};
			    exp_res = exp_sum-16;
			end
			22'b0000000000000000001xxx:
			begin
			    man_res = {man_product[2:0],7'b0};
			    exp_res = exp_sum-17;
			end
			22'b00000000000000000001xx:
			begin
			    man_res = {man_product[1:0],8'b0};
			    exp_res = exp_sum-18;
			end
			22'b000000000000000000001x:
			begin
			    man_res = {man_product[0],9'b0};
			    exp_res = exp_sum-19;
			end
			default:
			begin
			    man_res = 10'b0;
			    exp_res = 6'b0;
			end
		endcase
end
endmodule