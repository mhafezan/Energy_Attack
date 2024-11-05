module encoder_unit (clk,
							en,
							ib_0, ib_1, ib_2, ib_3, ib_4, ib_5, ib_6, ib_7, ib_8, ib_9, ib_10, ib_11, ib_12, ib_13, ib_14, ib_15,
							ob_0, ob_1, ob_2, ob_3, ob_4, ob_5, ob_6, ob_7, ob_8, ob_9, ob_10, ob_11, ob_12, ob_13, ob_14, ob_15,
							of_0, of_1, of_2, of_3, of_4, of_5, of_6, of_7, of_8, of_9, of_10, of_11, of_12, of_13, of_14, of_15,
							data_ready);

input clk; // System clk corresponding to the propagation delay of the front-end module
input en; // Activated by the controller after each Filter Window Processing
input [15:0] ib_0, ib_1, ib_2, ib_3, ib_4, ib_5, ib_6, ib_7, ib_8, ib_9, ib_10, ib_11, ib_12, ib_13, ib_14, ib_15; // IB[0:15]
output reg [15:0] ob_0, ob_1, ob_2, ob_3, ob_4, ob_5, ob_6, ob_7, ob_8, ob_9, ob_10, ob_11, ob_12, ob_13, ob_14, ob_15; // OB[0:15]
output reg [3:0] of_0, of_1, of_2, of_3, of_4, of_5, of_6, of_7, of_8, of_9, of_10, of_11, of_12, of_13, of_14, of_15; // Offset[0:15]
output reg data_ready; // To activate an output Flag when a brick is ready

reg [3:0] input_index;
reg [7:0] output_index;
reg [5:0] offset_index;

reg [255:0] out_buff; // composed output: 16*(16 bits) = 256 bits
reg [63:0] off_buff;  // composed offset: 16*(4 bits) = 64 bits

// To define and initialize FSM states. Each state is considered for processing one single IB.
reg [4:0] state;
parameter s0 = 5'b00000;
parameter s1 = 5'b00001;
parameter s2 = 5'b00010;
parameter s3 = 5'b00011;
parameter s4 = 5'b00100;
parameter s5 = 5'b00101;
parameter s6 = 5'b00110;
parameter s7 = 5'b00111;
parameter s8 = 5'b01000;
parameter s9 = 5'b01001;
parameter s10 = 5'b01010;
parameter s11 = 5'b01011;
parameter s12 = 5'b01100;
parameter s13 = 5'b01101;
parameter s14 = 5'b01110;
parameter s15 = 5'b01111;
parameter s16 = 5'b10000;

// First always
always @(posedge clk) begin
	if (!en)
		state = s0;
	else begin
		case (state)
		s0: begin // IB0 
			// Conversion is started by zeroing out all OB and Offset entries
			out_buff = 256'b0;
			off_buff = 64'b0;
			input_index  = 4'b0;
			output_index = 8'b0;
			offset_index = 6'b0;
			if (ib_0 != 16'd0) begin
				out_buff[output_index +: 16] = ib_0;
				off_buff[offset_index +:  4] = input_index;
				output_index = output_index + 8'b00010000;
				offset_index = offset_index + 6'b000100;
			end
			input_index = input_index + 4'b0001;
			state = s1;
		end
		s1: begin // IB1
			if (ib_1 != 16'd0) begin
				out_buff[output_index +: 16] = ib_1;
				off_buff[offset_index +:  4] = input_index;
				output_index = output_index + 8'b00010000;
				offset_index = offset_index + 6'b000100;
			end
			input_index = input_index + 4'b0001;
			state = s2;
		end
		s2: begin // IB2
			if (ib_2 != 16'd0) begin
				out_buff[output_index +: 16] = ib_2;
				off_buff[offset_index +:  4] = input_index;
				output_index = output_index + 8'b00010000;
				offset_index = offset_index + 6'b000100;
			end
			input_index = input_index + 4'b0001;
			state = s3;
		end
		s3: begin // IB3
			if (ib_3 != 16'd0) begin
				out_buff[output_index +: 16] = ib_3;
				off_buff[offset_index +:  4] = input_index;
				output_index = output_index + 8'b00010000;
				offset_index = offset_index + 6'b000100;
			end
			input_index = input_index + 4'b0001;
			state = s4;
		end
		s4: begin // IB4
			if (ib_4 != 16'd0) begin
				out_buff[output_index +: 16] = ib_4;
				off_buff[offset_index +:  4] = input_index;
				output_index = output_index + 8'b00010000;
				offset_index = offset_index + 6'b000100;
			end
			input_index = input_index + 4'b0001;
			state = s5;
		end
		s5: begin // IB5
			if (ib_5 != 16'd0) begin
				out_buff[output_index +: 16] = ib_5;
				off_buff[offset_index +:  4] = input_index;
				output_index = output_index + 8'b00010000;
				offset_index = offset_index + 6'b000100;
			end
			input_index = input_index + 4'b0001;
			state = s6;
		end
		s6: begin // IB6
			if (ib_6 != 16'd0) begin
				out_buff[output_index +: 16] = ib_6;
				off_buff[offset_index +:  4] = input_index;
				output_index = output_index + 8'b00010000;
				offset_index = offset_index + 6'b000100;
			end
			input_index = input_index + 4'b0001;
			state = s7;
		end
		s7: begin // IB7
			if (ib_7 != 16'd0) begin
				out_buff[output_index +: 16] = ib_7;
				off_buff[offset_index +:  4] = input_index;
				output_index = output_index + 8'b00010000;
				offset_index = offset_index + 6'b000100;
			end
			input_index = input_index + 4'b0001;
			state = s8;
		end
		s8: begin // IB8
			if (ib_8 != 16'd0) begin
				out_buff[output_index +: 16] = ib_8;
				off_buff[offset_index +:  4] = input_index;
				output_index = output_index + 8'b00010000;
				offset_index = offset_index + 6'b000100;
			end
			input_index = input_index + 4'b0001;
			state = s9;
		end
		s9: begin // IB9
			if (ib_9 != 16'd0) begin
				out_buff[output_index +: 16] = ib_9;
				off_buff[offset_index +:  4] = input_index;
				output_index = output_index + 8'b00010000;
				offset_index = offset_index + 6'b000100;
			end
			input_index = input_index + 4'b0001;
			state = s10;
		end
		s10: begin // IB10
			if (ib_10 != 16'd0) begin // IB10
				out_buff[output_index +: 16] = ib_10;
				off_buff[offset_index +:  4] = input_index;
				output_index = output_index + 8'b00010000;
				offset_index = offset_index + 6'b000100;
			end
			input_index = input_index + 4'b0001;
			state = s11;
		end
		s11: begin // IB11
			if (ib_11 != 16'd0) begin
				out_buff[output_index +: 16] = ib_11;
				off_buff[offset_index +:  4] = input_index;
				output_index = output_index + 8'b00010000;
				offset_index = offset_index + 6'b000100;
			end
			input_index = input_index + 4'b0001;
			state = s12;
		end
		s12: begin // IB12
			if (ib_12 != 16'd0) begin
				out_buff[output_index +: 16] = ib_12;
				off_buff[offset_index +:  4] = input_index;
				output_index = output_index + 8'b00010000;
				offset_index = offset_index + 6'b000100;
			end
			input_index = input_index + 4'b0001;
			state = s13;
		end
		s13: begin // IB13
			if (ib_13 != 16'd0) begin
				out_buff[output_index +: 16] = ib_13;
				off_buff[offset_index +:  4] = input_index;
				output_index = output_index + 8'b00010000;
				offset_index = offset_index + 6'b000100;
			end
			input_index = input_index + 4'b0001;
			state = s14;
		end
		s14: begin // IB14		
			if (ib_14 != 16'd0) begin
				out_buff[output_index +: 16] = ib_14;
				off_buff[offset_index +:  4] = input_index;
				output_index = output_index + 8'b00010000;
				offset_index = offset_index + 6'b000100;
			end
			input_index = input_index + 4'b0001;
			state = s15;
		end
		s15: begin // IB15
			if (ib_15 != 16'd0) begin // IB15
				out_buff[output_index +: 16] = ib_15;
				off_buff[offset_index +:  4] = input_index;
			end
			state = s16;
		end
		s16: state = s16;
			/*  It stays at state 16 keeping data in OB and Offset outputs and setting the data_ready signal to 1, forcing the controller to write back the output in NM
			    and set the EN signal to zero which results in getting back FSM to S0. */
		endcase
	end
end

// Second always
always @(state or out_buff or off_buff) begin
	// To decompose a brick and assign it to 16 outputs when its data is ready after 16 clocks
	if (state == s16) begin
		ob_0 <= out_buff[15:0];
		ob_1 <= out_buff[31:16];
		ob_2 <= out_buff[47:32];
		ob_3 <= out_buff[63:48];
		ob_4 <= out_buff[79:64];
		ob_5 <= out_buff[95:80];
		ob_6 <= out_buff[111:96];
		ob_7 <= out_buff[127:112];
		ob_8 <= out_buff[143:128];
		ob_9 <= out_buff[159:144];
		ob_10 <= out_buff[175:160];
		ob_11 <= out_buff[191:176];
		ob_12 <= out_buff[207:192];
		ob_13 <= out_buff[223:208];
		ob_14 <= out_buff[239:224];
		ob_15 <= out_buff[255:240];
		
		of_0 <= off_buff[3:0];
		of_1 <= off_buff[7:4];
		of_2 <= off_buff[11:8];
		of_3 <= off_buff[15:12];
		of_4 <= off_buff[19:16];
		of_5 <= off_buff[23:20];
		of_6 <= off_buff[27:24];
		of_7 <= off_buff[31:28];
		of_8 <= off_buff[35:32];
		of_9 <= off_buff[39:36];
		of_10 <= off_buff[43:40];
		of_11 <= off_buff[47:44];
		of_12 <= off_buff[51:48];
		of_13 <= off_buff[55:52];
		of_14 <= off_buff[59:56];
		of_15 <= off_buff[63:60];
		
		data_ready <= 1'b1;
	end
	else begin
		ob_0 <= out_buff[15:0];
		ob_1 <= out_buff[31:16];
		ob_2 <= out_buff[47:32];
		ob_3 <= out_buff[63:48];
		ob_4 <= out_buff[79:64];
		ob_5 <= out_buff[95:80];
		ob_6 <= out_buff[111:96];
		ob_7 <= out_buff[127:112];
		ob_8 <= out_buff[143:128];
		ob_9 <= out_buff[159:144];
		ob_10 <= out_buff[175:160];
		ob_11 <= out_buff[191:176];
		ob_12 <= out_buff[207:192];
		ob_13 <= out_buff[223:208];
		ob_14 <= out_buff[239:224];
		ob_15 <= out_buff[255:240];
		
		of_0 <= off_buff[3:0];
		of_1 <= off_buff[7:4];
		of_2 <= off_buff[11:8];
		of_3 <= off_buff[15:12];
		of_4 <= off_buff[19:16];
		of_5 <= off_buff[23:20];
		of_6 <= off_buff[27:24];
		of_7 <= off_buff[31:28];
		of_8 <= off_buff[35:32];
		of_9 <= off_buff[39:36];
		of_10 <= off_buff[43:40];
		of_11 <= off_buff[47:44];
		of_12 <= off_buff[51:48];
		of_13 <= off_buff[55:52];
		of_14 <= off_buff[59:56];
		of_15 <= off_buff[63:60];
		
		data_ready <= 1'b0;
	end	
end

endmodule