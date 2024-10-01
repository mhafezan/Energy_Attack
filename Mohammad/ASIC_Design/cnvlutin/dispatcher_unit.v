module dispatcher_unit (
	clk,
	zfnaf,
	ready0, ready1, ready2, ready3, ready4, ready5, ready6, ready7, ready8, ready9, ready10, ready11, ready12, ready13, ready14, ready15,
	brick0, brick1, brick2, brick3, brick4, brick5, brick6, brick7, brick8, brick9, brick10, brick11, brick12, brick13, brick14, brick15,
	offset0, offset1, offset2, offset3, offset4, offset5, offset6, offset7, offset8, offset9, offset10, offset11, offset12, offset13, offset14, offset15,
	out_l0, out_l1, out_l2, out_l3, out_l4, out_l5, out_l6, out_l7, out_l8, out_l9, out_l10, out_l11, out_l12, out_l13, out_l14, out_l15,
	off_l0, off_l1, off_l2, off_l3, off_l4, off_l5, off_l6, off_l7, off_l8, off_l9, off_l10, off_l11, off_l12, off_l13, off_l14, off_l15,
	b0_request_next, b1_request_next, b2_request_next, b3_request_next, b4_request_next, b5_request_next, b6_request_next, b7_request_next, b8_request_next,
	b9_request_next, b10_request_next, b11_request_next, b12_request_next, b13_request_next, b14_request_next, b15_request_next);

input clk;

/* To activate ZFNAF formatting. ZFNAF is deactive in processing the first layer, where the Dispatcher broadcasts all zeros and non-zeros to all units
	and front-end unit does not exploit offset buffer to index SB. */
input zfnaf;

// To inform a new data is ready for brick0, brick1, ..., brick15
input ready0, ready1, ready2, ready3, ready4, ready5, ready6, ready7, ready8, ready9, ready10, ready11, ready12, ready13, ready14, ready15;

input [255:0] brick0, brick1, brick2, brick3, brick4, brick5, brick6, brick7, brick8, brick9, brick10, brick11, brick12, brick13, brick14, brick15;
input [63:0] offset0, offset1, offset2, offset3, offset4, offset5, offset6, offset7, offset8, offset9, offset10, offset11, offset12, offset13, offset14, offset15;

output reg [15:0] out_l0, out_l1, out_l2, out_l3, out_l4, out_l5, out_l6, out_l7, out_l8, out_l9, out_l10, out_l11, out_l12, out_l13, out_l14, out_l15;
output reg [3:0]  off_l0, off_l1, off_l2, off_l3, off_l4, off_l5, off_l6, off_l7, off_l8, off_l9, off_l10, off_l11, off_l12, off_l13, off_l14, off_l15;

output reg b0_request_next, b1_request_next, b2_request_next, b3_request_next, b4_request_next, b5_request_next, b6_request_next, b7_request_next,
			  b8_request_next, b9_request_next, b10_request_next, b11_request_next, b12_request_next, b13_request_next, b14_request_next, b15_request_next;

reg [4:0] b0_counter; // One more bit is considered for checking counter overflow
reg [4:0] b1_counter;
reg [4:0] b2_counter;
reg [4:0] b3_counter;
reg [4:0] b4_counter;
reg [4:0] b5_counter;
reg [4:0] b6_counter;
reg [4:0] b7_counter;
reg [4:0] b8_counter;
reg [4:0] b9_counter;
reg [4:0] b10_counter;
reg [4:0] b11_counter;
reg [4:0] b12_counter;
reg [4:0] b13_counter;
reg [4:0] b14_counter;
reg [4:0] b15_counter;

reg [255:0] out_brick;
reg [63:0]  off_brick;

reg [255:0] b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15;
reg [63:0]  o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15;

reg write_permit;
reg b0_zero, b1_zero, b2_zero, b3_zero, b4_zero, b5_zero, b6_zero, b7_zero, b8_zero, b9_zero, b10_zero, b11_zero, b12_zero, b13_zero, b14_zero, b15_zero;
							  
// First always: To implement the dispatcher's main function
always @(posedge clk) begin
	if (ready0) begin
		b0_counter = 5'b0;
		b0_request_next = 1'b0;
		b0 = brick0;
		o0 = offset0;
	end
	if (ready1) begin
		b1_counter = 5'b0;
		b1_request_next = 1'b0;
		b1 = brick1;
		o1 = offset1;
	end
	if (ready2) begin
		b2_counter = 5'b0;
		b2_request_next = 1'b0;
		b2 = brick2;
		o2 = offset2;
	end
	if (ready3) begin
		b3_counter = 5'b0;
		b3_request_next = 1'b0;
		b3 = brick3;
		o3 = offset3;
	end
	if (ready4) begin
		b4_counter = 5'b0;
		b4_request_next = 1'b0;
		b4 = brick4;
		o4 = offset4;		
	end
	if (ready5) begin
		b5_counter = 5'b0;
		b5_request_next = 1'b0;
		b5 = brick5;
		o5 = offset5;		
	end
	if (ready6) begin
		b6_counter = 5'b0;
		b6_request_next = 1'b0;
		b6 = brick6;
		o6 = offset6;		
	end
	if (ready7) begin
		b7_counter = 5'b0;
		b7_request_next = 1'b0;
		b7 = brick7;
		o7 = offset7;
	end
	if (ready8) begin
		b8_counter = 5'b0;
		b8_request_next = 1'b0;
		b8 = brick8;
		o8 = offset8;
	end
	if (ready9) begin
		b9_counter = 5'b0;
		b9_request_next = 1'b0;
		b9 = brick9;
		o9 = offset9;
	end
	if (ready10) begin
		b10_counter = 5'b0;
		b10_request_next = 1'b0;
		b10 = brick10;
		o10 = offset10;
	end
	if (ready11) begin
		b11_counter = 5'b0;
		b11_request_next = 1'b0;
		b11 = brick11;
		o11 = offset11;
	end
	if (ready12) begin
		b12_counter = 5'b0;
		b12_request_next = 1'b0;
		b12 = brick12;
		o12 = offset12;
	end
	if (ready13) begin
		b13_counter = 5'b0;
		b13_request_next = 1'b0;
		b13 = brick13;
		o13 = offset13;
	end
	if (ready14) begin
		b14_counter = 5'b0;
		b14_request_next = 1'b0;
		b14 = brick14;
		o14 = offset14;
	end
	if (ready15) begin
		b15_counter = 5'b0;
		b15_request_next = 1'b0;
		b15 = brick15;
		o15 = offset15;
	end
	
	// To initialize the write permit in each cycle
	b0_zero  = (b0  [15:0] == 16'b0) ? (1'b1):(1'b0);
	b1_zero  = (b1  [15:0] == 16'b0) ? (1'b1):(1'b0);
	b2_zero  = (b2  [15:0] == 16'b0) ? (1'b1):(1'b0);
	b3_zero  = (b3  [15:0] == 16'b0) ? (1'b1):(1'b0);
	b4_zero  = (b4  [15:0] == 16'b0) ? (1'b1):(1'b0);
	b5_zero  = (b5  [15:0] == 16'b0) ? (1'b1):(1'b0);
	b6_zero  = (b6  [15:0] == 16'b0) ? (1'b1):(1'b0);
	b7_zero  = (b7  [15:0] == 16'b0) ? (1'b1):(1'b0);
	b8_zero  = (b8  [15:0] == 16'b0) ? (1'b1):(1'b0);
	b9_zero  = (b9  [15:0] == 16'b0) ? (1'b1):(1'b0);
	b10_zero = (b10 [15:0] == 16'b0) ? (1'b1):(1'b0);
	b11_zero = (b11 [15:0] == 16'b0) ? (1'b1):(1'b0);
	b12_zero = (b12 [15:0] == 16'b0) ? (1'b1):(1'b0);
	b13_zero = (b13 [15:0] == 16'b0) ? (1'b1):(1'b0);
	b14_zero = (b14 [15:0] == 16'b0) ? (1'b1):(1'b0);
	b15_zero = (b15 [15:0] == 16'b0) ? (1'b1):(1'b0);
	
	write_permit = (b0_zero==1'b0 && b1_zero==1'b0 && b2_zero==1'b0 && b3_zero==1'b0 && b4_zero==1'b0 && b5_zero==1'b0 && b6_zero==1'b0 && b7_zero==1'b0 && b8_zero==1'b0 &&
						 b9_zero==1'b0 && b10_zero==1'b0 && b11_zero==1'b0 && b12_zero==1'b0 && b13_zero==1'b0 && b14_zero==1'b0 && b15_zero==1'b0)?(1'b1):(1'b0);
	
	if (zfnaf) begin
		if (b0 [15:0] != 16'b0 && b0_counter <= 15 && write_permit == 1'b1) begin
			b0_request_next = 1'b0;
			out_brick [15:0] = b0 [15:0];
			off_brick [3:0]  = o0 [3:0];
			b0 = b0 >> 16;
			o0 = o0 >> 4; 
			b0_counter = b0_counter + 5'b00001;
		end else if (b0 [15:0] == 16'b0 || b0_counter > 15) begin
			b0_request_next = 1'b1;
			b0_counter = b0_counter;
		end
		if (b1 [15:0] != 16'b0 && b1_counter <= 15 && write_permit == 1'b1) begin
			b1_request_next = 1'b0;
			out_brick [31:16] = b1 [15:0];
			off_brick [7:4]   = o1 [3:0];
			b1 = b1 >> 16;
			o1 = o1 >> 4; 
			b1_counter = b1_counter + 5'b00001;
		end else if (b1 [15:0] == 16'b0 || b1_counter > 15) begin
			b1_request_next = 1'b1;
			b1_counter = b1_counter;
		end
		if (b2 [15:0] != 16'b0 && b2_counter <= 15 && write_permit == 1'b1) begin
			b2_request_next = 1'b0;
			out_brick [47:32] = b2 [15:0];
			off_brick [11:8]  = o2 [3:0];
			b2 = b2 >> 16;
			o2 = o2 >> 4; 
			b2_counter = b2_counter + 5'b00001;
		end else if (b2 [15:0] == 16'b0 || b2_counter > 15) begin
			b2_request_next = 1'b1;
			b2_counter = b2_counter;
		end
		if (b3 [15:0] != 16'b0 && b3_counter <= 15 && write_permit == 1'b1) begin
			b3_request_next = 1'b0;
			out_brick [63:48] = b3 [15:0];
			off_brick [15:12] = o3 [3:0];
			b3 = b3 >> 16;
			o3 = o3 >> 4; 
			b3_counter = b3_counter + 5'b00001;
		end else if (b3 [15:0] == 16'b0 || b3_counter > 15) begin
			b3_request_next = 1'b1;
			b3_counter = b3_counter;
		end
		if (b4 [15:0] != 16'b0 && b4_counter <= 15 && write_permit == 1'b1) begin
			b4_request_next = 1'b0;
			out_brick [79:64] = b4 [15:0];
			off_brick [19:16] = o4 [3:0];
			b4 = b4 >> 16;
			o4 = o4 >> 4; 
			b4_counter = b4_counter + 5'b00001;
		end else if (b4 [15:0] == 16'b0 || b4_counter > 15) begin
			b4_request_next = 1'b1;
			b4_counter = b4_counter;
		end
		if (b5 [15:0] != 16'b0 && b5_counter <= 15 && write_permit == 1'b1) begin
			b5_request_next = 1'b0;
			out_brick [95:80] = b5 [15:0];
			off_brick [23:20] = o5 [3:0];
			b5 = b5 >> 16;
			o5 = o5 >> 4; 
			b5_counter = b5_counter + 5'b00001;
		end else if (b5 [15:0] == 16'b0 || b5_counter > 15) begin
			b5_request_next = 1'b1;
			b5_counter = b5_counter;
		end
		if (b6 [15:0] != 16'b0 && b6_counter <= 15 && write_permit == 1'b1) begin
			b6_request_next = 1'b0;
			out_brick [111:96] = b6 [15:0];
			off_brick [27:24]  = o6 [3:0];
			b6 = b6 >> 16;
			o6 = o6 >> 4;
			b6_counter = b6_counter + 5'b00001;
		end else if (b6 [15:0] == 16'b0 || b6_counter > 15) begin
			b6_request_next = 1'b1;
			b6_counter = b6_counter;
		end
		if (b7 [15:0] != 16'b0 && b7_counter <= 15 && write_permit == 1'b1) begin
			b7_request_next = 1'b0;
			out_brick [127:112] = b7 [15:0];
			off_brick [31:28]   = o7 [3:0];
			b7 = b7 >> 16;
			o7 = o7 >> 4; 
			b7_counter = b7_counter + 5'b00001;
		end else if (b7 [15:0] == 16'b0 || b7_counter > 15) begin
			b7_request_next = 1'b1;
			b7_counter = b7_counter;
		end
		if (b8 [15:0] != 16'b0 && b8_counter <= 15 && write_permit == 1'b1) begin
			b8_request_next = 1'b0;
			out_brick [143:128] = b8 [15:0];
			off_brick [35:32]   = o8 [3:0];
			b8 = b8 >> 16;
			o8 = o8 >> 4; 
			b8_counter = b8_counter + 5'b00001;
		end else if (b8 [15:0] == 16'b0 || b8_counter > 15) begin
			b8_request_next = 1'b1;
			b8_counter = b8_counter;
		end
		if (b9 [15:0] != 16'b0 && b9_counter <= 15 && write_permit == 1'b1) begin
			b9_request_next = 1'b0;
			out_brick [159:144] = b9 [15:0];
			off_brick [39:36]   = o9 [3:0];
			b9 = b9 >> 16;
			o9 = o9 >> 4; 
			b9_counter = b9_counter + 5'b00001;
		end else if (b9 [15:0] == 16'b0 || b9_counter > 15) begin
			b9_request_next = 1'b1;
			b9_counter = b9_counter;
		end
		if (b10 [15:0] != 16'b0 && b10_counter <= 15 && write_permit == 1'b1) begin
			b10_request_next = 1'b0;
			out_brick [175:160] = b10 [15:0];
			off_brick [43:40]   = o10 [3:0];
			b10 = b10 >> 16;
			o10 = o10 >> 4; 
			b10_counter = b10_counter + 5'b00001;
		end else if (b10 [15:0] == 16'b0 || b10_counter > 15) begin
			b10_request_next = 1'b1;
			b10_counter = b10_counter;
		end
		if (b11 [15:0] != 16'b0 && b11_counter <= 15 && write_permit == 1'b1) begin
			b11_request_next = 1'b0;
			out_brick [191:176] = b11 [15:0];
			off_brick [47:44]   = o11 [3:0];
			b11 = b11 >> 16;
			o11 = o11 >> 4; 
			b11_counter = b11_counter + 5'b00001;
		end else if (b11 [15:0] == 16'b0 || b11_counter > 15) begin
			b11_request_next = 1'b1;
			b11_counter = b11_counter;
		end
		if (b12 [15:0] != 16'b0 && b12_counter <= 15 && write_permit == 1'b1) begin
			b12_request_next = 1'b0;
			out_brick [207:192] = b12 [15:0];
			off_brick [51:48]   = o12 [3:0];
			b12 = b12 >> 16;
			o12 = o12 >> 4; 
			b12_counter = b12_counter + 5'b00001;
		end else if (b12 [15:0] == 16'b0 || b12_counter > 15) begin
			b12_request_next = 1'b1;
			b12_counter = b12_counter;
		end
		if (b13 [15:0] != 16'b0 && b13_counter <= 15 && write_permit == 1'b1) begin
			b13_request_next = 1'b0;
			out_brick [223:208] = b13 [15:0];
			off_brick [55:52]   = o13 [3:0];
			b13 = b13 >> 16;
			o13 = o13 >> 4; 
			b13_counter = b13_counter + 5'b00001;
		end else if (b13 [15:0] == 16'b0 || b13_counter > 15) begin
			b13_request_next = 1'b1;
			b13_counter = b13_counter;
		end
		if (b14 [15:0] != 16'b0 && b14_counter <= 15 && write_permit == 1'b1) begin
			b14_request_next = 1'b0;
			out_brick [239:224] = b14 [15:0];
			off_brick [59:56]   = o14 [3:0];
			b14 = b14 >> 16;
			o14 = o14 >> 4; 
			b14_counter = b14_counter + 5'b00001;
		end else if (b14 [15:0] == 16'b0 || b14_counter > 15) begin
			b14_request_next = 1'b1;
			b14_counter = b14_counter;
		end
		if (b15 [15:0] != 16'b0 && b15_counter <= 15 && write_permit == 1'b1) begin
			b15_request_next = 1'b0;
			out_brick [255:240] = b15 [15:0];
			off_brick [63:60]   = o15 [3:0];
			b15 = b15 >> 16;
			o15 = o15 >> 4; 
			b15_counter = b15_counter + 5'b00001;
		end else if (b15 [15:0] == 16'b0 || b15_counter > 15) begin
			b15_request_next = 1'b1;
			b15_counter = b15_counter;
		end
	end else if (!zfnaf) begin
		if (b0_counter <= 15) begin
			
			out_brick [15:0]    = b0  [15:0];
			out_brick [31:16]   = b1  [15:0];
			out_brick [47:32]   = b2  [15:0];
			out_brick [63:48]   = b3  [15:0];
			out_brick [79:64]   = b4  [15:0];
			out_brick [95:80]   = b5  [15:0];
			out_brick [111:96]  = b6  [15:0];
			out_brick [127:112] = b7  [15:0];
			out_brick [143:128] = b8  [15:0];
			out_brick [159:144] = b9  [15:0];
			out_brick [175:160] = b10 [15:0];
			out_brick [191:176] = b11 [15:0];
			out_brick [207:192] = b12 [15:0];
			out_brick [223:208] = b13 [15:0];
			out_brick [239:224] = b14 [15:0];
			out_brick [255:240] = b15 [15:0];
			
			off_brick [3:0]   = o0  [3:0];
			off_brick [7:4]   = o1  [3:0];
			off_brick [11:8]  = o2  [3:0];
			off_brick [15:12] = o3  [3:0];
			off_brick [19:16] = o4  [3:0];
			off_brick [23:20] = o5  [3:0];
			off_brick [27:24] = o6  [3:0];
			off_brick [31:28] = o7  [3:0];
			off_brick [35:32] = o8  [3:0];
			off_brick [39:36] = o9  [3:0];
			off_brick [43:40] = o10 [3:0];
			off_brick [47:44] = o11 [3:0];
			off_brick [51:48] = o12 [3:0];
			off_brick [55:52] = o13 [3:0];
			off_brick [59:56] = o14 [3:0];
			off_brick [63:60] = o15 [3:0];

			if (b0_counter != 15) begin
				b0  = b0  >> 16;
				b1  = b1  >> 16;
				b2  = b2  >> 16;
				b3  = b3  >> 16;
				b4  = b4  >> 16;
				b5  = b5  >> 16;
				b6  = b6  >> 16;
				b7  = b7  >> 16;
				b8  = b8  >> 16;
				b9  = b9  >> 16;
				b10 = b10 >> 16;
				b11 = b11 >> 16;
				b12 = b12 >> 16;
				b13 = b13 >> 16;
				b14 = b14 >> 16;
				b15 = b15 >> 16;
				
				o0  = o0  >> 4;
				o1  = o1  >> 4;
				o2  = o2  >> 4;
				o3  = o3  >> 4;
				o4  = o4  >> 4;
				o5  = o5  >> 4;
				o6  = o6  >> 4;
				o7  = o7  >> 4;
				o8  = o8  >> 4;
				o9  = o9  >> 4;
				o10 = o10 >> 4;
				o11 = o11 >> 4;
				o12 = o12 >> 4;
				o13 = o13 >> 4;
				o14 = o14 >> 4;
				o15 = o15 >> 4;
			end
			
			b0_request_next  = 1'b0;
			b1_request_next  = 1'b0;
			b2_request_next  = 1'b0;
			b3_request_next  = 1'b0;
			b4_request_next  = 1'b0;
			b5_request_next  = 1'b0;
			b6_request_next  = 1'b0;
			b7_request_next  = 1'b0;
			b8_request_next  = 1'b0;
			b9_request_next  = 1'b0;
			b10_request_next = 1'b0;
			b11_request_next = 1'b0;
			b12_request_next = 1'b0;
			b13_request_next = 1'b0;
			b14_request_next = 1'b0;
			b15_request_next = 1'b0;
			
			b0_counter = b0_counter + 5'b00001;
			
		end else if (b0_counter > 15) begin
			b0_request_next  = 1'b1;
			b1_request_next  = 1'b1;
			b2_request_next  = 1'b1;
			b3_request_next  = 1'b1;
			b4_request_next  = 1'b1;
			b5_request_next  = 1'b1;
			b6_request_next  = 1'b1;
			b7_request_next  = 1'b1;
			b8_request_next  = 1'b1;
			b9_request_next  = 1'b1;
			b10_request_next = 1'b1;
			b11_request_next = 1'b1;
			b12_request_next = 1'b1;
			b13_request_next = 1'b1;
			b14_request_next = 1'b1;
			b15_request_next = 1'b1;
			
			b0_counter = b0_counter;
		end
	end
end

// Second always: To broadcast a prepared Output Brick and Offset Brick to different lanes (When there are no requests for a new brick, we broadcast to the output)
always @(off_brick[63:60] or out_brick [255:240]) begin
	out_l0  <= out_brick [15:0];
	out_l1  <= out_brick [31:16];
	out_l2  <= out_brick [47:32];
	out_l3  <= out_brick [63:48];
	out_l4  <= out_brick [79:64];
	out_l5  <= out_brick [95:80];
	out_l6  <= out_brick [111:96];
	out_l7  <= out_brick [127:112];
	out_l8  <= out_brick [143:128];
	out_l9  <= out_brick [159:144];
	out_l10 <= out_brick [175:160];
	out_l11 <= out_brick [191:176];
	out_l12 <= out_brick [207:192];
	out_l13 <= out_brick [223:208];
	out_l14 <= out_brick [239:224];
	out_l15 <= out_brick [255:240];

	off_l0  <= off_brick [3:0];
	off_l1  <= off_brick [7:4];
	off_l2  <= off_brick [11:8];
	off_l3  <= off_brick [15:12];
	off_l4  <= off_brick [19:16];
	off_l5  <= off_brick [23:20];
	off_l6  <= off_brick [27:24];
	off_l7  <= off_brick [31:28];
	off_l8  <= off_brick [35:32];
	off_l9  <= off_brick [39:36];
	off_l10 <= off_brick [43:40];
	off_l11 <= off_brick [47:44];
	off_l12 <= off_brick [51:48];
	off_l13 <= off_brick [55:52];
	off_l14 <= off_brick [59:56];
	off_l15 <= off_brick [63:60];
end

endmodule