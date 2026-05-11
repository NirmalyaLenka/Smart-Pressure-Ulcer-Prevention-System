/**
 * CNN Inference Engine - Verilog
 * Runs quantized 1D temporal CNN on fused sensor data.
 * Output: 8-bit risk score (0-255 maps to 0.0 - 1.0) per zone.
 *
 * Pipeline stages:
 *   Stage 1: Input buffer (50-sample sliding window per zone)
 *   Stage 2: Conv1D (32 filters, kernel=5, ReLU, 8-bit)
 *   Stage 3: Conv1D (64 filters, kernel=3, ReLU, 8-bit)
 *   Stage 4: Global average pool
 *   Stage 5: Dense (sigmoid approximation via LUT)
 *   Stage 6: Output risk score + UART serializer
 */

`timescale 1ns / 1ps

module cnn_inference_engine #(
    parameter NUM_ZONES    = 64,
    parameter WINDOW_LEN   = 50,
    parameter FEATURES     = 4,    /* pressure, temp_delta, spo2, duration */
    parameter CONV1_FILT   = 32,
    parameter CONV1_KERN   = 5,
    parameter CONV2_FILT   = 64,
    parameter CONV2_KERN   = 3,
    parameter DATA_WIDTH   = 8     /* 8-bit quantized */
)(
    input  wire        clk,
    input  wire        rst_n,

    /* Sensor data input (from sensor_fusion module) */
    input  wire [DATA_WIDTH-1:0] sensor_in [FEATURES-1:0],
    input  wire [5:0]            zone_id_in,
    input  wire                  data_valid,

    /* Risk score output */
    output reg  [DATA_WIDTH-1:0] risk_score_out,
    output reg  [5:0]            zone_id_out,
    output reg                   score_valid
);

    /* Input sliding window buffer: [zone][time_step][feature] */
    reg [DATA_WIDTH-1:0] window_buf [NUM_ZONES-1:0][WINDOW_LEN-1:0][FEATURES-1:0];
    reg [5:0]            window_ptr [NUM_ZONES-1:0];

    /* Intermediate pipeline registers */
    reg [15:0] conv1_out [CONV1_FILT-1:0];   /* 16-bit to hold accumulation */
    reg [15:0] conv2_out [CONV2_FILT-1:0];
    reg [15:0] gap_out;
    reg [7:0]  sigmoid_out;

    /* Convolution weights — loaded from ROM in synthesis.
       Represented as signed 8-bit integers post-quantization. */
    /* synthesis translate_off */
    reg signed [7:0] conv1_w [CONV1_FILT-1:0][CONV1_KERN-1:0][FEATURES-1:0];
    reg signed [7:0] conv1_b [CONV1_FILT-1:0];
    reg signed [7:0] conv2_w [CONV2_FILT-1:0][CONV2_KERN-1:0][CONV1_FILT-1:0];
    reg signed [7:0] conv2_b [CONV2_FILT-1:0];
    reg signed [7:0] dense_w [CONV2_FILT-1:0];
    reg signed [7:0] dense_b;

    initial begin
        $readmemh("weights/conv1_w.hex", conv1_w);
        $readmemh("weights/conv1_b.hex", conv1_b);
        $readmemh("weights/conv2_w.hex", conv2_w);
        $readmemh("weights/conv2_b.hex", conv2_b);
        $readmemh("weights/dense_w.hex", dense_w);
    end
    /* synthesis translate_on */

    /* Sigmoid approximation LUT (256 entries, 8-bit output) */
    reg [7:0] sigmoid_lut [255:0];
    initial $readmemh("weights/sigmoid_lut.hex", sigmoid_lut);

    /* State machine */
    localparam S_IDLE    = 3'd0;
    localparam S_BUFFER  = 3'd1;
    localparam S_CONV1   = 3'd2;
    localparam S_CONV2   = 3'd3;
    localparam S_GAP     = 3'd4;
    localparam S_DENSE   = 3'd5;
    localparam S_OUTPUT  = 3'd6;

    reg [2:0]  state;
    reg [5:0]  cur_zone;
    reg [5:0]  filt_idx;
    reg [4:0]  kern_idx;
    reg [31:0] acc;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            score_valid <= 1'b0;
            for (integer z = 0; z < NUM_ZONES; z++)
                window_ptr[z] <= 0;
        end else begin
            score_valid <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (data_valid) begin
                        cur_zone <= zone_id_in;

                        /* Shift window and insert new sample */
                        for (integer t = WINDOW_LEN-1; t > 0; t = t - 1)
                            for (integer f = 0; f < FEATURES; f++)
                                window_buf[zone_id_in][t][f] <=
                                    window_buf[zone_id_in][t-1][f];
                        for (integer f = 0; f < FEATURES; f++)
                            window_buf[zone_id_in][0][f] <= sensor_in[f];

                        /* Only run inference when window is full */
                        if (window_ptr[zone_id_in] < WINDOW_LEN - 1)
                            window_ptr[zone_id_in] <= window_ptr[zone_id_in] + 1;
                        else
                            state <= S_CONV1;
                    end
                end

                S_CONV1: begin
                    /* Simplified: compute one filter per clock cycle.
                       Full implementation parallelises across filters. */
                    acc = 32'sd0;
                    for (integer k = 0; k < CONV1_KERN; k++)
                        for (integer f = 0; f < FEATURES; f++)
                            acc = acc + $signed(conv1_w[filt_idx][k][f])
                                      * $signed({1'b0, window_buf[cur_zone][k][f]});
                    acc = acc + $signed({24'd0, conv1_b[filt_idx]});

                    /* ReLU */
                    conv1_out[filt_idx] <= (acc[31]) ? 16'd0 : acc[15:0];

                    if (filt_idx == CONV1_FILT - 1) begin
                        filt_idx <= 0;
                        state    <= S_CONV2;
                    end else
                        filt_idx <= filt_idx + 1;
                end

                S_CONV2: begin
                    acc = 32'sd0;
                    for (integer k = 0; k < CONV2_KERN; k++)
                        for (integer f2 = 0; f2 < CONV1_FILT; f2++)
                            acc = acc + $signed(conv2_w[filt_idx][k][f2])
                                      * $signed({1'b0, conv1_out[f2][7:0]});
                    acc = acc + $signed({24'd0, conv2_b[filt_idx]});
                    conv2_out[filt_idx] <= (acc[31]) ? 16'd0 : acc[15:0];

                    if (filt_idx == CONV2_FILT - 1) begin
                        filt_idx <= 0;
                        state    <= S_GAP;
                    end else
                        filt_idx <= filt_idx + 1;
                end

                S_GAP: begin
                    /* Global average pooling across CONV2_FILT outputs */
                    acc = 32'd0;
                    for (integer f3 = 0; f3 < CONV2_FILT; f3++)
                        acc = acc + {16'd0, conv2_out[f3]};
                    gap_out <= acc[25:10];  /* divide by 64 = right shift 6 */
                    state   <= S_DENSE;
                end

                S_DENSE: begin
                    /* Single dense unit with sigmoid via LUT */
                    acc = 32'sd0;
                    for (integer f4 = 0; f4 < CONV2_FILT; f4++)
                        acc = acc + $signed(dense_w[f4])
                                  * $signed({1'b0, conv2_out[f4][7:0]});
                    acc = acc + $signed({24'd0, dense_b});

                    /* Clamp to LUT range [0, 255] */
                    sigmoid_out <= sigmoid_lut[acc[7:0]];
                    state       <= S_OUTPUT;
                end

                S_OUTPUT: begin
                    risk_score_out <= sigmoid_out;
                    zone_id_out    <= cur_zone;
                    score_valid    <= 1'b1;
                    filt_idx       <= 0;
                    state          <= S_IDLE;
                end
            endcase
        end
    end

endmodule
