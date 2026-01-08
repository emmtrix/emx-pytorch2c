#include <stdint.h>
#include <stdbool.h>
#include "ops_scalar_f32.h"

static const float weight_Conv_8_weight[20][1][5][5] = {
    {
        {
            {
                -0x1.012de7p-2f, -0x1.294a77p-2f, 0x1.6d9813p-4f, 0x1.1947f9p-2f, 0x1.21cd4bp-3f
            },
            {
                -0x1.5237a4p-3f, -0x1.08c456p-3f, 0x1.2ae7e9p-2f, 0x1.0344c0p-1f, 0x1.735e1dp-3f
            },
            {
                ...
            }
        }
    }
};

static const float weight_Conv_10_weight[12][20][3][3] = {
    {
        {
            {
                -0x1.29fb62p-3f, -0x1.73fab8p-4f, 0x1.177409p-4f
            },
            {
                0x1.74ded0p-4f, 0x1.55001dp-4f, 0x1.14d81bp-4f
            },
            {
                0x1.62ad76p-4f, 0x1.214d93p-4f, 0x1.2a39bap-5f
            }
        },
        {
            {
                0x1.7e125ap-9f, ...
            }
        }
    }
};

static const int64_t weight_tensor_constant0[1] = {
    1
};

static const int64_t weight_tensor_constant1[1] = {
    -1
};

static const float weight_Gemm_network_output_weight[10][108] = {
    {
        -0x1.5ffb6ep-4f, -0x1.36f87dp-4f, -0x1.2bf395p-3f, -0x1.456cc4p-3f, 0x1.5894a3p-6f, -0x1.7af660p-5f, -0x1.1c7d25p-3f, 0x1.3819e1p-4f,
        0x1.1eaf43p-6f, -0x1.1e73d9p-4f, ...
    }
};

static const float weight_Gemm_network_output_bias[10] = {
    -0x1.33b636p-2f, 0x1.5cf820p-4f, 0x1.4eb986p-4f, 0x1.404075p-7f, -0x1.639f6ap-4f, 0x1.5646ccp-2f, -0x1.1be03ep-3f, 0x1.0e9310p-7f,
    -0x1.553391p-4f, 0x1.1ae21ep-4f
};

void node1_conv2d_f32(const float input[1][1][14][14], const float weight[20][1][5][5], float out[1][20][5][5]) {
    int64_t in_per_group = 1 / 1;
    int64_t out_per_group = 20 / 1;
    int64_t out_pad_h = 0;
    int64_t out_pad_w = 0;
    (void)out_pad_h;
    (void)out_pad_w;
    for (int64_t n = 0; n < 1; ++n) {
        for (int64_t oc = 0; oc < 20; ++oc) {
            int64_t group = oc / out_per_group;
            for (int64_t oh = 0; oh < 5; ++oh) {
                for (int64_t ow = 0; ow < 5; ++ow) {
                    float acc = 0.0f;
                    int64_t in_h_base = oh * 2 - 0;
                    int64_t in_w_base = ow * 2 - 0;
                    for (int64_t ic = 0; ic < in_per_group; ++ic) {
                        int64_t in_c = group * in_per_group + ic;
                        for (int64_t kh = 0; kh < 5; ++kh) {
                            int64_t in_h_idx = in_h_base + kh * 1;
                            if (in_h_idx < 0 || in_h_idx >= 14) {
                                continue;
                            }
                            for (int64_t kw = 0; kw < 5; ++kw) {
                                int64_t in_w_idx = in_w_base + kw * 1;
                                if (in_w_idx < 0 || in_w_idx >= 14) {
                                    continue;
                                }
                                acc += input[n][in_c][in_h_idx][in_w_idx] *
                                    weight[oc][ic][kh][kw];
                            }
                        }
                    }
                    out[n][oc][oh][ow] = acc;
                }
            }
        }
    }
}

void node2_relu_f32(const float a[1][20][5][5], float out[1][20][5][5]) {
    for (int64_t i0 = 0; i0 < 1; ++i0) {
        for (int64_t i1 = 0; i1 < 20; ++i1) {
            for (int64_t i2 = 0; i2 < 5; ++i2) {
                for (int64_t i3 = 0; i3 < 5; ++i3) {
                    out[i0][i1][i2][i3] = ref_scalar_f32_relu(a[i0][i1][i2][i3]);
                }
            }
        }
    }
}

void node3_conv2d_f32(const float input[1][20][5][5], const float weight[12][20][3][3], float out[1][12][3][3]) {
    int64_t in_per_group = 20 / 1;
    int64_t out_per_group = 12 / 1;
    int64_t out_pad_h = 0;
    int64_t out_pad_w = 0;
    (void)out_pad_h;
    (void)out_pad_w;
    for (int64_t n = 0; n < 1; ++n) {
        for (int64_t oc = 0; oc < 12; ++oc) {
            int64_t group = oc / out_per_group;
            for (int64_t oh = 0; oh < 3; ++oh) {
                for (int64_t ow = 0; ow < 3; ++ow) {
                    float acc = 0.0f;
                    int64_t in_h_base = oh * 1 - 0;
                    int64_t in_w_base = ow * 1 - 0;
                    for (int64_t ic = 0; ic < in_per_group; ++ic) {
                        int64_t in_c = group * in_per_group + ic;
                        for (int64_t kh = 0; kh < 3; ++kh) {
                            int64_t in_h_idx = in_h_base + kh * 1;
                            if (in_h_idx < 0 || in_h_idx >= 5) {
                                continue;
                            }
                            for (int64_t kw = 0; kw < 3; ++kw) {
                                int64_t in_w_idx = in_w_base + kw * 1;
                                if (in_w_idx < 0 || in_w_idx >= 5) {
                                    continue;
                                }
                                acc += input[n][in_c][in_h_idx][in_w_idx] *
                                    weight[oc][ic][kh][kw];
                            }
                        }
                    }
                    out[n][oc][oh][ow] = acc;
                }
            }
        }
    }
}

void node4_reshape_f32(const float a[1][12][3][3], float out[1][108]) {
    const float* a_ptr = (const float*)a;
    for (int64_t i0 = 0; i0 < 1; ++i0) {
        for (int64_t i1 = 0; i1 < 108; ++i1) {
            int64_t offset = i0 * 108 + i1 * 1;
            out[i0][i1] = a_ptr[offset];
        }
    }
}

void node5_linear_f32(const float input[1][108], const float weight[10][108], const float bias[10], float out[1][10]) {
    for (int64_t i0 = 0; i0 < 1; ++i0) {
        for (int64_t j = 0; j < 10; ++j) {
            float acc = 0.0f;
            for (int64_t t = 0; t < 108; ++t) {
                acc += input[i0][t] * weight[j][t];
            }
            out[i0][j] = acc + bias[j];
        }
    }
}

void ref_codegen_main_f32(const float input_0[1][1][14][14], const float input_1[20][1][5][5], const float input_2[12][20][3][3], const int64_t input_3[1], const int64_t input_4[1], const float input_5[10][108], const float input_6[10], float out[1][10]) {
    float tmp_0[1][20][5][5];
    float tmp_1[1][20][5][5];
    float tmp_2[1][12][3][3];
    float tmp_3[1][108];
    node1_conv2d_f32(input_0, input_1, tmp_0);
    node2_relu_f32(tmp_0, tmp_1);
    node3_conv2d_f32(tmp_1, input_2, tmp_2);
    node4_reshape_f32(tmp_2, tmp_3);
    node5_linear_f32(tmp_3, input_5, input_6, out);
}

void entry(const float* in0, float* out0) {
    ref_codegen_main_f32(in0, weight_Conv_8_weight, weight_Conv_10_weight, weight_tensor_constant0, weight_tensor_constant1, weight_Gemm_network_output_weight, weight_Gemm_network_output_bias, out0);
}
