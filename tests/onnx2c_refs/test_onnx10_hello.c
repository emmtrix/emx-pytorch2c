#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "ops_scalar_f32.h"

static const float weight_MatMul_biased_tensor_name1_weight[16][1] = {
    {
        0x1.669cb3p-3f
    },
    {
        0x1.27a1eap-2f
    },
    {
        0x1.213f30p-2f
    },
    {
        0x1.14a67fp-1f
    },
    {
        0x1.17f3f7p-1f
    },
    {
        0x1.40cb17p-3f
    },
    {
        -0x1.7f934ep-3f
    },
    {
        0x1.06eb33p-3f
    },
    {
        -0x1.4f4350p-4f
    },
    {
        -0x1.6188dep-3f
    },
    {
        ...
    }
};

static const float weight_MatMul_biased_tensor_name1_bias[16] = {
    -0x1.371d99p-2f, 0x1.709bddp-3f, -0x1.4167e1p-1f, -0x1.0bebccp-15f, 0x1.679513p-2f, -0x1.083b84p+0f, 0x0.0p+0f, -0x1.1b8891p-3f,
    0x0.0p+0f, 0x0.0p+0f, ...
};

static const float weight_MatMul_biased_tensor_name2_weight[16][16] = {
    {
        -0x1.1538d4p-2f, 0x1.393080p-5f, 0x1.5320c0p-5f, -0x1.700600p-10f, -0x1.4da03ep-2f, -0x1.223b96p-2f, 0x1.4a591bp-2f, -0x1.546b2bp-2f,
        -0x1.16949ap-2f, 0x1.22a32fp-2f, ...
    }
};

static const float weight_MatMul_biased_tensor_name2_bias[16] = {
    0x0.0p+0f, 0x1.239f79p-4f, 0x1.249124p-3f, -0x1.2f5076p-2f, 0x0.0p+0f, 0x1.4cc498p-3f, 0x0.0p+0f, 0x1.0e848bp-3f,
    -0x1.4796a0p-2f, 0x1.3279a0p-2f, ...
};

static const float weight_MatMul_dense_4_weight[1][16] = {
    {
        0x1.257da0p-2f, 0x1.26d0c7p+0f, -0x1.218ba4p-2f, 0x1.1038d0p-1f, 0x1.774494p-2f, -0x1.08a91bp-1f, 0x1.37f39ap-2f, -0x1.50261ep-3f,
        0x1.3b3c10p-2f, -0x1.2fd25cp-1f, ...
    }
};

static const float weight_MatMul_dense_4_bias[1] = {
    -0x1.712c2bp-2f
};

void node1_linear_f32(const float input[1][1], const float weight[16][1], const float bias[16], float out[1][16]) {
    for (size_t i0 = 0; i0 < 1; ++i0) {
        for (size_t j = 0; j < 16; ++j) {
            float acc = 0.0f;
            for (size_t t = 0; t < 1; ++t) {
                acc += input[i0][t] * weight[j][t];
            }
            out[i0][j] = acc + bias[j];
        }
    }
}

void node2_relu_f32(const float a[1][16], float out[1][16]) {
    for (size_t i0 = 0; i0 < 1; ++i0) {
        for (size_t i1 = 0; i1 < 16; ++i1) {
            out[i0][i1] = ref_scalar_f32_relu(a[i0][i1]);
        }
    }
}

void node3_linear_f32(const float input[1][16], const float weight[16][16], const float bias[16], float out[1][16]) {
    for (size_t i0 = 0; i0 < 1; ++i0) {
        for (size_t j = 0; j < 16; ++j) {
            float acc = 0.0f;
            for (size_t t = 0; t < 16; ++t) {
                acc += input[i0][t] * ((float*)weight)[j * 1 + t * 16];
            }
            out[i0][j] = acc + bias[j];
        }
    }
}

void node4_relu_f32(const float a[1][16], float out[1][16]) {
    for (size_t i0 = 0; i0 < 1; ++i0) {
        for (size_t i1 = 0; i1 < 16; ++i1) {
            out[i0][i1] = ref_scalar_f32_relu(a[i0][i1]);
        }
    }
}

void node5_linear_f32(const float input[1][16], const float weight[1][16], const float bias[1], float out[1][1]) {
    for (size_t i0 = 0; i0 < 1; ++i0) {
        for (size_t j = 0; j < 1; ++j) {
            float acc = 0.0f;
            for (size_t t = 0; t < 16; ++t) {
                acc += input[i0][t] * weight[j][t];
            }
            out[i0][j] = acc + bias[j];
        }
    }
}

void ref_codegen_main_f32(const float input_0[1][1], const float input_1[16][1], const float input_2[16], const float input_3[16][16], const float input_4[16], const float input_5[1][16], const float input_6[1], float out[1][1]) {
    float tmp_0[1][16];
    float tmp_1[1][16];
    float tmp_2[1][16];
    float tmp_3[1][16];
    node1_linear_f32(input_0, input_1, input_2, tmp_0);
    node2_relu_f32(tmp_0, tmp_1);
    node3_linear_f32(tmp_1, input_3, input_4, tmp_2);
    node4_relu_f32(tmp_2, tmp_3);
    node5_linear_f32(tmp_3, input_5, input_6, out);
}

void entry(const float in0[1][1], float out0[1][1]) {
    ref_codegen_main_f32(in0, weight_MatMul_biased_tensor_name1_weight, weight_MatMul_biased_tensor_name1_bias, weight_MatMul_biased_tensor_name2_weight, weight_MatMul_biased_tensor_name2_bias, weight_MatMul_dense_4_weight, weight_MatMul_dense_4_bias, out0);
}
