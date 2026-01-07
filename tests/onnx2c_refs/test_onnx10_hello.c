#include <stdint.h>
#include <stdbool.h>
#include "ops_scalar_f32.h"

static const float weight_initializers_onnx_initializer_0[1][16] = {
    {
        0x1.669cb3p-3f, 0x1.27a1eap-2f, 0x1.213f30p-2f, 0x1.14a67fp-1f, 0x1.17f3f7p-1f, 0x1.40cb17p-3f, -0x1.7f934ep-3f, 0x1.06eb33p-3f,
        -0x1.4f4350p-4f, -0x1.6188dep-3f, ...
    }
};

static const float weight_initializers_onnx_initializer_1[16] = {
    -0x1.371d99p-2f, 0x1.709bddp-3f, -0x1.4167e1p-1f, -0x1.0bebccp-15f, 0x1.679513p-2f, -0x1.083b84p+0f, 0x0.0p+0f, -0x1.1b8891p-3f,
    0x0.0p+0f, 0x0.0p+0f, ...
};

static const float weight_initializers_onnx_initializer_2[16][16] = {
    {
        -0x1.1538d4p-2f, -0x1.3e85c5p-1f, 0x1.22840dp-6f, 0x1.56044dp-2f, -0x1.33085dp-2f, -0x1.7bd78ep-3f, -0x1.57e254p-4f, 0x1.2bc583p-4f,
        -0x1.1335bap-6f, -0x1.770053p-6f, ...
    }
};

static const float weight_initializers_onnx_initializer_3[16] = {
    0x0.0p+0f, 0x1.239f79p-4f, 0x1.249124p-3f, -0x1.2f5076p-2f, 0x0.0p+0f, 0x1.4cc498p-3f, 0x0.0p+0f, 0x1.0e848bp-3f,
    -0x1.4796a0p-2f, 0x1.3279a0p-2f, ...
};

static const float weight_initializers_onnx_initializer_4[16][1] = {
    {
        0x1.257da0p-2f
    },
    {
        0x1.26d0c7p+0f
    },
    {
        -0x1.218ba4p-2f
    },
    {
        0x1.1038d0p-1f
    },
    {
        0x1.774494p-2f
    },
    {
        -0x1.08a91bp-1f
    },
    {
        0x1.37f39ap-2f
    },
    {
        -0x1.50261ep-3f
    },
    {
        0x1.3b3c10p-2f
    },
    {
        -0x1.2fd25cp-1f
    },
    {
        ...
    }
};

static const float weight_initializers_onnx_initializer_5[1] = {
    -0x1.712c2bp-2f
};

void node1_matmul_f32(const float a[1][1], const float b[1][16], float out[1][16]) {
        for (int64_t i = 0; i < 1; ++i) {
            for (int64_t j = 0; j < 16; ++j) {
                float acc = 0.0f;
                for (int64_t t = 0; t < 1; ++t) {
                    acc += a[i][t] * b[t][j];
                }
                out[i][j] = acc;
            }
        }
}

void node2_add_f32(const float a[1][16], const float b[16], float out[1][16]) {
    for (int64_t i0 = 0; i0 < 1; ++i0) {
        for (int64_t i1 = 0; i1 < 16; ++i1) {
            out[i0][i1] = ref_scalar_f32_add(a[0][i1], b[i1]);
        }
    }
}

void node3_relu_f32(const float a[1][16], float out[1][16]) {
    for (int64_t i0 = 0; i0 < 1; ++i0) {
        for (int64_t i1 = 0; i1 < 16; ++i1) {
            out[i0][i1] = ref_scalar_f32_relu(a[i0][i1]);
        }
    }
}

void node4_matmul_f32(const float a[1][16], const float b[16][16], float out[1][16]) {
        for (int64_t i = 0; i < 1; ++i) {
            for (int64_t j = 0; j < 16; ++j) {
                float acc = 0.0f;
                for (int64_t t = 0; t < 16; ++t) {
                    acc += a[i][t] * b[t][j];
                }
                out[i][j] = acc;
            }
        }
}

void node5_add_f32(const float a[1][16], const float b[16], float out[1][16]) {
    for (int64_t i0 = 0; i0 < 1; ++i0) {
        for (int64_t i1 = 0; i1 < 16; ++i1) {
            out[i0][i1] = ref_scalar_f32_add(a[0][i1], b[i1]);
        }
    }
}

void node6_relu_f32(const float a[1][16], float out[1][16]) {
    for (int64_t i0 = 0; i0 < 1; ++i0) {
        for (int64_t i1 = 0; i1 < 16; ++i1) {
            out[i0][i1] = ref_scalar_f32_relu(a[i0][i1]);
        }
    }
}

void node7_matmul_f32(const float a[1][16], const float b[16][1], float out[1][1]) {
        for (int64_t i = 0; i < 1; ++i) {
            for (int64_t j = 0; j < 1; ++j) {
                float acc = 0.0f;
                for (int64_t t = 0; t < 16; ++t) {
                    acc += a[i][t] * b[t][j];
                }
                out[i][j] = acc;
            }
        }
}

void node8_add_f32(const float a[1][1], const float b[1], float out[1][1]) {
    for (int64_t i0 = 0; i0 < 1; ++i0) {
        for (int64_t i1 = 0; i1 < 1; ++i1) {
            out[i0][i1] = ref_scalar_f32_add(a[0][0], b[0]);
        }
    }
}

void ref_codegen_main_f32(const float input_0[1][1], const float input_1[1][16], const float input_2[16], const float input_3[16][16], const float input_4[16], const float input_5[16][1], const float input_6[1], float out[1][1]) {
    float tmp_0[1][16];
    float tmp_1[1][16];
    float tmp_2[1][16];
    float tmp_3[1][16];
    float tmp_4[1][16];
    float tmp_5[1][16];
    float tmp_6[1][1];
    node1_matmul_f32(input_0, input_1, tmp_0);
    node2_add_f32(tmp_0, input_2, tmp_1);
    node3_relu_f32(tmp_1, tmp_2);
    node4_matmul_f32(tmp_2, input_3, tmp_3);
    node5_add_f32(tmp_3, input_4, tmp_4);
    node6_relu_f32(tmp_4, tmp_5);
    node7_matmul_f32(tmp_5, input_5, tmp_6);
    node8_add_f32(tmp_6, input_6, out);
}

void entry(const float* in0, float* out0) {
    ref_codegen_main_f32(in0, weight_initializers_onnx_initializer_0, weight_initializers_onnx_initializer_1, weight_initializers_onnx_initializer_2, weight_initializers_onnx_initializer_3, weight_initializers_onnx_initializer_4, weight_initializers_onnx_initializer_5, out0);
}
