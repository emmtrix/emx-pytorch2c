#include <stdint.h>

void node1_matmul_f32(const float a[2][3], const float b[3][4], float out[2][4]) {
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 4; ++j) {
            float acc = 0.0f;
            for (int64_t t = 0; t < 3; ++t) {
                acc += a[i][t] * b[t][j];
            }
            out[i][j] = acc;
        }
    }
}

void node2_add_f32(const float a[2][4], const float b[2][4], float out[2][4]) {
    for (int64_t i0 = 0; i0 < 2; ++i0) {
        for (int64_t i1 = 0; i1 < 4; ++i1) {
            out[i0][i1] = a[i0][i1] + b[i0][i1];
        }
    }
}

void node3_relu_f32(const float a[2][4], float out[2][4]) {
    for (int64_t i0 = 0; i0 < 2; ++i0) {
        for (int64_t i1 = 0; i1 < 4; ++i1) {
            out[i0][i1] = a[i0][i1] > 0.0f ? a[i0][i1] : 0.0f;
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][3], const float input_1[3][4], const float input_2[2][4], float out[2][4]) {
    float tmp_0[2][4];
    float tmp_1[2][4];
    node1_matmul_f32(input_0, input_1, tmp_0);
    node2_add_f32(tmp_0, input_2, tmp_1);
    node3_relu_f32(tmp_1, out);
}
