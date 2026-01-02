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

void ref_codegen_main_f32(const float input_0[2][3], const float input_1[3][4], float out[2][4]) {
    node1_matmul_f32(input_0, input_1, out);
}
