#include <stdint.h>

void node1_matmul_f32(const float a[4][3], const float b[3][5], float out[4][5]) {
    for (int64_t i = 0; i < 4; ++i) {
        for (int64_t j = 0; j < 5; ++j) {
            float acc = 0.0f;
            for (int64_t t = 0; t < 3; ++t) {
                acc += a[i][t] * b[t][j];
            }
            out[i][j] = acc;
        }
    }
}

void ref_codegen_main_f32(const float input_0[4][3], const float input_1[3][5], float out[4][5]) {
    node1_matmul_f32(input_0, input_1, out);
}
