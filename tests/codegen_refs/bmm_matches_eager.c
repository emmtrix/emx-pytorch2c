#include <stdint.h>

void node1_bmm_f32(const float a[2][3][4], const float b[2][4][5], float out[2][3][5]) {
    for (int64_t b_idx = 0; b_idx < 2; ++b_idx) {
        for (int64_t i = 0; i < 3; ++i) {
            for (int64_t j = 0; j < 5; ++j) {
                float acc = 0.0f;
                for (int64_t t = 0; t < 4; ++t) {
                    acc += a[b_idx][i][t] * b[b_idx][t][j];
                }
                out[b_idx][i][j] = acc;
            }
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][3][4], const float input_1[2][4][5], float out[2][3][5]) {
    node1_bmm_f32(input_0, input_1, out);
}
