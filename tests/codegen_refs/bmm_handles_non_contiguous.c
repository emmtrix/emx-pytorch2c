#include <stdint.h>

void ref_codegen_main_f32(const float input_0[2][4][3], const float input_1[2][3][5], float out[2][4][5]) {
    for (int64_t b = 0; b < 2; ++b) {
        for (int64_t i = 0; i < 4; ++i) {
            for (int64_t j = 0; j < 5; ++j) {
                float acc = 0.0f;
                for (int64_t t = 0; t < 3; ++t) {
                    acc += input_0[b][i][t] * input_1[b][t][j];
                }
                out[b][i][j] = acc;
            }
        }
    }
}
