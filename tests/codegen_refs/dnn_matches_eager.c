#include <stdint.h>

void ref_codegen_main_f32(const float input_0[2][3], const float input_1[3][4], const float input_2[2][4], float out[2][4]) {
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 4; ++j) {
            float acc = 0.0f;
            for (int64_t t = 0; t < 3; ++t) {
                acc += input_0[i][t] * input_1[t][j];
            }
            float sum = acc + input_2[i][j];
            out[i][j] = sum > 0.0f ? sum : 0.0f;
        }
    }
}
