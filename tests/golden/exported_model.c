#include <stdint.h>
#include <stdbool.h>
#include "ops_scalar_f32.h"

static const float weight_bias[3] = {
    -0x1.5b4f07p-1f, 0x1.0ce099p+0f, -0x1.091cabp+0f
};

static const float weight_weight[12] = {
    0x1.453f5cp+0f, -0x1.163c50p-2f, -0x1.0b7149p+1f, 0x1.1184b6p-1f, -0x1.0ad1a1p+0f, -0x1.33052dp+0f, 0x1.4e837ap-2f, 0x1.5688e5p-1f,
    -0x1.382144p-1f, -0x1.4e830bp-2f, -0x1.18bd18p-1f, 0x1.3a67c6p-3f
};

void node1_addmm_f32(const float input[3], const float mat1[2][4], const float mat2[4][3], float out[2][3]) {
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 3; ++j) {
            float acc = 0.0f;
            for (int64_t t = 0; t < 4; ++t) {
                acc += mat1[i][t] * mat2[t][j];
            }
            out[i][j] = (1.0f) * ((float*)input)[j * 1] + (1.0f) * acc;
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][4], const float input_1[3], const float input_2[4][3], float out[2][3]) {
    node1_addmm_f32(input_1, input_0, input_2, out);
}

void model_run(const float* in0, float* out0) {
    ref_codegen_main_f32(in0, weight_bias, weight_weight, out0);
}
