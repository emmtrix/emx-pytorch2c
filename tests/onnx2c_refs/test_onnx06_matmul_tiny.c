#include <stdint.h>
#include <stdbool.h>
#include "ops_scalar_f32.h"

static const float weight_initializers_onnx_initializer_0[4] = {
    0x1.000000p+0f, 0x0.0p+0f, 0x0.0p+0f, 0x1.000000p+0f
};

void node1_matmul_f32(const float a[1][2], const float b[2][2], float out[1][2]) {
        for (int64_t i = 0; i < 1; ++i) {
            for (int64_t j = 0; j < 2; ++j) {
                float acc = 0.0f;
                for (int64_t t = 0; t < 2; ++t) {
                    acc += a[i][t] * b[t][j];
                }
                out[i][j] = acc;
            }
        }
}

void ref_codegen_main_f32(const float input_0[1][2], const float input_1[2][2], float out[1][2]) {
    node1_matmul_f32(input_0, input_1, out);
}

void model_run(const float* in0, float* out0) {
    ref_codegen_main_f32(in0, weight_initializers_onnx_initializer_0, out0);
}
