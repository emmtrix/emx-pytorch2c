#include <stdint.h>
#include <stdbool.h>
#include "ops_scalar_f32.h"

static const float weight_initializers_onnx_initializer_0[4] = {
    0x1.4ccccdp-4f, -0x1.4ccccdp-3f, 0x1.19999ap-2f, 0x0.0p+0f
};

void node1_add_f32(const float a[1][4], const float b[4], float out[1][4]) {
    for (int64_t i0 = 0; i0 < 1; ++i0) {
        for (int64_t i1 = 0; i1 < 4; ++i1) {
            out[i0][i1] = ref_scalar_f32_add(a[0][i1], b[i1]);
        }
    }
}

void ref_codegen_main_f32(const float input_0[1][4], const float input_1[4], float out[1][4]) {
    node1_add_f32(input_0, input_1, out);
}

void entry(const float* in0, float* out0) {
    ref_codegen_main_f32(in0, weight_initializers_onnx_initializer_0, out0);
}
