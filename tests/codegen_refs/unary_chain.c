#include <stdint.h>
#include "ops_scalar.h"

void node1_silu_f32(const float a[2][3], float out[2][3]) {
    for (int64_t i0 = 0; i0 < 2; ++i0) {
        for (int64_t i1 = 0; i1 < 3; ++i1) {
            out[i0][i1] = ref_scalar_silu(a[i0][i1]);
        }
    }
}

void node2_hardsigmoid_f32(const float a[2][3], float out[2][3]) {
    for (int64_t i0 = 0; i0 < 2; ++i0) {
        for (int64_t i1 = 0; i1 < 3; ++i1) {
            out[i0][i1] = ref_scalar_hardsigmoid(a[i0][i1]);
        }
    }
}

void node3_hardswish_f32(const float a[2][3], float out[2][3]) {
    for (int64_t i0 = 0; i0 < 2; ++i0) {
        for (int64_t i1 = 0; i1 < 3; ++i1) {
            out[i0][i1] = ref_scalar_hardswish(a[i0][i1]);
        }
    }
}

void node4_mish_f32(const float a[2][3], float out[2][3]) {
    for (int64_t i0 = 0; i0 < 2; ++i0) {
        for (int64_t i1 = 0; i1 < 3; ++i1) {
            out[i0][i1] = ref_scalar_mish(a[i0][i1]);
        }
    }
}

void node5_softshrink_f32(const float a[2][3], float out[2][3]) {
    for (int64_t i0 = 0; i0 < 2; ++i0) {
        for (int64_t i1 = 0; i1 < 3; ++i1) {
            out[i0][i1] = ref_scalar_softshrink(a[i0][i1]);
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][3], float out[2][3]) {
    float tmp_0[2][3];
    float tmp_1[2][3];
    float tmp_2[2][3];
    float tmp_3[2][3];
    node1_silu_f32(input_0, tmp_0);
    node2_hardsigmoid_f32(tmp_0, tmp_1);
    node3_hardswish_f32(tmp_1, tmp_2);
    node4_mish_f32(tmp_2, tmp_3);
    node5_softshrink_f32(tmp_3, out);
}
