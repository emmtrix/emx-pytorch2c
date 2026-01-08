#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "ops_scalar_f32.h"

void node1_mul_f32(const float a[2][3], const float b[2][3], float out[2][3]) {
    for (size_t i0 = 0; i0 < 2; ++i0) {
        for (size_t i1 = 0; i1 < 3; ++i1) {
            out[i0][i1] = ref_scalar_f32_mul(a[i0][i1], b[i0][i1]);
        }
    }
}

void node2_mul_f32(const float a[2][3], const float b[2][3], float out[2][3]) {
    for (size_t i0 = 0; i0 < 2; ++i0) {
        for (size_t i1 = 0; i1 < 3; ++i1) {
            out[i0][i1] = ref_scalar_f32_mul(a[i0][i1], b[i0][i1]);
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][3], const float input_1[2][3], const float input_2[2][3], float out[2][3]) {
    float tmp_0[2][3];
    node1_mul_f32(input_0, input_1, tmp_0);
    node2_mul_f32(tmp_0, input_2, out);
}
