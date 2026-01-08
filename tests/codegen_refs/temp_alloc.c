#include <stdint.h>
#include <stdbool.h>
#include "ops_scalar_f32.h"
#include <stdlib.h>

void node1_add_f32(const float a[1][2][2][2], const float b[1][2][2][2], float out[1][2][2][2]) {
    for (int64_t i0 = 0; i0 < 1; ++i0) {
        for (int64_t i1 = 0; i1 < 2; ++i1) {
            for (int64_t i2 = 0; i2 < 2; ++i2) {
                for (int64_t i3 = 0; i3 < 2; ++i3) {
                    out[i0][i1][i2][i3] = ref_scalar_f32_add(a[0][i1][i2][i3], b[0][i1][i2][i3]);
                }
            }
        }
    }
}

void node2_add_f32(const float a[1][2][2][2], const float b[1][2][2][2], float out[1][2][2][2]) {
    for (int64_t i0 = 0; i0 < 1; ++i0) {
        for (int64_t i1 = 0; i1 < 2; ++i1) {
            for (int64_t i2 = 0; i2 < 2; ++i2) {
                for (int64_t i3 = 0; i3 < 2; ++i3) {
                    out[i0][i1][i2][i3] = ref_scalar_f32_add(a[0][i1][i2][i3], b[0][i1][i2][i3]);
                }
            }
        }
    }
}

void ref_codegen_main_f32(const float input_0[1][2][2][2], const float input_1[1][2][2][2], const float input_2[1][2][2][2], float out[1][2][2][2]) {
    float (*tmp_0)[2][2][2] = malloc(sizeof(float) * 1 * 2 * 2 * 2);
    node1_add_f32(input_0, input_1, tmp_0);
    node2_add_f32(tmp_0, input_2, out);
    free(tmp_0);
}
