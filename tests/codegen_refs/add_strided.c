#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "ops_scalar_f32.h"

void node1_add_f32(const float a[3][2], const float b[3][2], float out[3][2]) {
    for (size_t i0 = 0; i0 < 3; ++i0) {
        for (size_t i1 = 0; i1 < 2; ++i1) {
            out[i0][i1] = ref_scalar_f32_add(((float*)a)[i0 * 1 + i1 * 3], ((float*)b)[i0 * 1 + i1 * 3]);
        }
    }
}

void ref_codegen_main_f32(const float input_0[3][2], const float input_1[3][2], float out[3][2]) {
    node1_add_f32(input_0, input_1, out);
}
