#include <stdint.h>
#include "ops_scalar.h"

void node1_mul_f32(const float a[4][4], const float b[4][4], float out[4][4]) {
    for (int64_t i0 = 0; i0 < 4; ++i0) {
        for (int64_t i1 = 0; i1 < 4; ++i1) {
            out[i0][i1] = ref_scalar_mul(a[i0][i1], b[i0][i1]);
        }
    }
}

void ref_codegen_main_f32(const float input_0[4][4], const float input_1[4][4], float out[4][4]) {
    node1_mul_f32(input_0, input_1, out);
}
