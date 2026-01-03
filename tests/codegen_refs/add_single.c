#include <stdint.h>
#include "ops_scalar_f32.h"

void node1_add_f32(const float a[2][3], const float b[2][3], float out[2][3]) {
    for (int64_t i0 = 0; i0 < 2; ++i0) {
        for (int64_t i1 = 0; i1 < 3; ++i1) {
            out[i0][i1] = ref_scalar_f32_add(a[i0][i1], b[i0][i1]);
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][3], const float input_1[2][3], float out[2][3]) {
    node1_add_f32(input_0, input_1, out);
}
