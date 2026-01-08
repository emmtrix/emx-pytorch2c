#include <stdint.h>
#include <sys/types.h>
#include <stdbool.h>
#include "ops_scalar_f32.h"

void node1_atan_f32(const float a[2][3], float out[2][3]) {
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t i1 = 0; i1 < 3; ++i1) {
            out[i0][i1] = ref_scalar_f32_atan(a[i0][i1]);
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][3], float out[2][3]) {
    node1_atan_f32(input_0, out);
}
