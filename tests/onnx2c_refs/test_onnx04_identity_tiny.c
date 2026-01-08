#include <stdint.h>
#include <stdbool.h>
#include "ops_scalar_f32.h"

void node1_clone_f32(const float a[1][3], float out[1][3]) {
    for (int64_t i0 = 0; i0 < 1; ++i0) {
        for (int64_t i1 = 0; i1 < 3; ++i1) {
            out[i0][i1] = a[i0][i1];
        }
    }
}

void ref_codegen_main_f32(const float input_0[1][3], float out[1][3]) {
    node1_clone_f32(input_0, out);
}

void entry(const float in0[1][3], float out0[1][3]) {
    ref_codegen_main_f32(in0, out0);
}
