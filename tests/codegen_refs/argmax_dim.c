#include <stdint.h>
#include "ops_scalar_f32.h"

void node1_argmax_f32(const float a[2][3], int64_t out[2]) {
    for (int64_t i0 = 0; i0 < 2; ++i0) {
        float best_value = a[i0][0];
        int64_t best_index = 0;
        for (int64_t r1 = 1; r1 < 3; ++r1) {
            float value = a[i0][r1];
            if (value > best_value) {
                best_value = value;
                best_index = r1;
            }
        }
        out[i0] = best_index;
    }
}

void ref_codegen_main_f32(const float input_0[2][3], int64_t out[2]) {
    node1_argmax_f32(input_0, out);
}
