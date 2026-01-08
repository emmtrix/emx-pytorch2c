#include <stdint.h>
#include <sys/types.h>
#include <stdbool.h>
#include "ops_scalar_f32.h"

void node1_sum_f32(const float a[4][3], float out[3]) {
    for (ssize_t i0 = 0; i0 < 3; ++i0) {
        float acc = 0.0f;
        for (ssize_t r0 = 0; r0 < 4; ++r0) {
            acc += ((float*)a)[r0 * 1 + i0 * 4];
        }
        out[i0] = acc;
    }
}

void ref_codegen_main_f32(const float input_0[4][3], float out[3]) {
    node1_sum_f32(input_0, out);
}
