#include <stdint.h>
#include <sys/types.h>
#include <stdbool.h>
#include "ops_scalar_f32.h"

void node1_sum_f32(const float a[2][3][4], float out[2][4]) {
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t i1 = 0; i1 < 4; ++i1) {
            float acc = 0.0f;
            for (ssize_t r1 = 0; r1 < 3; ++r1) {
                acc += a[i0][r1][i1];
            }
            out[i0][i1] = acc;
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][3][4], float out[2][4]) {
    node1_sum_f32(input_0, out);
}
