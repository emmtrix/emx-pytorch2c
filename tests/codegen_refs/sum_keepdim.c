#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "ops_scalar_f32.h"

void node1_sum_f32(const float a[2][3][4], float out[2][1][4]) {
    for (size_t i0 = 0; i0 < 2; ++i0) {
        for (size_t i1 = 0; i1 < 1; ++i1) {
            for (size_t i2 = 0; i2 < 4; ++i2) {
                float acc = 0.0f;
                for (size_t r1 = 0; r1 < 3; ++r1) {
                    acc += a[i0][r1][i2];
                }
                out[i0][i1][i2] = acc;
            }
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][3][4], float out[2][1][4]) {
    node1_sum_f32(input_0, out);
}
