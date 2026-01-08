#include <stdint.h>
#include <sys/types.h>
#include <stdbool.h>
#include "ops_scalar_f32.h"

void node1_mean_f32(const float a[2][3], float out[1]) {
    float acc = 0.0f;
    for (ssize_t r0 = 0; r0 < 2; ++r0) {
        for (ssize_t r1 = 0; r1 < 3; ++r1) {
            acc += a[r0][r1];
        }
    }
    acc /= (float)6;
    out[0] = acc;
}

void ref_codegen_main_f32(const float input_0[2][3], float out[1]) {
    node1_mean_f32(input_0, out);
}
