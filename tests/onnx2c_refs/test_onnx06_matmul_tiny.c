#include <stdint.h>
#include <sys/types.h>
#include <stdbool.h>
#include "ops_scalar_f32.h"

static const float weight_MatMul_my_weight[2][2] = {
    {
        0x1.000000p+0f, 0x0.0p+0f
    },
    {
        0x0.0p+0f, 0x1.000000p+0f
    }
};

/*
* op: linear (kind: linear)
* inputs: [shape=(1, 2), size=2, shape=(2, 2), size=4]
* output: shape=(1, 2), size=2
* params: {'has_bias': False}
*/
void node1_linear_f32(const float input[1][2], const float weight[2][2], float out[1][2]) {
    for (ssize_t i0 = 0; i0 < 1; ++i0) {
        for (ssize_t j = 0; j < 2; ++j) {
            float acc = 0.0f;
            for (ssize_t t = 0; t < 2; ++t) {
                acc += input[i0][t] * ((float*)weight)[j * 1 + t * 2];
            }
            out[i0][j] = acc + 0;
        }
    }
}

void ref_codegen_main_f32(const float input_0[1][2], const float input_1[2][2], float out[1][2]) {
    node1_linear_f32(input_0, input_1, out);
}

void entry(const float in0[1][2], float out0[1][2]) {
    ref_codegen_main_f32(in0, weight_MatMul_my_weight, out0);
}
