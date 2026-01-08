#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "ops_scalar_f32.h"

void node1_add_f32(int dim1, const float a[dim1][4], float out[dim1][4]) {
    for (size_t i0 = 0; i0 < dim1; ++i0) {
        for (size_t i1 = 0; i1 < 4; ++i1) {
            out[i0][i1] = ref_scalar_f32_add(a[i0][i1], 1.0f);
        }
    }
}

void ref_codegen_main_f32(int dim1, const float input_0[dim1][4], float out[dim1][4]) {
    node1_add_f32(dim1, input_0, out);
}

void model_run(int dim1, const float in0[dim1][4], float out0[dim1][4]) {
    ref_codegen_main_f32(dim1, in0, out0);
}
