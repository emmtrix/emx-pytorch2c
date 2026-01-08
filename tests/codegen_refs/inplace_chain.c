#include <stdint.h>
#include <sys/types.h>
#include <stdbool.h>
#include "ops_scalar_f32.h"

/*
* op: atan (kind: unary)
* inputs: [shape=(2, 3), size=6]
* output: shape=(2, 3), size=6
* params: {}
*/
void node1_atan_f32(const float a[2][3], float out[2][3]) {
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t i1 = 0; i1 < 3; ++i1) {
            out[i0][i1] = ref_scalar_f32_atan(a[i0][i1]);
        }
    }
}

/*
* op: add (kind: binary)
* inputs: [shape=(2, 3), size=6, shape=(2, 3), size=6]
* output: shape=(2, 3), size=6
* params: {}
*/
void node2_add_f32(const float a[2][3], const float b[2][3], float out[2][3]) {
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t i1 = 0; i1 < 3; ++i1) {
            out[i0][i1] = ref_scalar_f32_add(a[i0][i1], b[i0][i1]);
        }
    }
}

/*
* op: mul (kind: binary)
* inputs: [shape=(2, 3), size=6, shape=(2, 3), size=6]
* output: shape=(2, 3), size=6
* params: {}
*/
void node3_mul_f32(const float a[2][3], const float b[2][3], float out[2][3]) {
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t i1 = 0; i1 < 3; ++i1) {
            out[i0][i1] = ref_scalar_f32_mul(a[i0][i1], b[i0][i1]);
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][3], float out[2][3]) {
    float tmp_0[2][3];
    node1_atan_f32(input_0, tmp_0);
    node2_add_f32(tmp_0, input_0, tmp_0);
    node3_mul_f32(tmp_0, input_0, out);
}
