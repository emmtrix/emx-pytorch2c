#include <stdint.h>
#include <sys/types.h>
#include <stdbool.h>
#include <limits.h>
#include <stdint.h>
#ifndef REF_PI_F
#define REF_PI_F 3.14159265358979323846f
#endif
#ifndef REF_PI_D
#define REF_PI_D 3.14159265358979323846
#endif

static inline int32_t ref_scalar_i32_mul(int32_t a, int32_t b) {
    return a * b;
}

/*
* op: mul (kind: binary)
* inputs: [shape=(2, 3), size=6, shape=(2, 3), size=6]
* output: shape=(2, 3), size=6
* params: {}
*/
void node1_mul_i32(const int32_t a[2][3], const int32_t b[2][3], int32_t out[2][3]) {
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t i1 = 0; i1 < 3; ++i1) {
            out[i0][i1] = ref_scalar_i32_mul(a[i0][i1], b[i0][i1]);
        }
    }
}

/*
* op: mul (kind: binary)
* inputs: [shape=(2, 3), size=6, shape=(2, 3), size=6]
* output: shape=(2, 3), size=6
* params: {}
*/
void node2_mul_i32(const int32_t a[2][3], const int32_t b[2][3], int32_t out[2][3]) {
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t i1 = 0; i1 < 3; ++i1) {
            out[i0][i1] = ref_scalar_i32_mul(a[i0][i1], b[i0][i1]);
        }
    }
}

void ref_codegen_main_i32(const int32_t input_0[2][3], const int32_t input_1[2][3], const int32_t input_2[2][3], int32_t out[2][3]) {
    int32_t tmp_0[2][3];
    node1_mul_i32(input_0, input_1, tmp_0);
    node2_mul_i32(tmp_0, input_2, out);
}
