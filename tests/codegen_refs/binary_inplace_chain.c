#include <stdint.h>
#include <sys/types.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
#ifndef REF_PI_F
#define REF_PI_F 3.14159265358979323846f
#endif
#ifndef REF_PI_D
#define REF_PI_D 3.14159265358979323846
#endif

static inline float ref_scalar_f32_atan(float a) {
    return atanf(a);
}

static inline float ref_scalar_f32_add(float a, float b) {
    return a + b;
}

static inline float ref_scalar_f32_mul(float a, float b) {
    return a * b;
}

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
