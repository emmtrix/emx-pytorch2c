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

static inline float ref_scalar_f32_add(float a, float b) {
    return a + b;
}

static inline float ref_scalar_f32_relu(float a) {
    return a > 0.0f ? a : 0.0f;
}

static inline float ref_scalar_f32_sub(float a, float b) {
    return a - b;
}

/*
* op: add (kind: binary)
* inputs: [shape=(2, 3), size=6, shape=(2, 3), size=6]
* output: shape=(2, 3), size=6
* params: {}
*/
void node1_add_f32(const float a[2][3], const float b[2][3], float out[2][3]) {
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t i1 = 0; i1 < 3; ++i1) {
            out[i0][i1] = ref_scalar_f32_add(a[i0][i1], b[i0][i1]);
        }
    }
}

/*
* op: relu (kind: unary)
* inputs: [shape=(2, 3), size=6]
* output: shape=(2, 3), size=6
* params: {}
*/
void node2_relu_f32(const float a[2][3], float out[2][3]) {
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t i1 = 0; i1 < 3; ++i1) {
            out[i0][i1] = ref_scalar_f32_relu(a[i0][i1]);
        }
    }
}

/*
* op: sub (kind: binary)
* inputs: [shape=(2, 3), size=6, shape=(2, 3), size=6]
* output: shape=(2, 3), size=6
* params: {}
*/
void node3_sub_f32(const float a[2][3], const float b[2][3], float out[2][3]) {
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t i1 = 0; i1 < 3; ++i1) {
            out[i0][i1] = ref_scalar_f32_sub(a[i0][i1], b[i0][i1]);
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][3], const float input_1[2][3], const float input_2[2][3], float out[2][3]) {
    float tmp_0[2][3];
    float tmp_1[2][3];
    node1_add_f32(input_0, input_1, tmp_0);
    node2_relu_f32(tmp_0, tmp_1);
    node3_sub_f32(tmp_1, input_2, out);
}
