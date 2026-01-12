#include <stdint.h>
#include <sys/types.h>
#include <stdbool.h>
#include <stdlib.h>
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

/*
* op: add (kind: binary)
* inputs: [shape=(1,), size=1, shape=(1,), size=1]
* output: shape=(1,), size=1
* params: {}
*/
void node1_add_f32(const float a[1], const float b[1], float out[1]) {
    for (ssize_t i0 = 0; i0 < 1; ++i0) {
        out[i0] = ref_scalar_f32_add(a[0], b[0]);
    }
}

/*
* op: add (kind: binary)
* inputs: [shape=(1, 2, 2, 5), size=20, shape=(1, 2, 2, 5), size=20]
* output: shape=(1, 2, 2, 5), size=20
* params: {}
*/
void node2_add_f32(const float a[1][2][2][5], const float b[1][2][2][5], float out[1][2][2][5]) {
    for (ssize_t i0 = 0; i0 < 1; ++i0) {
        for (ssize_t i1 = 0; i1 < 2; ++i1) {
            for (ssize_t i2 = 0; i2 < 2; ++i2) {
                for (ssize_t i3 = 0; i3 < 5; ++i3) {
                    out[i0][i1][i2][i3] = ref_scalar_f32_add(a[0][i1][i2][i3], b[0][i1][i2][i3]);
                }
            }
        }
    }
}

/*
* op: add (kind: binary)
* inputs: [shape=(1,), size=1, shape=(1, 2, 2, 5), size=20]
* output: shape=(1, 2, 2, 5), size=20
* params: {}
*/
void node3_add_f32(const float a[1], const float b[1][2][2][5], float out[1][2][2][5]) {
    for (ssize_t i0 = 0; i0 < 1; ++i0) {
        for (ssize_t i1 = 0; i1 < 2; ++i1) {
            for (ssize_t i2 = 0; i2 < 2; ++i2) {
                for (ssize_t i3 = 0; i3 < 5; ++i3) {
                    out[i0][i1][i2][i3] = ref_scalar_f32_add(a[0], b[0][i1][i2][i3]);
                }
            }
        }
    }
}

void ref_codegen_main_f32(const float input_0[1], const float input_1[1], const float input_2[1][2][2][5], const float input_3[1][2][2][5], float out[1][2][2][5]) {
    float tmp_0[1];
    float (*tmp_1)[2][2][5] = malloc(sizeof(float) * 1 * 2 * 2 * 5);
    node1_add_f32(input_0, input_1, tmp_0);
    node2_add_f32(input_2, input_3, tmp_1);
    node3_add_f32(tmp_0, tmp_1, out);
    free(tmp_1);
}
