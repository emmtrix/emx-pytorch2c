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

void ref_codegen_main_f32(const float input_0[2][3], const float input_1[2][3], float out[2][3]) {
    node1_add_f32(input_0, input_1, out);
}
