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

static inline float ref_scalar_f32_sqrt(float a) {
    return sqrtf(a);
}

/*
* op: _pdist_forward (kind: pdist)
* inputs: [shape=(4, 3), size=12]
* output: shape=(6,), size=6
* params: {'p': 2.0}
*/
void node1__pdist_forward_f32(const float input[4][3], float out[6]) {
    ssize_t out_index = 0;
    for (ssize_t i = 0; i < 4; ++i) {
        for (ssize_t j = i + 1; j < 4; ++j) {
            float acc = 0;
            for (ssize_t k = 0; k < 3; ++k) {
                float diff = input[i][k] - input[j][k];
                acc += diff * diff;
            }
            out[out_index++] = ref_scalar_f32_sqrt(acc);
        }
    }
}

void ref_codegen_main_f32(const float input_0[4][3], float out[6]) {
    node1__pdist_forward_f32(input_0, out);
}
