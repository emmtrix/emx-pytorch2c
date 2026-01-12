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

static inline float ref_scalar_f32_exp(float a) {
    return expf(a);
}

/*
* op: softmax (kind: softmax)
* inputs: [shape=(2, 3), size=6]
* output: shape=(2, 3), size=6
* params: {'dim': 1}
*/
void node1_softmax_f32(const float input[2][3], float out[2][3]) {
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t i1 = 0; i1 < 3; ++i1) {
            float max_val = input[i0][0];
            for (ssize_t r1 = 1; r1 < 3; ++r1) {
                float value = input[i0][r1];
                if (value > max_val) {
                    max_val = value;
                }
            }
            float sum = 0.0f;
            for (ssize_t r1 = 0; r1 < 3; ++r1) {
                sum += ref_scalar_f32_exp(input[i0][r1] - max_val);
            }
            ((float*)out)[i0 * 3 + i1 * 1] = ref_scalar_f32_exp(input[i0][i1] - max_val) / sum;
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][3], float out[2][3]) {
    node1_softmax_f32(input_0, out);
}
