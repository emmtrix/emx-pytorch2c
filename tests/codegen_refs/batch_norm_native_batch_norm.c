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
* op: _native_batch_norm_legit (kind: batch_norm)
* inputs: [shape=(2, 3, 2, 2), size=24, shape=(3,), size=3, shape=(3,), size=3, shape=(3,), size=3, shape=(3,), size=3]
* output: shape=(2, 3, 2, 2), size=24
* params: {'eps': 1e-05, 'momentum': 0.1, 'training': False, 'has_weight': True, 'has_bias': True}
*/
void node1__native_batch_norm_legit_f32(const float input[2][3][2][2], float running_mean[3], float running_var[3], const float weight[3], const float bias[3], float out[2][3][2][2]) {
    for (ssize_t c = 0; c < 3; ++c) {
        float mean = running_mean[c];
        float var = running_var[c];
        float inv = 1.0f / ref_scalar_f32_sqrt(var + 1e-05f);
        float scale = weight[c];
        float offset = bias[c];
        for (ssize_t n = 0; n < 2; ++n) {
            for (ssize_t i = 0; i < 4; ++i) {
                ssize_t idx = (n * 3 + c) * 4 + i;
                float value = ((float*)input)[idx];
                ((float*)out)[idx] = (value - mean) * inv * scale + offset;
            }
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][3][2][2], const float input_1[3], const float input_2[3], const float input_3[3], const float input_4[3], float out[2][3][2][2]) {
    node1__native_batch_norm_legit_f32(input_0, input_3, input_4, input_1, input_2, out);
}
