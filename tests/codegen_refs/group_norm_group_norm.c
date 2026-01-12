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
* op: native_group_norm (kind: group_norm)
* inputs: [shape=(2, 2, 2, 2), size=16, shape=(2,), size=2, shape=(2,), size=2]
* output: shape=(2, 2, 2, 2), size=16
* params: {'groups': 1, 'eps': 1e-05, 'has_weight': True, 'has_bias': True, 'N': 2, 'C': 2, 'HxW': 4}
*/
void node1_native_group_norm_f32(const float input[2][2][2][2], const float weight[2], const float bias[2], float out[2][2][2][2]) {
    const ssize_t batch = 2;
    const ssize_t channels = 2;
    const ssize_t spatial = 4;
    const ssize_t groups = 1;
    const ssize_t channels_per_group = channels / groups;
    const float *input_ptr = &input[0][0][0][0];
    float *out_ptr = &out[0][0][0][0];
    const float *weight_ptr = &weight[0];
    const float *bias_ptr = &bias[0];
    for (ssize_t n = 0; n < batch; ++n) {
        for (ssize_t g = 0; g < groups; ++g) {
            const ssize_t c_start = g * channels_per_group;
            const ssize_t group_offset = (n * channels + c_start) * spatial;
            float mean = 0;
            for (ssize_t c = 0; c < channels_per_group; ++c) {
                const ssize_t base = group_offset + c * spatial;
                for (ssize_t s = 0; s < spatial; ++s) {
                    mean += input_ptr[base + s];
                }
            }
            const float group_size = (float)(channels_per_group * spatial);
            mean /= group_size;
            float var = 0;
            for (ssize_t c = 0; c < channels_per_group; ++c) {
                const ssize_t base = group_offset + c * spatial;
                for (ssize_t s = 0; s < spatial; ++s) {
                    float diff = input_ptr[base + s] - mean;
                    var += diff * diff;
                }
            }
            var /= group_size;
            float rstd = (float)1 / ref_scalar_f32_sqrt(var + 1e-05f);
            for (ssize_t c = 0; c < channels_per_group; ++c) {
                const ssize_t channel = c_start + c;
                float w = weight_ptr[channel];
                float b = bias_ptr[channel];
                const ssize_t base = (n * channels + channel) * spatial;
                for (ssize_t s = 0; s < spatial; ++s) {
                    float val = (input_ptr[base + s] - mean) * rstd;
                    out_ptr[base + s] = val * w + b;
                }
            }
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][2][2][2], const float input_1[2], const float input_2[2], float out[2][2][2][2]) {
    node1_native_group_norm_f32(input_0, input_1, input_2, out);
}
