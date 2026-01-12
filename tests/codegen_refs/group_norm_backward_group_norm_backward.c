#include <stdint.h>
#include <sys/types.h>
#include <stdbool.h>


/*
* op: native_group_norm_backward (kind: group_norm_backward)
* inputs: [shape=(2, 2, 2, 2), size=16, shape=(2, 2, 2, 2), size=16, shape=(2, 1), size=2, shape=(2, 1), size=2, shape=(2,), size=2]
* output: shape=(2, 2, 2, 2), size=16
* params: {'groups': 1, 'has_weight': True, 'N': 2, 'C': 2, 'HxW': 4, 'output_mask': (True, True, True)}
*/
void node1_native_group_norm_backward_f32(const float grad_output[2][2][2][2], const float input[2][2][2][2], const float mean[2][1], const float rstd[2][1], const float weight[2], float out[2][2][2][2]) {
    const ssize_t batch = 2;
    const ssize_t channels = 2;
    const ssize_t spatial = 4;
    const ssize_t groups = 1;
    const ssize_t channels_per_group = channels / groups;
    const float *grad_ptr = &grad_output[0][0][0][0];
    const float *input_ptr = &input[0][0][0][0];
    const float *mean_ptr = &mean[0][0];
    const float *rstd_ptr = &rstd[0][0];
    float *out_ptr = &out[0][0][0][0];
    const float *weight_ptr = &weight[0];
    for (ssize_t n = 0; n < batch; ++n) {
        for (ssize_t g = 0; g < groups; ++g) {
            const ssize_t c_start = g * channels_per_group;
            float mean = mean_ptr[n * groups + g];
            float rstd = rstd_ptr[n * groups + g];
            float sum1 = 0;
            float sum2 = 0;
            for (ssize_t c = 0; c < channels_per_group; ++c) {
                const ssize_t channel = c_start + c;
                float w = weight_ptr[channel];
                const ssize_t base = (n * channels + channel) * spatial;
                for (ssize_t s = 0; s < spatial; ++s) {
                    const ssize_t idx = base + s;
                    float dy = grad_ptr[idx];
                    float dxhat = dy * w;
                    sum1 += dxhat;
                    sum2 += dxhat * (input_ptr[idx] - mean);
                }
            }
            const float inv_group = (float)1 / (float)(channels_per_group * spatial);
            for (ssize_t c = 0; c < channels_per_group; ++c) {
                const ssize_t channel = c_start + c;
                float w = weight_ptr[channel];
                const ssize_t base = (n * channels + channel) * spatial;
                for (ssize_t s = 0; s < spatial; ++s) {
                    const ssize_t idx = base + s;
                    float dy = grad_ptr[idx];
                    float dxhat = dy * w;
                    float term = (dxhat - sum1 * inv_group - (input_ptr[idx] - mean) * rstd * rstd * sum2 * inv_group);
                    out_ptr[idx] = term * rstd;
                }
            }
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][2][2][2], const float input_1[2], const float input_2[2][1], const float input_3[2][1], const float input_4[2][2][2][2], float out[2][2][2][2]) {
    node1_native_group_norm_backward_f32(input_4, input_0, input_2, input_3, input_1, out);
}
