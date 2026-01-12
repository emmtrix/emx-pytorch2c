#include <stdint.h>
#include <sys/types.h>
#include <stdbool.h>
#ifndef REF_PI_F
#define REF_PI_F 3.14159265358979323846f
#endif
#ifndef REF_PI_D
#define REF_PI_D 3.14159265358979323846
#endif


/*
* op: native_layer_norm_backward (kind: layer_norm_backward)
* inputs: [shape=(2, 4), size=8, shape=(2, 4), size=8, shape=(2, 1), size=2, shape=(2, 1), size=2, shape=(4,), size=4, shape=(4,), size=4]
* output: shape=(2, 4), size=8
* params: {'normalized_shape': (4,), 'has_weight': True, 'has_bias': True, 'output_mask': (True, True, True)}
*/
void node1_native_layer_norm_backward_f32(const float grad_output[2][4], const float input[2][4], const float mean[2][1], const float rstd[2][1], const float weight[4], const float bias[4], float out[2][4]) {
    const ssize_t outer_size = 2;
    const ssize_t inner_size = 4;
    const float *grad_ptr = &grad_output[0][0];
    const float *input_ptr = &input[0][0];
    const float *mean_ptr = &mean[0][0];
    const float *rstd_ptr = &rstd[0][0];
    float *out_ptr = &out[0][0];
    const float *weight_ptr = &weight[0];
    (void)bias;
    for (ssize_t outer = 0; outer < outer_size; ++outer) {
        float mean = mean_ptr[outer];
        float rstd = rstd_ptr[outer];
        float sum1 = 0;
        float sum2 = 0;
        for (ssize_t inner = 0; inner < inner_size; ++inner) {
            ssize_t idx = outer * inner_size + inner;
            float dy = grad_ptr[idx];
            float w = weight_ptr[inner];
            float x = input_ptr[idx];
            float dxhat = dy * w;
            sum1 += dxhat;
            sum2 += dxhat * (x - mean);
        }
        float inv_inner = (float)1 / (float)inner_size;
        for (ssize_t inner = 0; inner < inner_size; ++inner) {
            ssize_t idx = outer * inner_size + inner;
            float dy = grad_ptr[idx];
            float w = weight_ptr[inner];
            float x = input_ptr[idx];
            float dxhat = dy * w;
            float term = (dxhat - sum1 * inv_inner - (x - mean) * rstd * rstd * sum2 * inv_inner);
            out_ptr[idx] = term * rstd;
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][4], const float input_1[4], const float input_2[4], const float input_3[2][1], const float input_4[2][1], const float input_5[2][4], float out[2][4]) {
    node1_native_layer_norm_backward_f32(input_5, input_0, input_3, input_4, input_1, input_2, out);
}
