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
* op: native_layer_norm (kind: layer_norm)
* inputs: [shape=(2, 4), size=8, shape=(4,), size=4, shape=(4,), size=4]
* output: shape=(2, 4), size=8
* params: {'normalized_shape': (4,), 'eps': 1e-05, 'has_weight': True, 'has_bias': True}
*/
void node1_native_layer_norm_f32(const float input[2][4], const float weight[4], const float bias[4], float out[2][4]) {
    const ssize_t outer_size = 2;
    const ssize_t inner_size = 4;
    const float *input_ptr = &input[0][0];
    float *out_ptr = &out[0][0];
    const float *weight_ptr = &weight[0];
    const float *bias_ptr = &bias[0];
    for (ssize_t outer = 0; outer < outer_size; ++outer) {
        float mean = 0;
        for (ssize_t inner = 0; inner < inner_size; ++inner) {
            mean += input_ptr[outer * inner_size + inner];
        }
        mean /= (float)inner_size;
        float var = 0;
        for (ssize_t inner = 0; inner < inner_size; ++inner) {
            float diff = input_ptr[outer * inner_size + inner] - mean;
            var += diff * diff;
        }
        var /= (float)inner_size;
        float rstd = (float)1 / ref_scalar_f32_sqrt(var + 1e-05f);
        for (ssize_t inner = 0; inner < inner_size; ++inner) {
            float val = (input_ptr[outer * inner_size + inner] - mean) * rstd;
            val *= weight_ptr[inner];
            val += bias_ptr[inner];
            out_ptr[outer * inner_size + inner] = val;
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][4], const float input_1[4], const float input_2[4], float out[2][4]) {
    node1_native_layer_norm_f32(input_0, input_1, input_2, out);
}
