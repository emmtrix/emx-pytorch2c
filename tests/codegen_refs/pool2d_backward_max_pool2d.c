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
* op: _adaptive_avg_pool2d_backward (kind: pool2d_backward)
* inputs: [shape=(1, 1, 2, 2), size=4, shape=(1, 1, 4, 4), size=16]
* output: shape=(1, 1, 4, 4), size=16
* params: {'kernel_size': (2, 2), 'stride': (2, 2), 'padding': (0, 0), 'dilation': (1, 1), 'ceil_mode': False, 'count_include_pad': True, 'divisor_override': None}
*/
void node1__adaptive_avg_pool2d_backward_f32(const float grad_output[1][1][2][2], const float input[1][1][4][4], float out[1][1][4][4]) {
    for (ssize_t n = 0; n < 1; ++n) {
        for (ssize_t c = 0; c < 1; ++c) {
            for (ssize_t oh = 0; oh < 2; ++oh) {
                for (ssize_t ow = 0; ow < 2; ++ow) {
                    float grad = grad_output[n][c][oh][ow] / (float)(2 * 2);
                    ssize_t in_h_base = (ssize_t)oh * 2;
                    ssize_t in_w_base = (ssize_t)ow * 2;
                    for (ssize_t kh = 0; kh < 2; ++kh) {
                        ssize_t in_h_idx = in_h_base + (ssize_t)kh;
                        for (ssize_t kw = 0; kw < 2; ++kw) {
                            ssize_t in_w_idx = in_w_base + (ssize_t)kw;
                            out[n][c][in_h_idx][in_w_idx] = grad;
                        }
                    }
                }
            }
        }
    }
}

void ref_codegen_main_f32(const float input_0[1][1][4][4], const float input_1[1][1][2][2], float out[1][1][4][4]) {
    node1__adaptive_avg_pool2d_backward_f32(input_1, input_0, out);
}
