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
* op: max_pool3d_with_indices (kind: pool3d)
* inputs: [shape=(1, 2, 4, 4, 4), size=128]
* output: shape=(1, 2, 3, 3, 3), size=54
* params: {'kernel_size': (2, 2, 2), 'stride': (1, 1, 1), 'padding': (0, 0, 0), 'dilation': (1, 1, 1), 'pool_kind': 'max_pool3d', 'ceil_mode': False, 'count_include_pad': False, 'divisor_override': None}
*/
void node1_max_pool3d_with_indices_f32(const float input[1][2][4][4][4], float out[1][2][3][3][3]) {
    ssize_t out_d = 3;
    ssize_t out_h = 3;
    ssize_t out_w = 3;
    for (ssize_t n = 0; n < 1; ++n) {
        for (ssize_t c = 0; c < 2; ++c) {
            for (ssize_t od = 0; od < out_d; ++od) {
                for (ssize_t oh = 0; oh < out_h; ++oh) {
                    for (ssize_t ow = 0; ow < out_w; ++ow) {
                        ssize_t in_d_base = (ssize_t)od * 1 - 0;
                        ssize_t in_h_base = (ssize_t)oh * 1 - 0;
                        ssize_t in_w_base = (ssize_t)ow * 1 - 0;
                        bool has_value = false;
                        float max_val = 0;
                        for (ssize_t kd = 0; kd < 2; ++kd) {
                            ssize_t in_d_idx = in_d_base + (ssize_t)kd * 1;
                            if (in_d_idx < 0 || in_d_idx >= 4) {
                                continue;
                            }
                            for (ssize_t kh = 0; kh < 2; ++kh) {
                                ssize_t in_h_idx = in_h_base + (ssize_t)kh * 1;
                                if (in_h_idx < 0 || in_h_idx >= 4) {
                                    continue;
                                }
                                for (ssize_t kw = 0; kw < 2; ++kw) {
                                    ssize_t in_w_idx = in_w_base + (ssize_t)kw * 1;
                                    if (in_w_idx < 0 || in_w_idx >= 4) {
                                        continue;
                                    }
                                    float val = input[n][c][in_d_idx][in_h_idx][in_w_idx];
                                    if (!has_value || val > max_val) {
                                        max_val = val;
                                        has_value = true;
                                    }
                                }
                            }
                        }
                        out[n][c][od][oh][ow] = max_val;
                    }
                }
            }
        }
    }
}

void ref_codegen_main_f32(const float input_0[1][2][4][4][4], float out[1][2][3][3][3]) {
    node1_max_pool3d_with_indices_f32(input_0, out);
}
