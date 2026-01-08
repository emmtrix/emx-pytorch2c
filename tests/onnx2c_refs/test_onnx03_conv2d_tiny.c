#include <stdint.h>
#include <sys/types.h>
#include <stdbool.h>
#include "ops_scalar_f32.h"

static const float weight_Conv_out_weight[1][1][3][3] = {
    {
        {
            {
                0x0.0p+0f, 0x0.0p+0f, 0x0.0p+0f
            },
            {
                0x0.0p+0f, 0x1.000000p+0f, 0x0.0p+0f
            },
            {
                0x0.0p+0f, 0x0.0p+0f, 0x0.0p+0f
            }
        }
    }
};

static const float weight_Conv_out_bias[1] = {
    0x0.0p+0f
};

void node1_conv2d_f32(const float input[1][1][5][5], const float weight[1][1][3][3], const float bias[1], float out[1][1][3][3]) {
    ssize_t in_per_group = 1 / 1;
    ssize_t out_per_group = 1 / 1;
    ssize_t out_pad_h = 0;
    ssize_t out_pad_w = 0;
    (void)out_pad_h;
    (void)out_pad_w;
    for (ssize_t n = 0; n < 1; ++n) {
        for (ssize_t oc = 0; oc < 1; ++oc) {
            ssize_t group = (ssize_t)oc / out_per_group;
            for (ssize_t oh = 0; oh < 3; ++oh) {
                for (ssize_t ow = 0; ow < 3; ++ow) {
                    float acc = 0.0f;
                    ssize_t in_h_base = (ssize_t)oh * 1 - 0;
                    ssize_t in_w_base = (ssize_t)ow * 1 - 0;
                    for (ssize_t ic = 0; ic < in_per_group; ++ic) {
                        ssize_t in_c = group * in_per_group + (ssize_t)ic;
                        for (ssize_t kh = 0; kh < 3; ++kh) {
                            ssize_t in_h_idx = in_h_base + (ssize_t)kh * 1;
                            if (in_h_idx < 0 || in_h_idx >= 5) {
                                continue;
                            }
                            for (ssize_t kw = 0; kw < 3; ++kw) {
                                ssize_t in_w_idx = in_w_base + (ssize_t)kw * 1;
                                if (in_w_idx < 0 || in_w_idx >= 5) {
                                    continue;
                                }
                                acc += input[n][in_c][in_h_idx][in_w_idx] * weight[oc][ic][kh][kw];
                            }
                        }
                    }
                    acc += bias[oc];
                    out[n][oc][oh][ow] = acc;
                }
            }
        }
    }
}

void ref_codegen_main_f32(const float input_0[1][1][5][5], const float input_1[1][1][3][3], const float input_2[1], float out[1][1][3][3]) {
    node1_conv2d_f32(input_0, input_1, input_2, out);
}

void entry(const float in0[1][1][5][5], float out0[1][1][3][3]) {
    ref_codegen_main_f32(in0, weight_Conv_out_weight, weight_Conv_out_bias, out0);
}
