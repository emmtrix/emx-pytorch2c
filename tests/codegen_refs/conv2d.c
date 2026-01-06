#include <stdint.h>
#include <stdbool.h>
#include "ops_scalar_f32.h"

void node1_conv2d_f32(const float input[1][2][5][5], const float weight[3][2][3][3], const float bias[3], float out[1][3][3][3]) {
    int64_t in_per_group = 2 / 1;
    int64_t out_per_group = 3 / 1;
    for (int64_t n = 0; n < 1; ++n) {
        for (int64_t oc = 0; oc < 3; ++oc) {
            int64_t group = oc / out_per_group;
            for (int64_t oh = 0; oh < 3; ++oh) {
                for (int64_t ow = 0; ow < 3; ++ow) {
                    float acc = 0.0f;
                    int64_t in_h_base = oh * 1 - 0;
                    int64_t in_w_base = ow * 1 - 0;
                    for (int64_t ic = 0; ic < in_per_group; ++ic) {
                        int64_t in_c = group * in_per_group + ic;
                        for (int64_t kh = 0; kh < 3; ++kh) {
                            int64_t in_h_idx = in_h_base + kh * 1;
                            if (in_h_idx < 0 || in_h_idx >= 5) {
                                continue;
                            }
                            for (int64_t kw = 0; kw < 3; ++kw) {
                                int64_t in_w_idx = in_w_base + kw * 1;
                                if (in_w_idx < 0 || in_w_idx >= 5) {
                                    continue;
                                }
                                acc += input[n][in_c][in_h_idx][in_w_idx] *
                                    weight[oc][ic][kh][kw];
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

void ref_codegen_main_f32(const float input_0[1][2][5][5], const float input_1[3][2][3][3], const float input_2[3], float out[1][3][3][3]) {
    node1_conv2d_f32(input_0, input_1, input_2, out);
}
