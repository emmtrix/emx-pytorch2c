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
* op: conv1d (kind: conv1d)
* inputs: [shape=(1, 2, 5), size=10, shape=(3, 2, 3), size=18, shape=(3,), size=3]
* output: shape=(1, 3, 3), size=9
* params: {'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1, 'has_bias': True}
*/
void node1_conv1d_f32(const float input[1][2][5], const float weight[3][2][3], const float bias[3], float out[1][3][3]) {
    ssize_t in_per_group = 2 / 1;
    ssize_t out_per_group = 3 / 1;
    for (ssize_t n = 0; n < 1; ++n) {
        for (ssize_t oc = 0; oc < 3; ++oc) {
            ssize_t group = (ssize_t)oc / out_per_group;
            for (ssize_t ol = 0; ol < 3; ++ol) {
                float acc = 0.0f;
                ssize_t in_l_base = (ssize_t)ol * 1 - 0;
                for (ssize_t ic = 0; ic < in_per_group; ++ic) {
                    ssize_t in_c = group * in_per_group + (ssize_t)ic;
                    for (ssize_t kl = 0; kl < 3; ++kl) {
                        ssize_t in_l_idx = in_l_base + (ssize_t)kl * 1;
                        if (in_l_idx < 0 || in_l_idx >= 5) {
                            continue;
                        }
                        acc += input[n][in_c][in_l_idx] * weight[oc][ic][kl];
                    }
                }
                acc += bias[oc];
                out[n][oc][ol] = acc;
            }
        }
    }
}

void ref_codegen_main_f32(const float input_0[1][2][5], const float input_1[3][2][3], const float input_2[3], float out[1][3][3]) {
    node1_conv1d_f32(input_0, input_1, input_2, out);
}
