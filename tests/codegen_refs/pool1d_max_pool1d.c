#include <stdint.h>
#include <sys/types.h>
#include <stdbool.h>


/*
* op: max_pool1d (kind: pool1d)
* inputs: [shape=(1, 2, 4), size=8]
* output: shape=(1, 2, 3), size=6
* params: {'kernel_size': 2, 'stride': 1, 'padding': 0, 'dilation': 1, 'ceil_mode': False, 'count_include_pad': False, 'divisor_override': None}
*/
void node1_max_pool1d_f32(const float input[1][2][4], float out[1][2][3]) {
    for (ssize_t n = 0; n < 1; ++n) {
        for (ssize_t c = 0; c < 2; ++c) {
            for (ssize_t ol = 0; ol < 3; ++ol) {
                ssize_t in_l_base = (ssize_t)ol * 1 - 0;
                bool has_value = false;
                float max_val = 0;
                for (ssize_t kl = 0; kl < 2; ++kl) {
                    ssize_t in_l_idx = in_l_base + (ssize_t)kl * 1;
                    if (in_l_idx < 0 || in_l_idx >= 4) {
                        continue;
                    }
                    float val = input[n][c][in_l_idx];
                    if (!has_value || val > max_val) {
                        max_val = val;
                        has_value = true;
                    }
                }
                out[n][c][ol] = max_val;
            }
        }
    }
}

void ref_codegen_main_f32(const float input_0[1][2][4], float out[1][2][3]) {
    node1_max_pool1d_f32(input_0, out);
}
