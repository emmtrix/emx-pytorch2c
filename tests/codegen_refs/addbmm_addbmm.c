#include <stdint.h>
#include <sys/types.h>
#include <stdbool.h>


/*
* op: addbmm (kind: addbmm)
* inputs: [shape=(2, 4), size=8, shape=(5, 2, 3), size=30, shape=(5, 3, 4), size=60]
* output: shape=(2, 4), size=8
* params: {'alpha': 1.0, 'beta': 1.0}
*/
void node1_addbmm_f32(const float input[2][4], const float batch1[5][2][3], const float batch2[5][3][4], float out[2][4]) {
    for (ssize_t i = 0; i < 2; ++i) {
        for (ssize_t j = 0; j < 4; ++j) {
            float acc = 0.0f;
            for (ssize_t b_idx = 0; b_idx < 5; ++b_idx) {
                for (ssize_t t = 0; t < 3; ++t) {
                    acc += batch1[b_idx][i][t] * batch2[b_idx][t][j];
                }
            }
            out[i][j] = (1.0f) * ((float*)input)[i * 4 + j * 1] + (1.0f) * acc;
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][4], const float input_1[5][2][3], const float input_2[5][3][4], float out[2][4]) {
    node1_addbmm_f32(input_0, input_1, input_2, out);
}
