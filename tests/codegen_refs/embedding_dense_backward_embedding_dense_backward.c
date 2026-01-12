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
* op: embedding_dense_backward (kind: embedding_dense_backward)
* inputs: [shape=(2, 2, 3), size=12, shape=(2, 2), size=4]
* output: shape=(4, 3), size=12
* params: {'num_weights': 4, 'padding_idx': -1}
*/
void node1_embedding_dense_backward_f32(const float grad_output[2][2][3], const int64_t indices[2][2], float out[4][3]) {
    for (ssize_t i0 = 0; i0 < 4; ++i0) {
        for (ssize_t i1 = 0; i1 < 3; ++i1) {
            out[i0][i1] = 0.0f;
        }
    }
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t i1 = 0; i1 < 2; ++i1) {
            for (ssize_t i2 = 0; i2 < 3; ++i2) {
                ssize_t idx = (ssize_t)(indices[i0][i1]);
                out[idx][i2] += grad_output[i0][i1][i2];
            }
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][2][3], const int64_t input_1[2][2], float out[4][3]) {
    node1_embedding_dense_backward_f32(input_0, input_1, out);
}
