#include <stdint.h>
#include <sys/types.h>
#include <stdbool.h>


/*
* op: matmul (kind: matmul)
* inputs: [shape=(2, 3), size=6, shape=(3, 4), size=12]
* output: shape=(2, 4), size=8
* params: {}
*/
void node1_matmul_f32(const float a[2][3], const float b[3][4], float out[2][4]) {
    for (ssize_t i = 0; i < 2; ++i) {
        for (ssize_t j = 0; j < 4; ++j) {
            float acc = 0.0f;
            for (ssize_t t = 0; t < 3; ++t) {
                acc += a[i][t] * b[t][j];
            }
            out[i][j] = acc;
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][3], const float input_1[3][4], float out[2][4]) {
    node1_matmul_f32(input_0, input_1, out);
}
