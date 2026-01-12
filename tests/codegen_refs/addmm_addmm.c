#include <stdint.h>
#include <sys/types.h>
#include <stdbool.h>


/*
* op: addmm (kind: addmm)
* inputs: [shape=(2, 2), size=4, shape=(2, 3), size=6, shape=(3, 2), size=6]
* output: shape=(2, 2), size=4
* params: {'alpha': 1.0, 'beta': 1.0}
*/
void node1_addmm_f32(const float input[2][2], const float mat1[2][3], const float mat2[3][2], float out[2][2]) {
    for (ssize_t i = 0; i < 2; ++i) {
        for (ssize_t j = 0; j < 2; ++j) {
            float acc = 0.0f;
            for (ssize_t t = 0; t < 3; ++t) {
                acc += mat1[i][t] * mat2[t][j];
            }
            out[i][j] = (1.0f) * ((float*)input)[i * 2 + j * 1] + (1.0f) * acc;
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][2], const float input_1[2][3], const float input_2[3][2], float out[2][2]) {
    node1_addmm_f32(input_0, input_1, input_2, out);
}
