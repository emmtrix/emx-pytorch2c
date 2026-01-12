#include <stdint.h>
#include <sys/types.h>
#include <stdbool.h>


/*
* op: addmv (kind: addmv)
* inputs: [shape=(2,), size=2, shape=(2, 3), size=6, shape=(3,), size=3]
* output: shape=(2,), size=2
* params: {'alpha': 1.0, 'beta': 1.0}
*/
void node1_addmv_f32(const float input[2], const float mat[2][3], const float vec[3], float out[2]) {
    for (ssize_t i = 0; i < 2; ++i) {
        float acc = 0.0f;
        for (ssize_t t = 0; t < 3; ++t) {
            acc += mat[i][t] * vec[t];
        }
        out[i] = (1.0f) * input[i] + (1.0f) * acc;
    }
}

void ref_codegen_main_f32(const float input_0[2], const float input_1[2][3], const float input_2[3], float out[2]) {
    node1_addmv_f32(input_0, input_1, input_2, out);
}
