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
* op: linear (kind: linear)
* inputs: [shape=(2, 3), size=6, shape=(4, 3), size=12, shape=(4,), size=4]
* output: shape=(2, 4), size=8
* params: {'has_bias': True}
*/
void node1_linear_f32(const float input[2][3], const float weight[4][3], const float bias[4], float out[2][4]) {
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t j = 0; j < 4; ++j) {
            float acc = 0.0f;
            for (ssize_t t = 0; t < 3; ++t) {
                acc += input[i0][t] * weight[j][t];
            }
            out[i0][j] = acc + bias[j];
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][3], const float input_1[4][3], const float input_2[4], float out[2][4]) {
    node1_linear_f32(input_0, input_1, input_2, out);
}
