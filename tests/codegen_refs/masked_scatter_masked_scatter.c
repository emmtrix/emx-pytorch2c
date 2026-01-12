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
* op: masked_scatter (kind: masked_scatter)
* inputs: [shape=(2, 2), size=4, shape=(2, 2), size=4, shape=(2,), size=2]
* output: shape=(2, 2), size=4
* params: {}
*/
void node1_masked_scatter_f32(const float input[2][2], const uint8_t mask[2][2], const float source[2], float out[2][2]) {
    ssize_t source_index = 0;
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t i1 = 0; i1 < 2; ++i1) {
            if (mask[i0][i1] != 0) {
                ssize_t src_linear = source_index;
                ssize_t src_i0 = src_linear % 2;
                src_linear /= 2;
                out[i0][i1] = source[src_i0];
                source_index++;
            } else {
                out[i0][i1] = input[i0][i1];
            }
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][2], const uint8_t input_1[2][2], const float input_2[2], float out[2][2]) {
    node1_masked_scatter_f32(input_0, input_1, input_2, out);
}
