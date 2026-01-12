#include <stdint.h>
#include <sys/types.h>
#include <stdbool.h>


/*
* op: nonzero (kind: nonzero)
* inputs: [shape=(2, 2), size=4]
* output: shape=(4, 2), size=8
* params: {'output_shape': (4, 2)}
*/
void node1_nonzero_f32(const float input[2][2], int64_t out[4][2]) {
    ssize_t out_index = 0;
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t i1 = 0; i1 < 2; ++i1) {
            if (input[i0][i1] != 0) {
                out[out_index][0] = (int64_t)i0;
                out[out_index][1] = (int64_t)i1;
                out_index += 1;
            }
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][2], int64_t out[4][2]) {
    node1_nonzero_f32(input_0, out);
}
