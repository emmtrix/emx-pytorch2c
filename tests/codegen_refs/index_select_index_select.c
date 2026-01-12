#include <stdint.h>
#include <sys/types.h>
#include <stdbool.h>


/*
* op: index_select (kind: index_select)
* inputs: [shape=(4, 3), size=12, shape=(2,), size=2]
* output: shape=(2, 3), size=6
* params: {'dim': 0}
*/
void node1_index_select_f32(const float input[4][3], const int64_t index[2], float out[2][3]) {
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t i1 = 0; i1 < 3; ++i1) {
            ssize_t idx = (ssize_t)(index[i0]);
            out[i0][i1] = input[idx][i1];
        }
    }
}

void ref_codegen_main_f32(const float input_0[4][3], const int64_t input_1[2], float out[2][3]) {
    node1_index_select_f32(input_0, input_1, out);
}
