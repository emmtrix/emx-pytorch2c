#include <stdint.h>
#include <sys/types.h>
#include <stdbool.h>


/*
* op: split_with_sizes (kind: split_with_sizes)
* inputs: [shape=(2, 3), size=6]
* output: shape=(2, 1), size=2
* params: {'dim': 1, 'offset': 0, 'split_size': 1}
*/
void node1_split_with_sizes_f32(const float input[2][3], float out[2][1]) {
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t i1 = 0; i1 < 1; ++i1) {
            out[i0][i1] = input[i0][i1 + 0];
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][3], float out[2][1]) {
    node1_split_with_sizes_f32(input_0, out);
}
