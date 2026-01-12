#include <stdint.h>
#include <sys/types.h>
#include <stdbool.h>


/*
* op: flip (kind: flip)
* inputs: [shape=(2, 3), size=6]
* output: shape=(2, 3), size=6
* params: {'dims': (1,)}
*/
void node1_flip_f32(const float a[2][3], float out[2][3]) {
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t i1 = 0; i1 < 3; ++i1) {
            out[i0][i1] = a[i0][(2 - i1)];
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][3], float out[2][3]) {
    node1_flip_f32(input_0, out);
}
