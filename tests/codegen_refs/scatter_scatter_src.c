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
* op: scatter_src (kind: scatter)
* inputs: [shape=(2, 3), size=6, shape=(2, 2), size=4, shape=(2, 2), size=4]
* output: shape=(2, 3), size=6
* params: {'dim': 1}
*/
void node1_scatter_src_f32(const float input[2][3], const int64_t index[2][2], const float src[2][2], float out[2][3]) {
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t i1 = 0; i1 < 3; ++i1) {
            out[i0][i1] = input[i0][i1];
        }
    }
}
for (ssize_t i0 = 0; i0 < 2; ++i0) {
    for (ssize_t i1 = 0; i1 < 2; ++i1) {
        ssize_t idx = (ssize_t)(index[i0][i1]);
        if (idx < 0) { idx += 3; }
        out[i0][idx] = src[i0][i1];
    }
}
}

void ref_codegen_main_f32(const float input_0[2][3], const int64_t input_1[2][2], const float input_2[2][2], float out[2][3]) {
    node1_scatter_src_f32(input_0, input_1, input_2, out);
}
