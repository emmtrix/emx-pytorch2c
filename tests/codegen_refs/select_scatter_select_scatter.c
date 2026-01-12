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
* op: select_scatter (kind: select_scatter)
* inputs: [shape=(2, 3), size=6, shape=(3,), size=3]
* output: shape=(2, 3), size=6
* params: {'dim': 0, 'index': 1}
*/
void node1_select_scatter_f32(const float input[2][3], const float src[3], float out[2][3]) {
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t i1 = 0; i1 < 3; ++i1) {
            if (i0 == 1) {
                out[i0][i1] = src[i1];
            } else {
                out[i0][i1] = input[i0][i1];
            }
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][3], const float input_1[3], float out[2][3]) {
    node1_select_scatter_f32(input_0, input_1, out);
}
