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
* op: cat (kind: concat)
* inputs: [shape=(2, 2), size=4, shape=(2, 1), size=2]
* output: shape=(2, 3), size=6
* params: {'dim': 1}
*/
void node1_cat_f32(const float a0[2][2], const float a1[2][1], float out[2][3]) {
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t i1 = 0; i1 < 2; ++i1) {
            out[i0][i1 + 0] = a0[i0][i1];
        }
    }
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t i1 = 0; i1 < 1; ++i1) {
            out[i0][i1 + 2] = a1[i0][i1];
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][2], const float input_1[2][1], float out[2][3]) {
    node1_cat_f32(input_0, input_1, out);
}
