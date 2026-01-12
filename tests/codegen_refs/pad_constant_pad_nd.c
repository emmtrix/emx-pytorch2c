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
* op: constant_pad_nd (kind: pad)
* inputs: [shape=(2, 2), size=4]
* output: shape=(4, 4), size=16
* params: {'pad_before': (1, 1), 'pad_after': (1, 1), 'value': 0.0, 'mode': 'constant'}
*/
void node1_constant_pad_nd_f32(const float a[2][2], float out[4][4]) {
    for (ssize_t i0 = 0; i0 < 4; ++i0) {
        for (ssize_t i1 = 0; i1 < 4; ++i1) {
            bool in_bounds = true;
            ssize_t in_0 = (ssize_t)i0 - 1;
            if (in_0 < 0 || in_0 >= 2) {
                in_bounds = false;
            }
            ssize_t in_1 = (ssize_t)i1 - 1;
            if (in_1 < 0 || in_1 >= 2) {
                in_bounds = false;
            }
            if (in_bounds) {
                out[i0][i1] = a[in_0][in_1];
            } else {
                out[i0][i1] = 0.0f;
            }
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][2], float out[4][4]) {
    node1_constant_pad_nd_f32(input_0, out);
}
