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
* op: cumsum (kind: cumsum)
* inputs: [shape=(2, 3), size=6]
* output: shape=(2, 3), size=6
* params: {'dim': 1}
*/
void node1_cumsum_f32(const float input[2][3], float out[2][3]) {
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t i1 = 0; i1 < 3; ++i1) {
            float acc = 0.0f;
            for (ssize_t r1 = 0; r1 <= i1; ++r1) {
                acc += (float)input[i0][r1];
            }
            out[i0][i1] = acc;
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][3], float out[2][3]) {
    node1_cumsum_f32(input_0, out);
}
