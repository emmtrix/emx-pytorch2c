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
* op: full_like (kind: fill)
* inputs: [shape=(2, 3), size=6]
* output: shape=(2, 3), size=6
* params: {'value': 3.0}
*/
void node1_full_like_f32(const float a[2][3], float out[2][3]) {
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t i1 = 0; i1 < 3; ++i1) {
            out[i0][i1] = 3.0f;
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][3], float out[2][3]) {
    node1_full_like_f32(input_0, out);
}
