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
* op: native_dropout (kind: dropout)
* inputs: [shape=(2, 2), size=4]
* output: shape=(2, 2), size=4
* params: {'p': 0.0, 'train': False}
*/
void node1_native_dropout_f32(const float a[2][2], float out[2][2]) {
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t i1 = 0; i1 < 2; ++i1) {
            out[i0][i1] = a[i0][i1];
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][2], float out[2][2]) {
    node1_native_dropout_f32(input_0, out);
}
