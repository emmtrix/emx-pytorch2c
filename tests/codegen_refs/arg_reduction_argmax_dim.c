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
* op: argmax (kind: arg_reduction)
* inputs: [shape=(2, 3), size=6]
* output: shape=(2,), size=2
* params: {'reduce_all': False}
*/
void node1_argmax_f32(const float a[2][3], int64_t out[2]) {
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        float best_value = a[i0][0];
        ssize_t best_index = 0;
        for (ssize_t r1 = 1; r1 < 3; ++r1) {
            float value = a[i0][r1];
            if (value > best_value) {
                best_value = value;
                best_index = r1;
            }
        }
        out[i0] = best_index;
    }
}

void ref_codegen_main_f32(const float input_0[2][3], int64_t out[2]) {
    node1_argmax_f32(input_0, out);
}
