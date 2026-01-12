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
* op: sum (kind: reduction)
* inputs: [shape=(2, 3), size=6]
* output: shape=(), size=1
* params: {'reduce_all': True}
*/
void node1_sum_f32(const float a[2][3], float out[1]) {
    float acc = 0.0f;
    for (ssize_t r0 = 0; r0 < 2; ++r0) {
        for (ssize_t r1 = 0; r1 < 3; ++r1) {
            acc += a[r0][r1];
        }
    }
    out[0] = acc;
}

void ref_codegen_main_f32(const float input_0[2][3], float out[1]) {
    node1_sum_f32(input_0, out);
}
