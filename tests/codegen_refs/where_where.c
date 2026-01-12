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
* op: where (kind: where)
* inputs: [shape=(1, 3), size=3, shape=(1, 3), size=3, shape=(1, 3), size=3]
* output: shape=(1, 3), size=3
* params: {}
*/
void node1_where_f32(const uint8_t cond[1][3], const float a[1][3], const float b[1][3], float out[1][3]) {
    for (ssize_t i0 = 0; i0 < 1; ++i0) {
        for (ssize_t i1 = 0; i1 < 3; ++i1) {
            out[i0][i1] = (cond[0][i1] != 0) ? a[0][i1] : b[0][i1];
        }
    }
}

void ref_codegen_main_f32(const uint8_t input_0[1][3], const float input_1[1][3], const float input_2[1][3], float out[1][3]) {
    node1_where_f32(input_0, input_1, input_2, out);
}
