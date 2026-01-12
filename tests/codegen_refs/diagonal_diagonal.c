#include <stdint.h>
#include <sys/types.h>
#include <stdbool.h>


/*
* op: diagonal (kind: diagonal)
* inputs: [shape=(3, 3), size=9]
* output: shape=(3,), size=3
* params: {'offset': 0, 'dim1': 0, 'dim2': 1}
*/
void node1_diagonal_f32(const float a[3][3], float out[3]) {
    for (ssize_t i0 = 0; i0 < 3; ++i0) {
        out[i0] = ((float*)a)[i0 * 3 + (i0 + 0) * 1];
    }
}

void ref_codegen_main_f32(const float input_0[3][3], float out[3]) {
    node1_diagonal_f32(input_0, out);
}
