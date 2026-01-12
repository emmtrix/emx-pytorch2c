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
* op: reshape (kind: view)
* inputs: [shape=(2, 3, 4), size=24]
* output: shape=(2, 12), size=24
* params: {'size': (2, 12), 'view_strides': (12, 1), 'storage_offset': 0}
*/
void node1_reshape_f32(const float a[2][3][4], float out[2][12]) {
    const float* a_ptr = (const float*)a;
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t i1 = 0; i1 < 12; ++i1) {
            ssize_t offset = (ssize_t)i0 * (ssize_t)12 + (ssize_t)i1 * (ssize_t)1;
            out[i0][i1] = a_ptr[offset];
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][3][4], float out[2][12]) {
    node1_reshape_f32(input_0, out);
}
