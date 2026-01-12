#include <stdint.h>
#include <sys/types.h>
#include <stdbool.h>


/*
* op: resize_ (kind: resize)
* inputs: [shape=(6,), size=6]
* output: shape=(2, 3), size=6
* params: {'size': (2, 3)}
*/
void node1_resize__f32(const float a[6], float out[2][3]) {
    const float* a_ptr = (const float*)a;
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t i1 = 0; i1 < 3; ++i1) {
            ssize_t linear = i0 * 3 + i1 * 1;
            ssize_t remaining = linear;
            ssize_t idx0 = remaining % 6;
            ssize_t offset = idx0 * 1;
            out[i0][i1] = a_ptr[offset];
        }
    }
}

void ref_codegen_main_f32(const float input_0[6], float out[2][3]) {
    node1_resize__f32(input_0, out);
}
