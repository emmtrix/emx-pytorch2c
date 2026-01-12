#include <stdint.h>
#include <sys/types.h>
#include <stdbool.h>


/*
* op: arange (kind: arange)
* inputs: []
* output: shape=(6,), size=6
* params: {'start': 0, 'end': 6, 'step': 1}
*/
void node1_arange_i32(int32_t out[6]) {
    for (ssize_t i0 = 0; i0 < 6; ++i0) {
        out[i0] = 0 + (1 * i0);
    }
}

void ref_codegen_main_i32(int32_t out[6]) {
    node1_arange_i32(out);
}
