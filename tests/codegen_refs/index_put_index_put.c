#include <stdint.h>
#include <sys/types.h>
#include <stdbool.h>


/*
* op: index_put (kind: index_put)
* inputs: [shape=(3,), size=3, shape=(2,), size=2, shape=(2,), size=2]
* output: shape=(3,), size=3
* params: {'index_rank': 1, 'accumulate': False}
*/
void node1_index_put_f32(const float input[3], const int64_t index0[2], const float values[2], float out[3]) {
    for (ssize_t i0 = 0; i0 < 3; ++i0) {
        out[i0] = input[i0];
    }
}
for (ssize_t i0 = 0; i0 < 2; ++i0) {
    ssize_t idx0 = (ssize_t)(index0[i0]);
    if (idx0 < 0) { idx0 += 3; }
    out[idx0] = values[i0];
}
}

void ref_codegen_main_f32(const float input_0[3], const int64_t input_1[2], const float input_2[2], float out[3]) {
    node1_index_put_f32(input_0, input_1, input_2, out);
}
