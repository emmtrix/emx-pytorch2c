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
* op: _embedding_bag (kind: embedding_bag)
* inputs: [shape=(4, 3), size=12, shape=(4,), size=4, shape=(2,), size=2]
* output: shape=(2, 3), size=6
* params: {'mode': 0, 'padding_idx': -1, 'include_last_offset': False}
*/
void node1__embedding_bag_f32(const float weight[4][3], const int64_t indices[4], const int64_t offsets[2], float out[2][3]) {
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t i1 = 0; i1 < 3; ++i1) {
            ssize_t start = (ssize_t)(offsets[i0]);
            ssize_t end = (i0 + 1 < 2) ? (ssize_t)(offsets[i0 + 1]) : 4;
            float acc = 0.0f;
            ssize_t count = 0;
            for (ssize_t j = start; j < end; ++j) {
                ssize_t idx = (ssize_t)(indices[j]);
                acc += weight[idx][i1];
                count += 1;
            }
            out[i0][i1] = acc;
        }
    }
}

void ref_codegen_main_f32(const float input_0[4][3], const int64_t input_1[4], const int64_t input_2[2], float out[2][3]) {
    node1__embedding_bag_f32(input_0, input_1, input_2, out);
}
