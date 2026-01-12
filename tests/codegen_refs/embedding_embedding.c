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
* op: embedding (kind: embedding)
* inputs: [shape=(4, 3), size=12, shape=(2, 2), size=4]
* output: shape=(2, 2, 3), size=12
* params: {'padding_idx': -1}
*/
void node1_embedding_f32(const float weight[4][3], const int64_t indices[2][2], float out[2][2][3]) {
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t i1 = 0; i1 < 2; ++i1) {
            for (ssize_t i2 = 0; i2 < 3; ++i2) {
                ssize_t idx = (ssize_t)(indices[i0][i1]);
                out[i0][i1][i2] = weight[idx][i2];
            }
        }
    }
}

void ref_codegen_main_f32(const float input_0[4][3], const int64_t input_1[2][2], float out[2][2][3]) {
    node1_embedding_f32(input_0, input_1, out);
}
