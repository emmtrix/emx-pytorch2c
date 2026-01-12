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
* op: addr (kind: addr)
* inputs: [shape=(2, 3), size=6, shape=(2,), size=2, shape=(3,), size=3]
* output: shape=(2, 3), size=6
* params: {'alpha': 1.0, 'beta': 1.0}
*/
void node1_addr_f32(const float input[2][3], const float vec1[2], const float vec2[3], float out[2][3]) {
    for (ssize_t i = 0; i < 2; ++i) {
        for (ssize_t j = 0; j < 3; ++j) {            out[i][j] = (1.0f) * input[i][j] + (1.0f) * vec1[i] * vec2[j];        }
    }
}

void ref_codegen_main_f32(const float input_0[2][3], const float input_1[2], const float input_2[3], float out[2][3]) {
    node1_addr_f32(input_0, input_1, input_2, out);
}
