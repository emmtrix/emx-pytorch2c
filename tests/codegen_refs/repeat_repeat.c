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
* op: repeat (kind: repeat)
* inputs: [shape=(2, 3), size=6]
* output: shape=(2, 6), size=12
* params: {'repeats': (1, 2)}
*/
void node1_repeat_f32(const float input[2][3], float out[2][6]) {
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t i1 = 0; i1 < 6; ++i1) {
            out[i0][i1] = input[(i0 % 2)][(i1 % 3)];
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][3], float out[2][6]) {
    node1_repeat_f32(input_0, out);
}
