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
* op: sort (kind: sort)
* inputs: [shape=(2, 3), size=6]
* output: shape=(2, 3), size=6
* params: {'dim': 1, 'descending': False, 'stable': False}
*/
void node1_sort_f32(const float input[2][3], float out[2][3]) {
    for (ssize_t i0 = 0; i0 < 2; ++i0) {
        for (ssize_t k = 0; k < 3; ++k) {
            out[i0][k] = input[i0][k];
        }
        for (ssize_t pass = 0; pass < 3; ++pass) {
            ssize_t start = pass & 1;
            for (ssize_t j = start; j + 1 < 3; j += 2) {
                float a = out[i0][j];
                float b = out[i0][j + 1];
                if (a > b) {
                    out[i0][j] = b;
                    out[i0][j + 1] = a;
                }
            }
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][3], float out[2][3]) {
    node1_sort_f32(input_0, out);
}
