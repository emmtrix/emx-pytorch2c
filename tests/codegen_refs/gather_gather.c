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
* op: gather (kind: gather)
* inputs: [shape=(2, 3), size=6, shape=(2, 2), size=4]
* output: shape=(2, 2), size=4
* params: {'dim': 1}
*/
void node1_gather_f32(const float input[2][3], const int64_t index[2][2], float out[2][2]) {for (ssize_t i0 = 0; i0 < 2; ++i0) {for (ssize_t i1 = 0; i1 < 2; ++i1) {/* gather dim: 1 */
            ssize_t idx = (ssize_t)(index[i0][i1]);
            out[i0][i1] = input[i0][idx];}}}

void ref_codegen_main_f32(const float input_0[2][3], const int64_t input_1[2][2], float out[2][2]) {
    node1_gather_f32(input_0, input_1, out);
}
