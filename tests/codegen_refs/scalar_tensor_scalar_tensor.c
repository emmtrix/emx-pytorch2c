#include <stdint.h>
#include <sys/types.h>
#include <stdbool.h>


/*
* op: scalar_tensor (kind: scalar_tensor)
* inputs: []
* output: shape=(), size=1
* params: {'value': 3.0}
*/
void node1_scalar_tensor_f32(float out[1]) {
    out[0] = 3.0f;
}

void ref_codegen_main_f32(float out[1]) {
    node1_scalar_tensor_f32(out);
}
