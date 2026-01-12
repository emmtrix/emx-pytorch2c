#include <stdint.h>
#include <sys/types.h>
#include <stdbool.h>


/*
* op: rand (kind: random)
* inputs: []
* output: shape=(2, 3), size=6
* params: {'size': (2, 3)}
*/
void node1_rand_f32(float out[2][3]) {
    (void)out;
}

void ref_codegen_main_f32(float out[2][3]) {
    node1_rand_f32(out);
}
