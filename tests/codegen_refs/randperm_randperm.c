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
* op: randperm (kind: randperm)
* inputs: []
* output: shape=(5,), size=5
* params: {'size': (5,)}
*/
void node1_randperm_i64(int64_t out[5]) {
    (void)out;
}

void ref_codegen_main_i64(int64_t out[5]) {
    node1_randperm_i64(out);
}
