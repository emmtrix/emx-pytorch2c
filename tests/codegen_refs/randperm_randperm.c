#include <stdint.h>
#include <sys/types.h>
#include <stdbool.h>


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
