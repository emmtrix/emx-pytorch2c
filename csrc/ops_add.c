#include "ops_binary.h"

static float add_op(float a, float b) {
    return a + b;
}

int ref_run_add(const RefOpCall *call, char *err_msg, size_t err_cap) {
    return ref_run_binary_f32(call, err_msg, err_cap, add_op, "add");
}
