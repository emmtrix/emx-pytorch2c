#include "ops_binary.h"

static float sub_op(float a, float b) {
    return a - b;
}

int ref_run_sub(const RefOpCall *call, char *err_msg, size_t err_cap) {
    return ref_run_binary_f32(call, err_msg, err_cap, sub_op, "sub");
}
