#include "ops_binary.h"

#include <math.h>

static float minimum_op(float a, float b) {
    return fminf(a, b);
}

int ref_run_minimum(const RefOpCall *call, char *err_msg, size_t err_cap) {
    return ref_run_binary_f32(call, err_msg, err_cap, minimum_op, "minimum");
}
