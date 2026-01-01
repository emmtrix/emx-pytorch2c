#include "ops_binary.h"

#include <math.h>

static float maximum_op(float a, float b) {
    return fmaxf(a, b);
}

int ref_run_maximum(const RefOpCall *call, char *err_msg, size_t err_cap) {
    return ref_run_binary_f32(call, err_msg, err_cap, maximum_op, "maximum");
}
