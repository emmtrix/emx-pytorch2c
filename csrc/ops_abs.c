#include "ops_unary.h"

#include <math.h>

static float abs_op(float a) {
    return fabsf(a);
}

int ref_run_abs(const RefOpCall *call, char *err_msg, size_t err_cap) {
    return ref_run_unary_f32(call, err_msg, err_cap, abs_op, "abs");
}
