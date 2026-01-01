#include "ops_unary.h"

#include <math.h>

static float ceil_op(float a) {
    return ceilf(a);
}

int ref_run_ceil(const RefOpCall *call, char *err_msg, size_t err_cap) {
    return ref_run_unary_f32(call, err_msg, err_cap, ceil_op, "ceil");
}
