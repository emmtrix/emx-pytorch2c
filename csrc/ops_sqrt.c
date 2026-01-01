#include "ops_unary.h"

#include <math.h>

static float sqrt_op(float a) {
    return sqrtf(a);
}

int ref_run_sqrt(const RefOpCall *call, char *err_msg, size_t err_cap) {
    return ref_run_unary_f32(call, err_msg, err_cap, sqrt_op, "sqrt");
}
