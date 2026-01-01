#include "ops_unary.h"

#include <math.h>

static float exp_op(float a) {
    return expf(a);
}

int ref_run_exp(const RefOpCall *call, char *err_msg, size_t err_cap) {
    return ref_run_unary_f32(call, err_msg, err_cap, exp_op, "exp");
}
