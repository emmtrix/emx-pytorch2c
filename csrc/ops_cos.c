#include "ops_unary.h"

#include <math.h>

static float cos_op(float a) {
    return cosf(a);
}

int ref_run_cos(const RefOpCall *call, char *err_msg, size_t err_cap) {
    return ref_run_unary_f32(call, err_msg, err_cap, cos_op, "cos");
}
