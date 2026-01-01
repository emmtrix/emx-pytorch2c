#include "ops_unary.h"

#include <math.h>

static float sin_op(float a) {
    return sinf(a);
}

int ref_run_sin(const RefOpCall *call, char *err_msg, size_t err_cap) {
    return ref_run_unary_f32(call, err_msg, err_cap, sin_op, "sin");
}
