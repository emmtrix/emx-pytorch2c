#include "ops_unary.h"

static float reciprocal_op(float a) {
    return 1.0f / a;
}

int ref_run_reciprocal(const RefOpCall *call, char *err_msg, size_t err_cap) {
    return ref_run_unary_f32(call, err_msg, err_cap, reciprocal_op, "reciprocal");
}
