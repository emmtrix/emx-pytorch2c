#include "ops_unary.h"

static float neg_op(float a) {
    return -a;
}

int ref_run_neg(const RefOpCall *call, char *err_msg, size_t err_cap) {
    return ref_run_unary_f32(call, err_msg, err_cap, neg_op, "neg");
}
