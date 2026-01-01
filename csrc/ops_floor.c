#include "ops_unary.h"

#include <math.h>

static float floor_op(float a) {
    return floorf(a);
}

int ref_run_floor(const RefOpCall *call, char *err_msg, size_t err_cap) {
    return ref_run_unary_f32(call, err_msg, err_cap, floor_op, "floor");
}
