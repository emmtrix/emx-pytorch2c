#include "ops_unary.h"

static float relu_op(float a) {
    return a > 0.0f ? a : 0.0f;
}

int ref_run_relu(const RefOpCall *call, char *err_msg, size_t err_cap) {
    return ref_run_unary_f32(call, err_msg, err_cap, relu_op, "relu");
}
