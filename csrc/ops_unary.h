#ifndef REF_BACKEND_OPS_UNARY_H
#define REF_BACKEND_OPS_UNARY_H

#include "ref_backend.h"

typedef float (*RefUnaryF32Op)(float);

int ref_run_unary_f32(
    const RefOpCall *call,
    char *err_msg,
    size_t err_cap,
    RefUnaryF32Op op,
    const char *op_name
);

#endif
