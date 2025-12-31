#ifndef REF_BACKEND_OPS_BINARY_H
#define REF_BACKEND_OPS_BINARY_H

#include "ref_backend.h"

typedef float (*RefBinaryF32Op)(float, float);

int ref_run_binary_f32(
    const RefOpCall *call,
    char *err_msg,
    size_t err_cap,
    RefBinaryF32Op op,
    const char *op_name
);

#endif
