#ifndef REF_BACKEND_H
#define REF_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum RefDType {
    REF_F32 = 0
} RefDType;

typedef struct RefTensorView {
    void *data;
    int32_t ndim;
    int64_t *sizes;
    int64_t *strides;
    int32_t dtype;
} RefTensorView;

typedef struct RefOpCall {
    RefTensorView *inputs;
    int32_t n_inputs;
    RefTensorView *outputs;
    int32_t n_outputs;
    void *params;
} RefOpCall;

#define REF_MAX_DIMS 8

typedef enum RefOpKind {
    REF_OP_ADD = 0,
    REF_OP_SUB = 1,
    REF_OP_MUL = 2,
    REF_OP_MATMUL = 3,
    REF_OP_BMM = 4,
    REF_OP_BROADCAST_IN_DIM = 5,
    REF_OP_DIV = 6,
    REF_OP_MAXIMUM = 7,
    REF_OP_MINIMUM = 8,
    REF_OP_NEG = 9,
    REF_OP_EXP = 10,
    REF_OP_ABS = 11,
    REF_OP_SQRT = 12,
    REF_OP_LOG = 13,
    REF_OP_SIN = 14,
    REF_OP_COS = 15,
    REF_OP_TANH = 16,
    REF_OP_FLOOR = 17,
    REF_OP_CEIL = 18,
    REF_OP_RECIPROCAL = 19,
    REF_OP_RELU = 20
} RefOpKind;

typedef struct RefBroadcastInDimParams {
    int32_t n_dims;
    int32_t *broadcast_dimensions;
} RefBroadcastInDimParams;

int ref_run_op(int32_t op_kind, const RefOpCall *call, char *err_msg, size_t err_cap);

#ifdef __cplusplus
}
#endif

#endif
