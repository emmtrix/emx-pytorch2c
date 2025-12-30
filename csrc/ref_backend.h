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

typedef enum RefOpKind {
    REF_OP_ADD = 0,
    REF_OP_MATMUL = 1
} RefOpKind;

int ref_run_op(int32_t op_kind, const RefOpCall *call, char *err_msg, size_t err_cap);

#ifdef __cplusplus
}
#endif

#endif
