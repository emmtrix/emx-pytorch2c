#ifndef C_REF_BACKEND_H
#define C_REF_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define REF_BACKEND_API __declspec(dllexport)
#elif defined(__GNUC__)
#define REF_BACKEND_API __attribute__((visibility("default")))
#else
#define REF_BACKEND_API
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
    REF_OP_RELU = 20,
    REF_OP_ACOS = 21,
    REF_OP_ACOSH = 22,
    REF_OP_ASIN = 23,
    REF_OP_ASINH = 24,
    REF_OP_ATAN = 25,
    REF_OP_ATANH = 26,
    REF_OP_COSH = 27,
    REF_OP_SINH = 28,
    REF_OP_TAN = 29,
    REF_OP_ERF = 30,
    REF_OP_ERFC = 31,
    REF_OP_EXPM1 = 32,
    REF_OP_LOG1P = 33,
    REF_OP_LOG2 = 34,
    REF_OP_LOG10 = 35,
    REF_OP_RSQRT = 36,
    REF_OP_SIGMOID = 37,
    REF_OP_SIGN = 38,
    REF_OP_ROUND = 39,
    REF_OP_TRUNC = 40,
    REF_OP_CONV2D = 41,
    REF_OP_ANGLE = 42,
    REF_OP_CONJ = 43,
    REF_OP_CONJ_PHYSICAL = 44,
    REF_OP_DEG2RAD = 45,
    REF_OP_DIGAMMA = 46,
    REF_OP_ERFINV = 47,
    REF_OP_EXP2 = 48,
    REF_OP_FRAC = 49,
    REF_OP_I0 = 50,
    REF_OP_LGAMMA = 51,
    REF_OP_LOGIT = 52,
    REF_OP_NAN_TO_NUM = 53,
    REF_OP_POSITIVE = 54,
    REF_OP_RAD2DEG = 55,
    REF_OP_REAL = 56,
    REF_OP_SGN = 57,
    REF_OP_SINC = 58,
    REF_OP_SQUARE = 59
} RefOpKind;

typedef struct RefBroadcastInDimParams {
    int32_t n_dims;
    int32_t *broadcast_dimensions;
} RefBroadcastInDimParams;

REF_BACKEND_API int ref_run_op(
    int32_t op_kind,
    const RefOpCall *call,
    char *err_msg,
    size_t err_cap);

#ifdef __cplusplus
}
#endif

#endif
