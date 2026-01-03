#include "c_ref_backend.h"

#include <stdio.h>
#include <string.h>

static void write_error(char *err_msg, size_t err_cap, const char *msg) {
    if (err_msg == NULL || err_cap == 0) {
        return;
    }
    strncpy(err_msg, msg, err_cap - 1);
    err_msg[err_cap - 1] = '\0';
}

int ref_run_add(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_sub(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_mul(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_div(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_maximum(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_minimum(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_neg(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_exp(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_abs(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_sqrt(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_cbrt(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_log(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_sin(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_cos(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_acos(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_acosh(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_asin(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_asinh(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_atan(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_atanh(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_cosh(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_sinh(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_tan(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_erf(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_erfc(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_expm1(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_log1p(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_log2(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_log10(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_rsqrt(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_sigmoid(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_sign(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_round(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_trunc(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_tanh(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_floor(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_ceil(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_reciprocal(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_relu(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_angle(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_conj(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_conj_physical(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_deg2rad(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_digamma(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_erfinv(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_exp2(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_frac(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_i0(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_lgamma(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_logit(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_nan_to_num(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_positive(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_rad2deg(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_real(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_sgn(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_sinc(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_square(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_atan2(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_pow(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_remainder(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_fmod(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_floor_divide(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_fmax(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_fmin(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_copysign(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_hypot(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_logaddexp(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_nextafter(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_xlogy(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_heaviside(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_ldexp(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_clamp_min(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_clamp_max(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_matmul(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_bmm(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_broadcast_in_dim(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_conv2d(const RefOpCall *call, char *err_msg, size_t err_cap);

REF_BACKEND_API int ref_run_op(
    int32_t op_kind,
    const RefOpCall *call,
    char *err_msg,
    size_t err_cap) {
    if (call == NULL) {
        write_error(err_msg, err_cap, "RefOpCall is NULL");
        return 1;
    }
    switch (op_kind) {
        case REF_OP_ADD:
            return ref_run_add(call, err_msg, err_cap);
        case REF_OP_SUB:
            return ref_run_sub(call, err_msg, err_cap);
        case REF_OP_MUL:
            return ref_run_mul(call, err_msg, err_cap);
        case REF_OP_DIV:
            return ref_run_div(call, err_msg, err_cap);
        case REF_OP_MAXIMUM:
            return ref_run_maximum(call, err_msg, err_cap);
        case REF_OP_MINIMUM:
            return ref_run_minimum(call, err_msg, err_cap);
        case REF_OP_NEG:
            return ref_run_neg(call, err_msg, err_cap);
        case REF_OP_EXP:
            return ref_run_exp(call, err_msg, err_cap);
        case REF_OP_ABS:
            return ref_run_abs(call, err_msg, err_cap);
        case REF_OP_SQRT:
            return ref_run_sqrt(call, err_msg, err_cap);
        case REF_OP_CBRT:
            return ref_run_cbrt(call, err_msg, err_cap);
        case REF_OP_LOG:
            return ref_run_log(call, err_msg, err_cap);
        case REF_OP_SIN:
            return ref_run_sin(call, err_msg, err_cap);
        case REF_OP_COS:
            return ref_run_cos(call, err_msg, err_cap);
        case REF_OP_ACOS:
            return ref_run_acos(call, err_msg, err_cap);
        case REF_OP_ACOSH:
            return ref_run_acosh(call, err_msg, err_cap);
        case REF_OP_ASIN:
            return ref_run_asin(call, err_msg, err_cap);
        case REF_OP_ASINH:
            return ref_run_asinh(call, err_msg, err_cap);
        case REF_OP_ATAN:
            return ref_run_atan(call, err_msg, err_cap);
        case REF_OP_ATANH:
            return ref_run_atanh(call, err_msg, err_cap);
        case REF_OP_COSH:
            return ref_run_cosh(call, err_msg, err_cap);
        case REF_OP_SINH:
            return ref_run_sinh(call, err_msg, err_cap);
        case REF_OP_TAN:
            return ref_run_tan(call, err_msg, err_cap);
        case REF_OP_ERF:
            return ref_run_erf(call, err_msg, err_cap);
        case REF_OP_ERFC:
            return ref_run_erfc(call, err_msg, err_cap);
        case REF_OP_EXPM1:
            return ref_run_expm1(call, err_msg, err_cap);
        case REF_OP_LOG1P:
            return ref_run_log1p(call, err_msg, err_cap);
        case REF_OP_LOG2:
            return ref_run_log2(call, err_msg, err_cap);
        case REF_OP_LOG10:
            return ref_run_log10(call, err_msg, err_cap);
        case REF_OP_RSQRT:
            return ref_run_rsqrt(call, err_msg, err_cap);
        case REF_OP_SIGMOID:
            return ref_run_sigmoid(call, err_msg, err_cap);
        case REF_OP_SIGN:
            return ref_run_sign(call, err_msg, err_cap);
        case REF_OP_ROUND:
            return ref_run_round(call, err_msg, err_cap);
        case REF_OP_TRUNC:
            return ref_run_trunc(call, err_msg, err_cap);
        case REF_OP_TANH:
            return ref_run_tanh(call, err_msg, err_cap);
        case REF_OP_FLOOR:
            return ref_run_floor(call, err_msg, err_cap);
        case REF_OP_CEIL:
            return ref_run_ceil(call, err_msg, err_cap);
        case REF_OP_RECIPROCAL:
            return ref_run_reciprocal(call, err_msg, err_cap);
        case REF_OP_RELU:
            return ref_run_relu(call, err_msg, err_cap);
        case REF_OP_ANGLE:
            return ref_run_angle(call, err_msg, err_cap);
        case REF_OP_CONJ:
            return ref_run_conj(call, err_msg, err_cap);
        case REF_OP_CONJ_PHYSICAL:
            return ref_run_conj_physical(call, err_msg, err_cap);
        case REF_OP_DEG2RAD:
            return ref_run_deg2rad(call, err_msg, err_cap);
        case REF_OP_DIGAMMA:
            return ref_run_digamma(call, err_msg, err_cap);
        case REF_OP_ERFINV:
            return ref_run_erfinv(call, err_msg, err_cap);
        case REF_OP_EXP2:
            return ref_run_exp2(call, err_msg, err_cap);
        case REF_OP_FRAC:
            return ref_run_frac(call, err_msg, err_cap);
        case REF_OP_I0:
            return ref_run_i0(call, err_msg, err_cap);
        case REF_OP_LGAMMA:
            return ref_run_lgamma(call, err_msg, err_cap);
        case REF_OP_LOGIT:
            return ref_run_logit(call, err_msg, err_cap);
        case REF_OP_NAN_TO_NUM:
            return ref_run_nan_to_num(call, err_msg, err_cap);
        case REF_OP_POSITIVE:
            return ref_run_positive(call, err_msg, err_cap);
        case REF_OP_RAD2DEG:
            return ref_run_rad2deg(call, err_msg, err_cap);
        case REF_OP_REAL:
            return ref_run_real(call, err_msg, err_cap);
        case REF_OP_SGN:
            return ref_run_sgn(call, err_msg, err_cap);
        case REF_OP_SINC:
            return ref_run_sinc(call, err_msg, err_cap);
        case REF_OP_SQUARE:
            return ref_run_square(call, err_msg, err_cap);
        case REF_OP_ATAN2:
            return ref_run_atan2(call, err_msg, err_cap);
        case REF_OP_POW:
            return ref_run_pow(call, err_msg, err_cap);
        case REF_OP_REMAINDER:
            return ref_run_remainder(call, err_msg, err_cap);
        case REF_OP_FMOD:
            return ref_run_fmod(call, err_msg, err_cap);
        case REF_OP_FLOOR_DIVIDE:
            return ref_run_floor_divide(call, err_msg, err_cap);
        case REF_OP_FMAX:
            return ref_run_fmax(call, err_msg, err_cap);
        case REF_OP_FMIN:
            return ref_run_fmin(call, err_msg, err_cap);
        case REF_OP_COPYSIGN:
            return ref_run_copysign(call, err_msg, err_cap);
        case REF_OP_HYPOT:
            return ref_run_hypot(call, err_msg, err_cap);
        case REF_OP_LOGADDEXP:
            return ref_run_logaddexp(call, err_msg, err_cap);
        case REF_OP_NEXTAFTER:
            return ref_run_nextafter(call, err_msg, err_cap);
        case REF_OP_XLOGY:
            return ref_run_xlogy(call, err_msg, err_cap);
        case REF_OP_HEAVISIDE:
            return ref_run_heaviside(call, err_msg, err_cap);
        case REF_OP_LDEXP:
            return ref_run_ldexp(call, err_msg, err_cap);
        case REF_OP_CLAMP_MIN:
            return ref_run_clamp_min(call, err_msg, err_cap);
        case REF_OP_CLAMP_MAX:
            return ref_run_clamp_max(call, err_msg, err_cap);
        case REF_OP_MATMUL:
            return ref_run_matmul(call, err_msg, err_cap);
        case REF_OP_BMM:
            return ref_run_bmm(call, err_msg, err_cap);
        case REF_OP_BROADCAST_IN_DIM:
            return ref_run_broadcast_in_dim(call, err_msg, err_cap);
        case REF_OP_CONV2D:
            return ref_run_conv2d(call, err_msg, err_cap);
        default:
            write_error(err_msg, err_cap, "Unsupported op kind");
            return 2;
    }
}
