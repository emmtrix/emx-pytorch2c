#ifndef REF_BACKEND_OPS_SCALAR_U16_H
#define REF_BACKEND_OPS_SCALAR_U16_H

#include <stdint.h>

#include "ops_scalar_f32.h"

static inline uint16_t ref_scalar_u16_from_f32(float value) {
    if (!isfinite(value)) {
        return 0;
    }
    return (uint16_t)value;
}

static inline uint16_t ref_scalar_u16_abs(uint16_t a) {
    return a;
}

static inline uint16_t ref_scalar_u16_absolute(uint16_t a) {
    return a;
}

static inline uint16_t ref_scalar_u16_add(uint16_t a, uint16_t b) {
    return (uint16_t)(a + b);
}

static inline uint16_t ref_scalar_u16_sub(uint16_t a, uint16_t b) {
    return (uint16_t)(a - b);
}

static inline uint16_t ref_scalar_u16_mul(uint16_t a, uint16_t b) {
    return (uint16_t)(a * b);
}

static inline uint16_t ref_scalar_u16_bitwise_and(uint16_t a, uint16_t b) {
    return (uint16_t)(a & b);
}

static inline uint16_t ref_scalar_u16_bitwise_or(uint16_t a, uint16_t b) {
    return (uint16_t)(a | b);
}

static inline uint16_t ref_scalar_u16_bitwise_xor(uint16_t a, uint16_t b) {
    return (uint16_t)(a ^ b);
}

static inline uint16_t ref_scalar_u16_bitwise_left_shift(uint16_t a, uint16_t b) {
    return (uint16_t)(a << b);
}

static inline uint16_t ref_scalar_u16_bitwise_right_shift(uint16_t a, uint16_t b) {
    return (uint16_t)(a >> b);
}

static inline uint16_t ref_scalar_u16_bitwise_not(uint16_t a) {
    return (uint16_t)(~a);
}

static inline uint16_t ref_scalar_u16_div(uint16_t a, uint16_t b) {
    if (b == 0) {
        return 0;
    }
    return (uint16_t)(a / b);
}

static inline uint16_t ref_scalar_u16_maximum(uint16_t a, uint16_t b) {
    return a > b ? a : b;
}

static inline uint16_t ref_scalar_u16_minimum(uint16_t a, uint16_t b) {
    return a < b ? a : b;
}

static inline uint16_t ref_scalar_u16_le(uint16_t a, uint16_t b) {
    return a <= b ? (uint16_t)1 : (uint16_t)0;
}

static inline uint16_t ref_scalar_u16_lt(uint16_t a, uint16_t b) {
    return a < b ? (uint16_t)1 : (uint16_t)0;
}

static inline uint16_t ref_scalar_u16_ge(uint16_t a, uint16_t b) {
    return a >= b ? (uint16_t)1 : (uint16_t)0;
}

static inline uint16_t ref_scalar_u16_gt(uint16_t a, uint16_t b) {
    return a > b ? (uint16_t)1 : (uint16_t)0;
}

static inline uint16_t ref_scalar_u16_eq(uint16_t a, uint16_t b) {
    return a == b ? (uint16_t)1 : (uint16_t)0;
}

static inline uint16_t ref_scalar_u16_ne(uint16_t a, uint16_t b) {
    return a != b ? (uint16_t)1 : (uint16_t)0;
}

static inline uint16_t ref_scalar_u16_logical_or(uint16_t a, uint16_t b) {
    return (a != 0 || b != 0) ? (uint16_t)1 : (uint16_t)0;
}

static inline uint16_t ref_scalar_u16_logical_and(uint16_t a, uint16_t b) {
    return (a != 0 && b != 0) ? (uint16_t)1 : (uint16_t)0;
}

static inline uint16_t ref_scalar_u16_logical_xor(uint16_t a, uint16_t b) {
    return ((a != 0) != (b != 0)) ? (uint16_t)1 : (uint16_t)0;
}

static inline uint16_t ref_scalar_u16_logical_not(uint16_t a) {
    return a == 0 ? (uint16_t)1 : (uint16_t)0;
}

static inline uint16_t ref_scalar_u16_fmax(uint16_t a, uint16_t b) {
    return a > b ? a : b;
}

static inline uint16_t ref_scalar_u16_fmin(uint16_t a, uint16_t b) {
    return a < b ? a : b;
}

static inline uint16_t ref_scalar_u16_copysign(uint16_t a, uint16_t b) {
    (void)b;
    return a;
}

static inline uint16_t ref_scalar_u16_fmod(uint16_t a, uint16_t b) {
    if (b == 0) {
        return 0;
    }
    return (uint16_t)(a % b);
}

static inline uint16_t ref_scalar_u16_remainder(uint16_t a, uint16_t b) {
    if (b == 0) {
        return 0;
    }
    return (uint16_t)(a % b);
}

static inline uint16_t ref_scalar_u16_floor_divide(uint16_t a, uint16_t b) {
    if (b == 0) {
        return 0;
    }
    return (uint16_t)(a / b);
}

static inline uint16_t ref_scalar_u16_clamp_min(uint16_t a, uint16_t b) {
    return a > b ? a : b;
}

static inline uint16_t ref_scalar_u16_clamp_max(uint16_t a, uint16_t b) {
    return a < b ? a : b;
}

static inline uint16_t ref_scalar_u16_neg(uint16_t a) {
    return (uint16_t)(0 - a);
}

static inline uint16_t ref_scalar_u16_reciprocal(uint16_t a) {
    if (a == 0) {
        return 0;
    }
    return (uint16_t)(1 / a);
}

static inline uint16_t ref_scalar_u16_relu(uint16_t a) {
    return a;
}

static inline uint16_t ref_scalar_u16_ceil(uint16_t a) {
    return a;
}

static inline uint16_t ref_scalar_u16_floor(uint16_t a) {
    return a;
}

static inline uint16_t ref_scalar_u16_round(uint16_t a) {
    return a;
}

#define REF_U16_UNARY_FROM_F32(name)                          \
    static inline uint16_t ref_scalar_u16_##name(uint16_t a) { \
        return ref_scalar_u16_from_f32(ref_scalar_f32_##name((float)a)); \
    }

#define REF_U16_BINARY_FROM_F32(name)                                 \
    static inline uint16_t ref_scalar_u16_##name(uint16_t a, uint16_t b) { \
        return ref_scalar_u16_from_f32(ref_scalar_f32_##name((float)a, (float)b)); \
    }

REF_U16_UNARY_FROM_F32(acos)
REF_U16_UNARY_FROM_F32(arccos)
REF_U16_UNARY_FROM_F32(acosh)
REF_U16_UNARY_FROM_F32(angle)
REF_U16_UNARY_FROM_F32(asin)
REF_U16_UNARY_FROM_F32(arcsin)
REF_U16_UNARY_FROM_F32(asinh)
REF_U16_UNARY_FROM_F32(arcsinh)
REF_U16_UNARY_FROM_F32(atan)
REF_U16_UNARY_FROM_F32(arctan)
REF_U16_UNARY_FROM_F32(atanh)
REF_U16_UNARY_FROM_F32(cbrt)
REF_U16_UNARY_FROM_F32(cos)
REF_U16_UNARY_FROM_F32(cosh)
REF_U16_UNARY_FROM_F32(deg2rad)
REF_U16_UNARY_FROM_F32(digamma)
REF_U16_UNARY_FROM_F32(erf)
REF_U16_UNARY_FROM_F32(erfc)
REF_U16_UNARY_FROM_F32(erfinv)
REF_U16_UNARY_FROM_F32(exp)
REF_U16_UNARY_FROM_F32(exp2)
REF_U16_UNARY_FROM_F32(expm1)
REF_U16_UNARY_FROM_F32(i0)
REF_U16_UNARY_FROM_F32(lgamma)
REF_U16_UNARY_FROM_F32(log)
REF_U16_UNARY_FROM_F32(log10)
REF_U16_UNARY_FROM_F32(log1p)
REF_U16_UNARY_FROM_F32(log2)
REF_U16_UNARY_FROM_F32(isfinite)
REF_U16_UNARY_FROM_F32(isnan)
REF_U16_UNARY_FROM_F32(logit)
REF_U16_UNARY_FROM_F32(log_sigmoid)
REF_U16_UNARY_FROM_F32(gelu)
REF_U16_UNARY_FROM_F32(elu)
REF_U16_UNARY_FROM_F32(leaky_relu)
REF_U16_UNARY_FROM_F32(softplus)
REF_U16_UNARY_FROM_F32(isinf)
REF_U16_UNARY_FROM_F32(isneginf)
REF_U16_UNARY_FROM_F32(isposinf)
REF_U16_UNARY_FROM_F32(nan_to_num)
REF_U16_UNARY_FROM_F32(rad2deg)
REF_U16_UNARY_FROM_F32(rsqrt)
REF_U16_UNARY_FROM_F32(sigmoid)
REF_U16_UNARY_FROM_F32(selu)
REF_U16_UNARY_FROM_F32(relu6)
REF_U16_UNARY_FROM_F32(hardsigmoid)
REF_U16_UNARY_FROM_F32(silu)
REF_U16_UNARY_FROM_F32(mish)
REF_U16_UNARY_FROM_F32(hardswish)
REF_U16_UNARY_FROM_F32(sin)
REF_U16_UNARY_FROM_F32(sinc)
REF_U16_UNARY_FROM_F32(sinh)
REF_U16_UNARY_FROM_F32(sqrt)
REF_U16_UNARY_FROM_F32(tan)
REF_U16_UNARY_FROM_F32(tanh)

REF_U16_BINARY_FROM_F32(atan2)
REF_U16_BINARY_FROM_F32(heaviside)
REF_U16_BINARY_FROM_F32(hypot)
REF_U16_BINARY_FROM_F32(ldexp)
REF_U16_BINARY_FROM_F32(logaddexp)
REF_U16_BINARY_FROM_F32(logaddexp2)
REF_U16_BINARY_FROM_F32(nextafter)
REF_U16_BINARY_FROM_F32(pow)
REF_U16_BINARY_FROM_F32(xlogy)

#undef REF_U16_UNARY_FROM_F32
#undef REF_U16_BINARY_FROM_F32

#endif
