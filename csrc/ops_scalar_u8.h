#ifndef REF_BACKEND_OPS_SCALAR_U8_H
#define REF_BACKEND_OPS_SCALAR_U8_H

#include <stdint.h>

#include "ops_scalar_f32.h"

static inline uint8_t ref_scalar_u8_from_f32(float value) {
    if (!isfinite(value)) {
        return 0;
    }
    return (uint8_t)value;
}

static inline uint8_t ref_scalar_u8_abs(uint8_t a) {
    return a;
}

static inline uint8_t ref_scalar_u8_absolute(uint8_t a) {
    return a;
}

static inline uint8_t ref_scalar_u8_add(uint8_t a, uint8_t b) {
    return (uint8_t)(a + b);
}

static inline uint8_t ref_scalar_u8_sub(uint8_t a, uint8_t b) {
    return (uint8_t)(a - b);
}

static inline uint8_t ref_scalar_u8_mul(uint8_t a, uint8_t b) {
    return (uint8_t)(a * b);
}

static inline uint8_t ref_scalar_u8_bitwise_and(uint8_t a, uint8_t b) {
    return (uint8_t)(a & b);
}

static inline uint8_t ref_scalar_u8_bitwise_or(uint8_t a, uint8_t b) {
    return (uint8_t)(a | b);
}

static inline uint8_t ref_scalar_u8_bitwise_xor(uint8_t a, uint8_t b) {
    return (uint8_t)(a ^ b);
}

static inline uint8_t ref_scalar_u8_bitwise_left_shift(uint8_t a, uint8_t b) {
    return (uint8_t)(a << b);
}

static inline uint8_t ref_scalar_u8_bitwise_right_shift(uint8_t a, uint8_t b) {
    return (uint8_t)(a >> b);
}

static inline uint8_t ref_scalar_u8_bitwise_not(uint8_t a) {
    return (uint8_t)(~a);
}

static inline uint8_t ref_scalar_u8_div(uint8_t a, uint8_t b) {
    if (b == 0) {
        return 0;
    }
    return (uint8_t)(a / b);
}

static inline uint8_t ref_scalar_u8_maximum(uint8_t a, uint8_t b) {
    return a > b ? a : b;
}

static inline uint8_t ref_scalar_u8_minimum(uint8_t a, uint8_t b) {
    return a < b ? a : b;
}

static inline uint8_t ref_scalar_u8_le(uint8_t a, uint8_t b) {
    return a <= b ? (uint8_t)1 : (uint8_t)0;
}

static inline uint8_t ref_scalar_u8_lt(uint8_t a, uint8_t b) {
    return a < b ? (uint8_t)1 : (uint8_t)0;
}

static inline uint8_t ref_scalar_u8_ge(uint8_t a, uint8_t b) {
    return a >= b ? (uint8_t)1 : (uint8_t)0;
}

static inline uint8_t ref_scalar_u8_gt(uint8_t a, uint8_t b) {
    return a > b ? (uint8_t)1 : (uint8_t)0;
}

static inline uint8_t ref_scalar_u8_eq(uint8_t a, uint8_t b) {
    return a == b ? (uint8_t)1 : (uint8_t)0;
}

static inline uint8_t ref_scalar_u8_ne(uint8_t a, uint8_t b) {
    return a != b ? (uint8_t)1 : (uint8_t)0;
}

static inline uint8_t ref_scalar_u8_logical_or(uint8_t a, uint8_t b) {
    return (a != 0 || b != 0) ? (uint8_t)1 : (uint8_t)0;
}

static inline uint8_t ref_scalar_u8_logical_and(uint8_t a, uint8_t b) {
    return (a != 0 && b != 0) ? (uint8_t)1 : (uint8_t)0;
}

static inline uint8_t ref_scalar_u8_logical_xor(uint8_t a, uint8_t b) {
    return ((a != 0) != (b != 0)) ? (uint8_t)1 : (uint8_t)0;
}

static inline uint8_t ref_scalar_u8_logical_not(uint8_t a) {
    return a == 0 ? (uint8_t)1 : (uint8_t)0;
}

static inline uint8_t ref_scalar_u8_fmax(uint8_t a, uint8_t b) {
    return a > b ? a : b;
}

static inline uint8_t ref_scalar_u8_fmin(uint8_t a, uint8_t b) {
    return a < b ? a : b;
}

static inline uint8_t ref_scalar_u8_copysign(uint8_t a, uint8_t b) {
    (void)b;
    return a;
}

static inline uint8_t ref_scalar_u8_fmod(uint8_t a, uint8_t b) {
    if (b == 0) {
        return 0;
    }
    return (uint8_t)(a % b);
}

static inline uint8_t ref_scalar_u8_remainder(uint8_t a, uint8_t b) {
    if (b == 0) {
        return 0;
    }
    return (uint8_t)(a % b);
}

static inline uint8_t ref_scalar_u8_floor_divide(uint8_t a, uint8_t b) {
    if (b == 0) {
        return 0;
    }
    return (uint8_t)(a / b);
}

static inline uint8_t ref_scalar_u8_clamp_min(uint8_t a, uint8_t b) {
    return a > b ? a : b;
}

static inline uint8_t ref_scalar_u8_clamp_max(uint8_t a, uint8_t b) {
    return a < b ? a : b;
}

static inline uint8_t ref_scalar_u8_neg(uint8_t a) {
    return (uint8_t)(0 - a);
}

static inline uint8_t ref_scalar_u8_reciprocal(uint8_t a) {
    if (a == 0) {
        return 0;
    }
    return (uint8_t)(1 / a);
}

static inline uint8_t ref_scalar_u8_relu(uint8_t a) {
    return a;
}

static inline uint8_t ref_scalar_u8_ceil(uint8_t a) {
    return a;
}

static inline uint8_t ref_scalar_u8_floor(uint8_t a) {
    return a;
}

static inline uint8_t ref_scalar_u8_round(uint8_t a) {
    return a;
}

#define REF_U8_UNARY_FROM_F32(name)                         \
    static inline uint8_t ref_scalar_u8_##name(uint8_t a) { \
        return ref_scalar_u8_from_f32(ref_scalar_f32_##name((float)a)); \
    }

#define REF_U8_BINARY_FROM_F32(name)                                \
    static inline uint8_t ref_scalar_u8_##name(uint8_t a, uint8_t b) { \
        return ref_scalar_u8_from_f32(ref_scalar_f32_##name(         \
            (float)a, (float)b));                                   \
    }

REF_U8_UNARY_FROM_F32(acos)
REF_U8_UNARY_FROM_F32(arccos)
REF_U8_UNARY_FROM_F32(acosh)
REF_U8_UNARY_FROM_F32(angle)
REF_U8_UNARY_FROM_F32(asin)
REF_U8_UNARY_FROM_F32(arcsin)
REF_U8_UNARY_FROM_F32(asinh)
REF_U8_UNARY_FROM_F32(arcsinh)
REF_U8_UNARY_FROM_F32(atan)
REF_U8_UNARY_FROM_F32(arctan)
REF_U8_UNARY_FROM_F32(atanh)
REF_U8_UNARY_FROM_F32(cbrt)
REF_U8_UNARY_FROM_F32(cos)
REF_U8_UNARY_FROM_F32(cosh)
REF_U8_UNARY_FROM_F32(deg2rad)
REF_U8_UNARY_FROM_F32(digamma)
REF_U8_UNARY_FROM_F32(erf)
REF_U8_UNARY_FROM_F32(erfc)
REF_U8_UNARY_FROM_F32(erfinv)
REF_U8_UNARY_FROM_F32(exp)
REF_U8_UNARY_FROM_F32(exp2)
REF_U8_UNARY_FROM_F32(expm1)
REF_U8_UNARY_FROM_F32(i0)
REF_U8_UNARY_FROM_F32(lgamma)
REF_U8_UNARY_FROM_F32(log)
REF_U8_UNARY_FROM_F32(log10)
REF_U8_UNARY_FROM_F32(log1p)
REF_U8_UNARY_FROM_F32(log2)
REF_U8_UNARY_FROM_F32(isfinite)
REF_U8_UNARY_FROM_F32(isnan)
REF_U8_UNARY_FROM_F32(logit)
REF_U8_UNARY_FROM_F32(log_sigmoid)
REF_U8_UNARY_FROM_F32(gelu)
REF_U8_UNARY_FROM_F32(elu)
REF_U8_UNARY_FROM_F32(leaky_relu)
REF_U8_UNARY_FROM_F32(softplus)
REF_U8_UNARY_FROM_F32(isinf)
REF_U8_UNARY_FROM_F32(isneginf)
REF_U8_UNARY_FROM_F32(isposinf)
REF_U8_UNARY_FROM_F32(nan_to_num)
REF_U8_UNARY_FROM_F32(rad2deg)
REF_U8_UNARY_FROM_F32(rsqrt)
REF_U8_UNARY_FROM_F32(sigmoid)
REF_U8_UNARY_FROM_F32(selu)
REF_U8_UNARY_FROM_F32(relu6)
REF_U8_UNARY_FROM_F32(hardsigmoid)
REF_U8_UNARY_FROM_F32(silu)
REF_U8_UNARY_FROM_F32(mish)
REF_U8_UNARY_FROM_F32(hardswish)
REF_U8_UNARY_FROM_F32(sin)
REF_U8_UNARY_FROM_F32(sinc)
REF_U8_UNARY_FROM_F32(sinh)
REF_U8_UNARY_FROM_F32(sqrt)
REF_U8_UNARY_FROM_F32(tan)
REF_U8_UNARY_FROM_F32(tanh)

REF_U8_BINARY_FROM_F32(atan2)
REF_U8_BINARY_FROM_F32(heaviside)
REF_U8_BINARY_FROM_F32(hypot)
REF_U8_BINARY_FROM_F32(ldexp)
REF_U8_BINARY_FROM_F32(logaddexp)
REF_U8_BINARY_FROM_F32(logaddexp2)
REF_U8_BINARY_FROM_F32(nextafter)
REF_U8_BINARY_FROM_F32(pow)
REF_U8_BINARY_FROM_F32(xlogy)

#undef REF_U8_UNARY_FROM_F32
#undef REF_U8_BINARY_FROM_F32

#endif
