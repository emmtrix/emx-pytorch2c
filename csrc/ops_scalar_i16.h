#ifndef REF_BACKEND_OPS_SCALAR_I16_H
#define REF_BACKEND_OPS_SCALAR_I16_H

#include <limits.h>
#include <stdint.h>

#include "ops_scalar_f32.h"

static inline int16_t ref_scalar_i16_from_f32(float value) {
    if (!isfinite(value)) {
        return 0;
    }
    return (int16_t)value;
}

static inline int16_t ref_scalar_i16_abs(int16_t a) {
    if (a == INT16_MIN) {
        return INT16_MIN;
    }
    return a < 0 ? (int16_t)-a : a;
}

static inline int16_t ref_scalar_i16_absolute(int16_t a) {
    return ref_scalar_i16_abs(a);
}

static inline int16_t ref_scalar_i16_add(int16_t a, int16_t b) {
    return (int16_t)(a + b);
}

static inline int16_t ref_scalar_i16_sub(int16_t a, int16_t b) {
    return (int16_t)(a - b);
}

static inline int16_t ref_scalar_i16_mul(int16_t a, int16_t b) {
    return (int16_t)(a * b);
}

static inline int16_t ref_scalar_i16_bitwise_and(int16_t a, int16_t b) {
    return (int16_t)(a & b);
}

static inline int16_t ref_scalar_i16_bitwise_or(int16_t a, int16_t b) {
    return (int16_t)(a | b);
}

static inline int16_t ref_scalar_i16_bitwise_xor(int16_t a, int16_t b) {
    return (int16_t)(a ^ b);
}

static inline int16_t ref_scalar_i16_bitwise_left_shift(int16_t a, int16_t b) {
    return (int16_t)(a << b);
}

static inline int16_t ref_scalar_i16_bitwise_right_shift(int16_t a, int16_t b) {
    return (int16_t)(a >> b);
}

static inline int16_t ref_scalar_i16_bitwise_not(int16_t a) {
    return (int16_t)(~a);
}

static inline int16_t ref_scalar_i16_div(int16_t a, int16_t b) {
    if (b == 0) {
        return 0;
    }
    return (int16_t)(a / b);
}

static inline int16_t ref_scalar_i16_maximum(int16_t a, int16_t b) {
    return a > b ? a : b;
}

static inline int16_t ref_scalar_i16_minimum(int16_t a, int16_t b) {
    return a < b ? a : b;
}

static inline int16_t ref_scalar_i16_le(int16_t a, int16_t b) {
    return a <= b ? (int16_t)1 : (int16_t)0;
}

static inline int16_t ref_scalar_i16_lt(int16_t a, int16_t b) {
    return a < b ? (int16_t)1 : (int16_t)0;
}

static inline int16_t ref_scalar_i16_ge(int16_t a, int16_t b) {
    return a >= b ? (int16_t)1 : (int16_t)0;
}

static inline int16_t ref_scalar_i16_gt(int16_t a, int16_t b) {
    return a > b ? (int16_t)1 : (int16_t)0;
}

static inline int16_t ref_scalar_i16_eq(int16_t a, int16_t b) {
    return a == b ? (int16_t)1 : (int16_t)0;
}

static inline int16_t ref_scalar_i16_ne(int16_t a, int16_t b) {
    return a != b ? (int16_t)1 : (int16_t)0;
}

static inline int16_t ref_scalar_i16_logical_or(int16_t a, int16_t b) {
    return (a != 0 || b != 0) ? (int16_t)1 : (int16_t)0;
}

static inline int16_t ref_scalar_i16_logical_and(int16_t a, int16_t b) {
    return (a != 0 && b != 0) ? (int16_t)1 : (int16_t)0;
}

static inline int16_t ref_scalar_i16_logical_xor(int16_t a, int16_t b) {
    return ((a != 0) != (b != 0)) ? (int16_t)1 : (int16_t)0;
}

static inline int16_t ref_scalar_i16_logical_not(int16_t a) {
    return a == 0 ? (int16_t)1 : (int16_t)0;
}

static inline int16_t ref_scalar_i16_fmax(int16_t a, int16_t b) {
    return a > b ? a : b;
}

static inline int16_t ref_scalar_i16_fmin(int16_t a, int16_t b) {
    return a < b ? a : b;
}

static inline int16_t ref_scalar_i16_copysign(int16_t a, int16_t b) {
    int16_t magnitude = ref_scalar_i16_abs(a);
    return b < 0 ? (int16_t)-magnitude : magnitude;
}

static inline int16_t ref_scalar_i16_fmod(int16_t a, int16_t b) {
    if (b == 0) {
        return 0;
    }
    return (int16_t)(a % b);
}

static inline int16_t ref_scalar_i16_remainder(int16_t a, int16_t b) {
    if (b == 0) {
        return 0;
    }
    int16_t mod = (int16_t)(a % b);
    if (mod == 0) {
        return mod;
    }
    if ((mod < 0) != (b < 0)) {
        mod = (int16_t)(mod + b);
    }
    return mod;
}

static inline int16_t ref_scalar_i16_floor_divide(int16_t a, int16_t b) {
    if (b == 0) {
        return 0;
    }
    int16_t quo = (int16_t)(a / b);
    int16_t rem = (int16_t)(a % b);
    if (rem != 0 && ((rem < 0) != (b < 0))) {
        quo = (int16_t)(quo - 1);
    }
    return quo;
}

static inline int16_t ref_scalar_i16_clamp_min(int16_t a, int16_t b) {
    return a > b ? a : b;
}

static inline int16_t ref_scalar_i16_clamp_max(int16_t a, int16_t b) {
    return a < b ? a : b;
}

static inline int16_t ref_scalar_i16_neg(int16_t a) {
    if (a == INT16_MIN) {
        return INT16_MIN;
    }
    return (int16_t)-a;
}

static inline int16_t ref_scalar_i16_reciprocal(int16_t a) {
    if (a == 0) {
        return 0;
    }
    return (int16_t)(1 / a);
}

static inline int16_t ref_scalar_i16_relu(int16_t a) {
    return a > 0 ? a : 0;
}

static inline int16_t ref_scalar_i16_ceil(int16_t a) {
    return a;
}

static inline int16_t ref_scalar_i16_floor(int16_t a) {
    return a;
}

static inline int16_t ref_scalar_i16_round(int16_t a) {
    return a;
}

static inline int16_t ref_scalar_i16_trunc(int16_t a) {
    return a;
}

static inline int16_t ref_scalar_i16_frac(int16_t a) {
    (void)a;
    return 0;
}

static inline int16_t ref_scalar_i16_sign(int16_t a) {
    if (a > 0) {
        return 1;
    }
    if (a < 0) {
        return -1;
    }
    return 0;
}

static inline int16_t ref_scalar_i16_conj(int16_t a) {
    return a;
}

static inline int16_t ref_scalar_i16_conj_physical(int16_t a) {
    return a;
}

static inline int16_t ref_scalar_i16_positive(int16_t a) {
    return a;
}

static inline int16_t ref_scalar_i16_real(int16_t a) {
    return a;
}

static inline int16_t ref_scalar_i16_sgn(int16_t a) {
    if (a > 0) {
        return 1;
    }
    if (a < 0) {
        return -1;
    }
    return 0;
}

static inline int16_t ref_scalar_i16_square(int16_t a) {
    return (int16_t)(a * a);
}

#define REF_I16_UNARY_FROM_F32(name)                          \
    static inline int16_t ref_scalar_i16_##name(int16_t a) {  \
        return ref_scalar_i16_from_f32(ref_scalar_f32_##name( \
            (float)a));                                       \
    }

#define REF_I16_BINARY_FROM_F32(name)                                \
    static inline int16_t ref_scalar_i16_##name(int16_t a, int16_t b) { \
        return ref_scalar_i16_from_f32(ref_scalar_f32_##name(         \
            (float)a, (float)b));                                    \
    }

REF_I16_UNARY_FROM_F32(acos)
REF_I16_UNARY_FROM_F32(arccos)
REF_I16_UNARY_FROM_F32(acosh)
REF_I16_UNARY_FROM_F32(angle)
REF_I16_UNARY_FROM_F32(asin)
REF_I16_UNARY_FROM_F32(arcsin)
REF_I16_UNARY_FROM_F32(asinh)
REF_I16_UNARY_FROM_F32(arcsinh)
REF_I16_UNARY_FROM_F32(atan)
REF_I16_UNARY_FROM_F32(arctan)
REF_I16_UNARY_FROM_F32(atanh)
REF_I16_UNARY_FROM_F32(cbrt)
REF_I16_UNARY_FROM_F32(cos)
REF_I16_UNARY_FROM_F32(cosh)
REF_I16_UNARY_FROM_F32(deg2rad)
REF_I16_UNARY_FROM_F32(digamma)
REF_I16_UNARY_FROM_F32(erf)
REF_I16_UNARY_FROM_F32(erfc)
REF_I16_UNARY_FROM_F32(erfinv)
REF_I16_UNARY_FROM_F32(exp)
REF_I16_UNARY_FROM_F32(exp2)
REF_I16_UNARY_FROM_F32(expm1)
REF_I16_UNARY_FROM_F32(i0)
REF_I16_UNARY_FROM_F32(lgamma)
REF_I16_UNARY_FROM_F32(log)
REF_I16_UNARY_FROM_F32(log10)
REF_I16_UNARY_FROM_F32(log1p)
REF_I16_UNARY_FROM_F32(log2)
REF_I16_UNARY_FROM_F32(isfinite)
REF_I16_UNARY_FROM_F32(isnan)
REF_I16_UNARY_FROM_F32(logit)
REF_I16_UNARY_FROM_F32(log_sigmoid)
REF_I16_UNARY_FROM_F32(gelu)
REF_I16_UNARY_FROM_F32(elu)
REF_I16_UNARY_FROM_F32(leaky_relu)
REF_I16_UNARY_FROM_F32(softplus)
REF_I16_UNARY_FROM_F32(isinf)
REF_I16_UNARY_FROM_F32(isneginf)
REF_I16_UNARY_FROM_F32(isposinf)
REF_I16_UNARY_FROM_F32(nan_to_num)
REF_I16_UNARY_FROM_F32(rad2deg)
REF_I16_UNARY_FROM_F32(rsqrt)
REF_I16_UNARY_FROM_F32(sigmoid)
REF_I16_UNARY_FROM_F32(selu)
REF_I16_UNARY_FROM_F32(relu6)
REF_I16_UNARY_FROM_F32(hardsigmoid)
REF_I16_UNARY_FROM_F32(silu)
REF_I16_UNARY_FROM_F32(mish)
REF_I16_UNARY_FROM_F32(hardswish)
REF_I16_UNARY_FROM_F32(sin)
REF_I16_UNARY_FROM_F32(sinc)
REF_I16_UNARY_FROM_F32(sinh)
REF_I16_UNARY_FROM_F32(sqrt)
REF_I16_UNARY_FROM_F32(tan)
REF_I16_UNARY_FROM_F32(tanh)

REF_I16_BINARY_FROM_F32(atan2)
REF_I16_BINARY_FROM_F32(heaviside)
REF_I16_BINARY_FROM_F32(hypot)
REF_I16_BINARY_FROM_F32(ldexp)
REF_I16_BINARY_FROM_F32(logaddexp)
REF_I16_BINARY_FROM_F32(logaddexp2)
REF_I16_BINARY_FROM_F32(nextafter)
REF_I16_BINARY_FROM_F32(pow)
REF_I16_BINARY_FROM_F32(xlogy)

#undef REF_I16_UNARY_FROM_F32
#undef REF_I16_BINARY_FROM_F32

#endif
