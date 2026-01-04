#ifndef REF_BACKEND_OPS_SCALAR_I32_H
#define REF_BACKEND_OPS_SCALAR_I32_H

#include <limits.h>
#include <stdint.h>

#include "ops_scalar_f32.h"

static inline int32_t ref_scalar_i32_from_f32(float value) {
    if (!isfinite(value)) {
        return INT32_MIN;
    }
    if (value > (float)INT32_MAX) {
        return INT32_MAX;
    }
    if (value < (float)INT32_MIN) {
        return INT32_MIN;
    }
    return (int32_t)value;
}

static inline int32_t ref_scalar_i32_abs(int32_t a) {
    if (a == INT32_MIN) {
        return INT32_MIN;
    }
    return a < 0 ? -a : a;
}

static inline int32_t ref_scalar_i32_add(int32_t a, int32_t b) {
    return a + b;
}

static inline int32_t ref_scalar_i32_sub(int32_t a, int32_t b) {
    return a - b;
}

static inline int32_t ref_scalar_i32_mul(int32_t a, int32_t b) {
    return a * b;
}

static inline int32_t ref_scalar_i32_bitwise_and(int32_t a, int32_t b) {
    return a & b;
}

static inline int32_t ref_scalar_i32_bitwise_or(int32_t a, int32_t b) {
    return a | b;
}

static inline int32_t ref_scalar_i32_bitwise_xor(int32_t a, int32_t b) {
    return a ^ b;
}

static inline int32_t ref_scalar_i32_bitwise_left_shift(int32_t a, int32_t b) {
    return a << b;
}

static inline int32_t ref_scalar_i32_bitwise_right_shift(int32_t a, int32_t b) {
    return a >> b;
}

static inline int32_t ref_scalar_i32_bitwise_not(int32_t a) {
    return ~a;
}

static inline int32_t ref_scalar_i32_div(int32_t a, int32_t b) {
    if (b == 0) {
        return 0;
    }
    return a / b;
}

static inline int32_t ref_scalar_i32_maximum(int32_t a, int32_t b) {
    return a > b ? a : b;
}

static inline int32_t ref_scalar_i32_minimum(int32_t a, int32_t b) {
    return a < b ? a : b;
}

static inline int32_t ref_scalar_i32_le(int32_t a, int32_t b) {
    return a <= b ? 1 : 0;
}

static inline int32_t ref_scalar_i32_lt(int32_t a, int32_t b) {
    return a < b ? 1 : 0;
}

static inline int32_t ref_scalar_i32_ge(int32_t a, int32_t b) {
    return a >= b ? 1 : 0;
}

static inline int32_t ref_scalar_i32_gt(int32_t a, int32_t b) {
    return a > b ? 1 : 0;
}

static inline int32_t ref_scalar_i32_eq(int32_t a, int32_t b) {
    return a == b ? 1 : 0;
}

static inline int32_t ref_scalar_i32_ne(int32_t a, int32_t b) {
    return a != b ? 1 : 0;
}

static inline int32_t ref_scalar_i32_logical_or(int32_t a, int32_t b) {
    return (a != 0 || b != 0) ? 1 : 0;
}

static inline int32_t ref_scalar_i32_logical_and(int32_t a, int32_t b) {
    return (a != 0 && b != 0) ? 1 : 0;
}

static inline int32_t ref_scalar_i32_logical_xor(int32_t a, int32_t b) {
    return ((a != 0) != (b != 0)) ? 1 : 0;
}

static inline int32_t ref_scalar_i32_logical_not(int32_t a) {
    return a == 0 ? 1 : 0;
}

static inline int32_t ref_scalar_i32_fmax(int32_t a, int32_t b) {
    return a > b ? a : b;
}

static inline int32_t ref_scalar_i32_fmin(int32_t a, int32_t b) {
    return a < b ? a : b;
}

static inline int32_t ref_scalar_i32_copysign(int32_t a, int32_t b) {
    int32_t magnitude = ref_scalar_i32_abs(a);
    return b < 0 ? -magnitude : magnitude;
}

static inline int32_t ref_scalar_i32_fmod(int32_t a, int32_t b) {
    if (b == 0) {
        return 0;
    }
    return a % b;
}

static inline int32_t ref_scalar_i32_remainder(int32_t a, int32_t b) {
    if (b == 0) {
        return 0;
    }
    int32_t mod = a % b;
    if (mod == 0) {
        return mod;
    }
    if ((mod < 0) != (b < 0)) {
        mod += b;
    }
    return mod;
}

static inline int32_t ref_scalar_i32_floor_divide(int32_t a, int32_t b) {
    if (b == 0) {
        return 0;
    }
    int32_t quo = a / b;
    int32_t rem = a % b;
    if (rem != 0 && ((rem < 0) != (b < 0))) {
        quo -= 1;
    }
    return quo;
}

static inline int32_t ref_scalar_i32_clamp_min(int32_t a, int32_t b) {
    return a > b ? a : b;
}

static inline int32_t ref_scalar_i32_clamp_max(int32_t a, int32_t b) {
    return a < b ? a : b;
}

static inline int32_t ref_scalar_i32_neg(int32_t a) {
    if (a == INT32_MIN) {
        return INT32_MIN;
    }
    return -a;
}

static inline int32_t ref_scalar_i32_reciprocal(int32_t a) {
    if (a == 0) {
        return 0;
    }
    return 1 / a;
}

static inline int32_t ref_scalar_i32_relu(int32_t a) {
    return a > 0 ? a : 0;
}

static inline int32_t ref_scalar_i32_ceil(int32_t a) {
    return a;
}

static inline int32_t ref_scalar_i32_floor(int32_t a) {
    return a;
}

static inline int32_t ref_scalar_i32_round(int32_t a) {
    return a;
}

static inline int32_t ref_scalar_i32_trunc(int32_t a) {
    return a;
}

static inline int32_t ref_scalar_i32_frac(int32_t a) {
    (void)a;
    return 0;
}

static inline int32_t ref_scalar_i32_sign(int32_t a) {
    if (a > 0) {
        return 1;
    }
    if (a < 0) {
        return -1;
    }
    return 0;
}

static inline int32_t ref_scalar_i32_conj(int32_t a) {
    return a;
}

static inline int32_t ref_scalar_i32_conj_physical(int32_t a) {
    return a;
}

static inline int32_t ref_scalar_i32_positive(int32_t a) {
    return a;
}

static inline int32_t ref_scalar_i32_real(int32_t a) {
    return a;
}

static inline int32_t ref_scalar_i32_sgn(int32_t a) {
    if (a > 0) {
        return 1;
    }
    if (a < 0) {
        return -1;
    }
    return 0;
}

static inline int32_t ref_scalar_i32_square(int32_t a) {
    return a * a;
}

#define REF_I32_UNARY_FROM_F32(name)                          \
    static inline int32_t ref_scalar_i32_##name(int32_t a) {   \
        return ref_scalar_i32_from_f32(ref_scalar_f32_##name(  \
            (float)a));                                       \
    }

#define REF_I32_BINARY_FROM_F32(name)                                 \
    static inline int32_t ref_scalar_i32_##name(int32_t a, int32_t b) { \
        return ref_scalar_i32_from_f32(ref_scalar_f32_##name(         \
            (float)a, (float)b));                                     \
    }

REF_I32_UNARY_FROM_F32(acos)
REF_I32_UNARY_FROM_F32(acosh)
REF_I32_UNARY_FROM_F32(angle)
REF_I32_UNARY_FROM_F32(asin)
REF_I32_UNARY_FROM_F32(asinh)
REF_I32_UNARY_FROM_F32(atan)
REF_I32_UNARY_FROM_F32(atanh)
REF_I32_UNARY_FROM_F32(cbrt)
REF_I32_UNARY_FROM_F32(cos)
REF_I32_UNARY_FROM_F32(cosh)
REF_I32_UNARY_FROM_F32(deg2rad)
REF_I32_UNARY_FROM_F32(digamma)
REF_I32_UNARY_FROM_F32(erf)
REF_I32_UNARY_FROM_F32(erfc)
REF_I32_UNARY_FROM_F32(erfinv)
REF_I32_UNARY_FROM_F32(exp)
REF_I32_UNARY_FROM_F32(exp2)
REF_I32_UNARY_FROM_F32(expm1)
REF_I32_UNARY_FROM_F32(i0)
REF_I32_UNARY_FROM_F32(lgamma)
REF_I32_UNARY_FROM_F32(log)
REF_I32_UNARY_FROM_F32(log10)
REF_I32_UNARY_FROM_F32(log1p)
REF_I32_UNARY_FROM_F32(log2)
REF_I32_UNARY_FROM_F32(isfinite)
REF_I32_UNARY_FROM_F32(isnan)
REF_I32_UNARY_FROM_F32(logit)
REF_I32_UNARY_FROM_F32(isinf)
REF_I32_UNARY_FROM_F32(isneginf)
REF_I32_UNARY_FROM_F32(isposinf)
REF_I32_UNARY_FROM_F32(nan_to_num)
REF_I32_UNARY_FROM_F32(rad2deg)
REF_I32_UNARY_FROM_F32(rsqrt)
REF_I32_UNARY_FROM_F32(sigmoid)
REF_I32_UNARY_FROM_F32(silu)
REF_I32_UNARY_FROM_F32(mish)
REF_I32_UNARY_FROM_F32(hardswish)
REF_I32_UNARY_FROM_F32(sin)
REF_I32_UNARY_FROM_F32(sinc)
REF_I32_UNARY_FROM_F32(sinh)
REF_I32_UNARY_FROM_F32(sqrt)
REF_I32_UNARY_FROM_F32(tan)
REF_I32_UNARY_FROM_F32(tanh)

REF_I32_BINARY_FROM_F32(atan2)
REF_I32_BINARY_FROM_F32(heaviside)
REF_I32_BINARY_FROM_F32(hypot)
REF_I32_BINARY_FROM_F32(ldexp)
REF_I32_BINARY_FROM_F32(logaddexp)
REF_I32_BINARY_FROM_F32(nextafter)
REF_I32_BINARY_FROM_F32(pow)
REF_I32_BINARY_FROM_F32(xlogy)

#undef REF_I32_UNARY_FROM_F32
#undef REF_I32_BINARY_FROM_F32

#endif
