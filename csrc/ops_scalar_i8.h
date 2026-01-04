#ifndef REF_BACKEND_OPS_SCALAR_I8_H
#define REF_BACKEND_OPS_SCALAR_I8_H

#include <limits.h>
#include <stdint.h>

#include "ops_scalar_f32.h"

static inline int8_t ref_scalar_i8_from_f32(float value) {
    if (!isfinite(value)) {
        return 0;
    }
    return (int8_t)value;
}

static inline int8_t ref_scalar_i8_abs(int8_t a) {
    if (a == INT8_MIN) {
        return INT8_MIN;
    }
    return a < 0 ? (int8_t)-a : a;
}

static inline int8_t ref_scalar_i8_add(int8_t a, int8_t b) {
    return (int8_t)(a + b);
}

static inline int8_t ref_scalar_i8_sub(int8_t a, int8_t b) {
    return (int8_t)(a - b);
}

static inline int8_t ref_scalar_i8_mul(int8_t a, int8_t b) {
    return (int8_t)(a * b);
}

static inline int8_t ref_scalar_i8_bitwise_and(int8_t a, int8_t b) {
    return (int8_t)(a & b);
}

static inline int8_t ref_scalar_i8_div(int8_t a, int8_t b) {
    if (b == 0) {
        return 0;
    }
    return (int8_t)(a / b);
}

static inline int8_t ref_scalar_i8_maximum(int8_t a, int8_t b) {
    return a > b ? a : b;
}

static inline int8_t ref_scalar_i8_minimum(int8_t a, int8_t b) {
    return a < b ? a : b;
}

static inline int8_t ref_scalar_i8_fmax(int8_t a, int8_t b) {
    return a > b ? a : b;
}

static inline int8_t ref_scalar_i8_fmin(int8_t a, int8_t b) {
    return a < b ? a : b;
}

static inline int8_t ref_scalar_i8_copysign(int8_t a, int8_t b) {
    int8_t magnitude = ref_scalar_i8_abs(a);
    return b < 0 ? (int8_t)-magnitude : magnitude;
}

static inline int8_t ref_scalar_i8_fmod(int8_t a, int8_t b) {
    if (b == 0) {
        return 0;
    }
    return (int8_t)(a % b);
}

static inline int8_t ref_scalar_i8_remainder(int8_t a, int8_t b) {
    if (b == 0) {
        return 0;
    }
    int8_t mod = (int8_t)(a % b);
    if (mod == 0) {
        return mod;
    }
    if ((mod < 0) != (b < 0)) {
        mod = (int8_t)(mod + b);
    }
    return mod;
}

static inline int8_t ref_scalar_i8_floor_divide(int8_t a, int8_t b) {
    if (b == 0) {
        return 0;
    }
    int8_t quo = (int8_t)(a / b);
    int8_t rem = (int8_t)(a % b);
    if (rem != 0 && ((rem < 0) != (b < 0))) {
        quo = (int8_t)(quo - 1);
    }
    return quo;
}

static inline int8_t ref_scalar_i8_clamp_min(int8_t a, int8_t b) {
    return a > b ? a : b;
}

static inline int8_t ref_scalar_i8_clamp_max(int8_t a, int8_t b) {
    return a < b ? a : b;
}

static inline int8_t ref_scalar_i8_neg(int8_t a) {
    if (a == INT8_MIN) {
        return INT8_MIN;
    }
    return (int8_t)-a;
}

static inline int8_t ref_scalar_i8_reciprocal(int8_t a) {
    if (a == 0) {
        return 0;
    }
    return (int8_t)(1 / a);
}

static inline int8_t ref_scalar_i8_relu(int8_t a) {
    return a > 0 ? a : 0;
}

static inline int8_t ref_scalar_i8_ceil(int8_t a) {
    return a;
}

static inline int8_t ref_scalar_i8_floor(int8_t a) {
    return a;
}

static inline int8_t ref_scalar_i8_round(int8_t a) {
    return a;
}

static inline int8_t ref_scalar_i8_trunc(int8_t a) {
    return a;
}

static inline int8_t ref_scalar_i8_frac(int8_t a) {
    (void)a;
    return 0;
}

static inline int8_t ref_scalar_i8_sign(int8_t a) {
    if (a > 0) {
        return 1;
    }
    if (a < 0) {
        return -1;
    }
    return 0;
}

static inline int8_t ref_scalar_i8_conj(int8_t a) {
    return a;
}

static inline int8_t ref_scalar_i8_conj_physical(int8_t a) {
    return a;
}

static inline int8_t ref_scalar_i8_positive(int8_t a) {
    return a;
}

static inline int8_t ref_scalar_i8_real(int8_t a) {
    return a;
}

static inline int8_t ref_scalar_i8_sgn(int8_t a) {
    if (a > 0) {
        return 1;
    }
    if (a < 0) {
        return -1;
    }
    return 0;
}

static inline int8_t ref_scalar_i8_square(int8_t a) {
    return (int8_t)(a * a);
}

#define REF_I8_UNARY_FROM_F32(name)                         \
    static inline int8_t ref_scalar_i8_##name(int8_t a) {    \
        return ref_scalar_i8_from_f32(ref_scalar_f32_##name( \
            (float)a));                                     \
    }

#define REF_I8_BINARY_FROM_F32(name)                                \
    static inline int8_t ref_scalar_i8_##name(int8_t a, int8_t b) {  \
        return ref_scalar_i8_from_f32(ref_scalar_f32_##name(         \
            (float)a, (float)b));                                   \
    }

REF_I8_UNARY_FROM_F32(acos)
REF_I8_UNARY_FROM_F32(acosh)
REF_I8_UNARY_FROM_F32(angle)
REF_I8_UNARY_FROM_F32(asin)
REF_I8_UNARY_FROM_F32(asinh)
REF_I8_UNARY_FROM_F32(atan)
REF_I8_UNARY_FROM_F32(atanh)
REF_I8_UNARY_FROM_F32(cbrt)
REF_I8_UNARY_FROM_F32(cos)
REF_I8_UNARY_FROM_F32(cosh)
REF_I8_UNARY_FROM_F32(deg2rad)
REF_I8_UNARY_FROM_F32(digamma)
REF_I8_UNARY_FROM_F32(erf)
REF_I8_UNARY_FROM_F32(erfc)
REF_I8_UNARY_FROM_F32(erfinv)
REF_I8_UNARY_FROM_F32(exp)
REF_I8_UNARY_FROM_F32(exp2)
REF_I8_UNARY_FROM_F32(expm1)
REF_I8_UNARY_FROM_F32(i0)
REF_I8_UNARY_FROM_F32(lgamma)
REF_I8_UNARY_FROM_F32(log)
REF_I8_UNARY_FROM_F32(log10)
REF_I8_UNARY_FROM_F32(log1p)
REF_I8_UNARY_FROM_F32(log2)
REF_I8_UNARY_FROM_F32(logit)
REF_I8_UNARY_FROM_F32(nan_to_num)
REF_I8_UNARY_FROM_F32(rad2deg)
REF_I8_UNARY_FROM_F32(rsqrt)
REF_I8_UNARY_FROM_F32(sigmoid)
REF_I8_UNARY_FROM_F32(silu)
REF_I8_UNARY_FROM_F32(sin)
REF_I8_UNARY_FROM_F32(sinc)
REF_I8_UNARY_FROM_F32(sinh)
REF_I8_UNARY_FROM_F32(sqrt)
REF_I8_UNARY_FROM_F32(tan)
REF_I8_UNARY_FROM_F32(tanh)

REF_I8_BINARY_FROM_F32(atan2)
REF_I8_BINARY_FROM_F32(heaviside)
REF_I8_BINARY_FROM_F32(hypot)
REF_I8_BINARY_FROM_F32(ldexp)
REF_I8_BINARY_FROM_F32(logaddexp)
REF_I8_BINARY_FROM_F32(nextafter)
REF_I8_BINARY_FROM_F32(pow)
REF_I8_BINARY_FROM_F32(xlogy)

#undef REF_I8_UNARY_FROM_F32
#undef REF_I8_BINARY_FROM_F32

#endif
