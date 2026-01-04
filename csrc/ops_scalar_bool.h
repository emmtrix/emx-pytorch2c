#ifndef REF_BACKEND_OPS_SCALAR_BOOL_H
#define REF_BACKEND_OPS_SCALAR_BOOL_H

#include <stdbool.h>

#include "ops_scalar_f32.h"

static inline float ref_scalar_bool_to_f32(bool value) {
    return value ? 1.0f : 0.0f;
}

static inline bool ref_scalar_bool_from_f32(float value) {
    return value != 0.0f;
}

#define REF_BOOL_UNARY(name)                                       \
    static inline bool ref_scalar_bool_##name(bool a) {            \
        return ref_scalar_bool_from_f32(                           \
            ref_scalar_f32_##name(ref_scalar_bool_to_f32(a))        \
        );                                                         \
    }

#define REF_BOOL_BINARY(name)                                      \
    static inline bool ref_scalar_bool_##name(bool a, bool b) {     \
        return ref_scalar_bool_from_f32(                           \
            ref_scalar_f32_##name(                                 \
                ref_scalar_bool_to_f32(a),                         \
                ref_scalar_bool_to_f32(b)                          \
            )                                                      \
        );                                                         \
    }

REF_BOOL_UNARY(abs)
REF_BOOL_BINARY(add)
REF_BOOL_BINARY(sub)
REF_BOOL_BINARY(mul)
REF_BOOL_BINARY(div)
REF_BOOL_BINARY(maximum)
REF_BOOL_BINARY(minimum)
REF_BOOL_BINARY(fmax)
REF_BOOL_BINARY(fmin)
REF_BOOL_BINARY(copysign)
REF_BOOL_BINARY(hypot)
REF_BOOL_BINARY(atan2)
REF_BOOL_BINARY(pow)
REF_BOOL_BINARY(fmod)
REF_BOOL_BINARY(remainder)
REF_BOOL_BINARY(floor_divide)
REF_BOOL_BINARY(logaddexp)
REF_BOOL_BINARY(nextafter)
REF_BOOL_BINARY(xlogy)
REF_BOOL_BINARY(heaviside)
REF_BOOL_BINARY(ldexp)
REF_BOOL_BINARY(clamp_min)
REF_BOOL_BINARY(clamp_max)
REF_BOOL_UNARY(neg)
REF_BOOL_UNARY(reciprocal)
REF_BOOL_UNARY(relu)
REF_BOOL_UNARY(ceil)
REF_BOOL_UNARY(floor)
REF_BOOL_UNARY(sin)
REF_BOOL_UNARY(cos)
REF_BOOL_UNARY(sqrt)
REF_BOOL_UNARY(cbrt)
REF_BOOL_UNARY(exp)
REF_BOOL_UNARY(tanh)
REF_BOOL_UNARY(log)
REF_BOOL_UNARY(acos)
REF_BOOL_UNARY(acosh)
REF_BOOL_UNARY(asin)
REF_BOOL_UNARY(asinh)
REF_BOOL_UNARY(atan)
REF_BOOL_UNARY(atanh)
REF_BOOL_UNARY(cosh)
REF_BOOL_UNARY(sinh)
REF_BOOL_UNARY(tan)
REF_BOOL_UNARY(erf)
REF_BOOL_UNARY(erfc)
REF_BOOL_UNARY(expm1)
REF_BOOL_UNARY(log1p)
REF_BOOL_UNARY(log2)
REF_BOOL_UNARY(log10)
REF_BOOL_UNARY(rsqrt)
REF_BOOL_UNARY(sigmoid)
REF_BOOL_UNARY(silu)
REF_BOOL_UNARY(mish)
REF_BOOL_UNARY(sign)
REF_BOOL_UNARY(round)
REF_BOOL_UNARY(trunc)
REF_BOOL_UNARY(angle)
REF_BOOL_UNARY(conj)
REF_BOOL_UNARY(conj_physical)
REF_BOOL_UNARY(deg2rad)
REF_BOOL_UNARY(digamma)
REF_BOOL_UNARY(erfinv)
REF_BOOL_UNARY(exp2)
REF_BOOL_UNARY(frac)
REF_BOOL_UNARY(i0)
REF_BOOL_UNARY(lgamma)
REF_BOOL_UNARY(logit)
REF_BOOL_UNARY(nan_to_num)
REF_BOOL_UNARY(positive)
REF_BOOL_UNARY(rad2deg)
REF_BOOL_UNARY(real)
REF_BOOL_UNARY(sgn)
REF_BOOL_UNARY(sinc)
REF_BOOL_UNARY(square)

#undef REF_BOOL_UNARY
#undef REF_BOOL_BINARY

#endif
