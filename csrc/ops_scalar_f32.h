#ifndef REF_BACKEND_OPS_SCALAR_F32_H
#define REF_BACKEND_OPS_SCALAR_F32_H

#include <float.h>
#include <math.h>

#ifndef REF_PI_F
#define REF_PI_F 3.14159265358979323846f
#endif

static inline float ref_scalar_f32_abs(float a) {
    return fabsf(a);
}

static inline float ref_scalar_f32_add(float a, float b) {
    return a + b;
}

static inline float ref_scalar_f32_sub(float a, float b) {
    return a - b;
}

static inline float ref_scalar_f32_mul(float a, float b) {
    return a * b;
}

static inline float ref_scalar_f32_div(float a, float b) {
    return a / b;
}

static inline float ref_scalar_f32_maximum(float a, float b) {
    return fmaxf(a, b);
}

static inline float ref_scalar_f32_minimum(float a, float b) {
    return fminf(a, b);
}

static inline float ref_scalar_f32_le(float a, float b) {
    return a <= b ? 1.0f : 0.0f;
}

static inline float ref_scalar_f32_lt(float a, float b) {
    return a < b ? 1.0f : 0.0f;
}

static inline float ref_scalar_f32_ge(float a, float b) {
    return a >= b ? 1.0f : 0.0f;
}

static inline float ref_scalar_f32_gt(float a, float b) {
    return a > b ? 1.0f : 0.0f;
}

static inline float ref_scalar_f32_eq(float a, float b) {
    return a == b ? 1.0f : 0.0f;
}

static inline float ref_scalar_f32_ne(float a, float b) {
    return a != b ? 1.0f : 0.0f;
}

static inline float ref_scalar_f32_logical_or(float a, float b) {
    return (a != 0.0f || b != 0.0f) ? 1.0f : 0.0f;
}

static inline float ref_scalar_f32_logical_and(float a, float b) {
    return (a != 0.0f && b != 0.0f) ? 1.0f : 0.0f;
}

static inline float ref_scalar_f32_logical_xor(float a, float b) {
    return ((a != 0.0f) != (b != 0.0f)) ? 1.0f : 0.0f;
}

static inline float ref_scalar_f32_logical_not(float a) {
    return a == 0.0f ? 1.0f : 0.0f;
}

static inline float ref_scalar_f32_fmax(float a, float b) {
    return fmaxf(a, b);
}

static inline float ref_scalar_f32_fmin(float a, float b) {
    return fminf(a, b);
}

static inline float ref_scalar_f32_copysign(float a, float b) {
    return copysignf(a, b);
}

static inline float ref_scalar_f32_hypot(float a, float b) {
    return hypotf(a, b);
}

static inline float ref_scalar_f32_atan2(float a, float b) {
    return atan2f(a, b);
}

static inline float ref_scalar_f32_pow(float a, float b) {
    return powf(a, b);
}

static inline float ref_scalar_f32_fmod(float a, float b) {
    return fmodf(a, b);
}

static inline float ref_scalar_f32_remainder(float a, float b) {
    if (isnan(a) || isnan(b)) {
        return NAN;
    }
    if (b == 0.0f) {
        return NAN;
    }
    float mod = fmodf(a, b);
    if (mod == 0.0f) {
        return mod;
    }
    if ((mod < 0.0f) != (b < 0.0f)) {
        mod += b;
    }
    return mod;
}

static inline float ref_scalar_f32_floor_divide(float a, float b) {
    return floorf(a / b);
}

static inline float ref_scalar_f32_logaddexp(float a, float b) {
    if (isnan(a) || isnan(b)) {
        return NAN;
    }
    float max_val = fmaxf(a, b);
    float min_val = fminf(a, b);
    if (max_val == -INFINITY) {
        return -INFINITY;
    }
    return max_val + log1pf(expf(min_val - max_val));
}

static inline float ref_scalar_f32_nextafter(float a, float b) {
    return nextafterf(a, b);
}

static inline float ref_scalar_f32_xlogy(float a, float b) {
    if (isnan(a) || isnan(b)) {
        return NAN;
    }
    if (a == 0.0f) {
        return 0.0f;
    }
    return a * logf(b);
}

static inline float ref_scalar_f32_heaviside(float a, float b) {
    if (a > 0.0f) {
        return 1.0f;
    }
    if (a == 0.0f) {
        return b;
    }
    return 0.0f;
}

static inline float ref_scalar_f32_ldexp(float a, float b) {
    return a * exp2f(b);
}

static inline float ref_scalar_f32_clamp_min(float a, float b) {
    return fmaxf(a, b);
}

static inline float ref_scalar_f32_clamp_max(float a, float b) {
    return fminf(a, b);
}

static inline float ref_scalar_f32_neg(float a) {
    return -a;
}

static inline float ref_scalar_f32_reciprocal(float a) {
    return 1.0f / a;
}

static inline float ref_scalar_f32_relu(float a) {
    return a > 0.0f ? a : 0.0f;
}

static inline float ref_scalar_f32_ceil(float a) {
    return ceilf(a);
}

static inline float ref_scalar_f32_floor(float a) {
    return floorf(a);
}

static inline float ref_scalar_f32_sin(float a) {
    return sinf(a);
}

static inline float ref_scalar_f32_cos(float a) {
    return cosf(a);
}

static inline float ref_scalar_f32_sqrt(float a) {
    return sqrtf(a);
}

static inline float ref_scalar_f32_cbrt(float a) {
    return cbrtf(a);
}

static inline float ref_scalar_f32_exp(float a) {
    return expf(a);
}

static inline float ref_scalar_f32_tanh(float a) {
    return tanhf(a);
}

static inline float ref_scalar_f32_log(float a) {
    return logf(a);
}

static inline float ref_scalar_f32_acos(float a) {
    return acosf(a);
}

static inline float ref_scalar_f32_acosh(float a) {
    return acoshf(a);
}

static inline float ref_scalar_f32_asin(float a) {
    return asinf(a);
}

static inline float ref_scalar_f32_asinh(float a) {
    return asinhf(a);
}

static inline float ref_scalar_f32_atan(float a) {
    return atanf(a);
}

static inline float ref_scalar_f32_atanh(float a) {
    return atanhf(a);
}

static inline float ref_scalar_f32_cosh(float a) {
    return coshf(a);
}

static inline float ref_scalar_f32_sinh(float a) {
    return sinhf(a);
}

static inline float ref_scalar_f32_tan(float a) {
    return tanf(a);
}

static inline float ref_scalar_f32_erf(float a) {
    return erff(a);
}

static inline float ref_scalar_f32_erfc(float a) {
    return erfcf(a);
}

static inline float ref_scalar_f32_expm1(float a) {
    return expm1f(a);
}

static inline float ref_scalar_f32_log1p(float a) {
    return log1pf(a);
}

static inline float ref_scalar_f32_log2(float a) {
    return log2f(a);
}

static inline float ref_scalar_f32_log10(float a) {
    return log10f(a);
}

static inline float ref_scalar_f32_isfinite(float a) {
    return isfinite(a) ? 1.0f : 0.0f;
}

static inline float ref_scalar_f32_rsqrt(float a) {
    return 1.0f / sqrtf(a);
}

static inline float ref_scalar_f32_sigmoid(float a) {
    return 1.0f / (1.0f + expf(-a));
}

static inline float ref_scalar_f32_silu(float a) {
    return a / (1.0f + expf(-a));
}

static inline float ref_scalar_f32_mish(float a) {
    if (a > 20.0f) {
        return a;
    }
    if (a < -20.0f) {
        float exp_a = expf(a);
        return a * exp_a;
    }
    float softplus = log1pf(expf(a));
    return a * tanhf(softplus);
  }
  
static inline float ref_scalar_f32_hardswish(float a) {
    float shifted = a + 3.0f;
    float clamped = fminf(6.0f, fmaxf(0.0f, shifted));
    return a * clamped / 6.0f;
}

static inline float ref_scalar_f32_sign(float a) {
    if (isnan(a)) {
        return a;
    }
    if (a > 0.0f) {
        return 1.0f;
    }
    if (a < 0.0f) {
        return -1.0f;
    }
    return 0.0f;
}

static inline float ref_scalar_f32_round(float a) {
    return roundf(a);
}

static inline float ref_scalar_f32_trunc(float a) {
    return truncf(a);
}

static inline float ref_scalar_f32_angle(float a) {
    if (isnan(a)) {
        return a;
    }
    return a < 0.0f ? REF_PI_F : 0.0f;
}

static inline float ref_scalar_f32_conj(float a) {
    return a;
}

static inline float ref_scalar_f32_conj_physical(float a) {
    return a;
}

static inline float ref_scalar_f32_deg2rad(float a) {
    return a * (REF_PI_F / 180.0f);
}

static inline float ref_scalar_f32_digamma(float x) {
    if (isnan(x) || isinf(x)) {
        return x;
    }
    if (x <= 0.0f) {
        float frac = x - floorf(x);
        if (frac == 0.0f) {
            return NAN;
        }
        return ref_scalar_f32_digamma(1.0f - x) - REF_PI_F / tanf(REF_PI_F * x);
    }
    float result = 0.0f;
    while (x < 6.0f) {
        result -= 1.0f / x;
        x += 1.0f;
    }
    float inv = 1.0f / x;
    float inv2 = inv * inv;
    result += logf(x) - 0.5f * inv
        - inv2 * (1.0f / 12.0f - inv2 * (1.0f / 120.0f - inv2 * (1.0f / 252.0f)));
    return result;
}

static inline float ref_scalar_f32_erfinv(float x) {
    if (isnan(x)) {
        return x;
    }
    if (x <= -1.0f) {
        return x == -1.0f ? -INFINITY : NAN;
    }
    if (x >= 1.0f) {
        return x == 1.0f ? INFINITY : NAN;
    }
    if (x == 0.0f) {
        return 0.0f;
    }
    float a = 0.147f;
    float ln = logf(1.0f - x * x);
    float term = 2.0f / (REF_PI_F * a) + ln / 2.0f;
    float inner = term * term - ln / a;
    float approx = sqrtf(fmaxf(0.0f, sqrtf(inner) - term));
    if (x < 0.0f) {
        approx = -approx;
    }
    for (int i = 0; i < 2; ++i) {
        float err = erff(approx) - x;
        float deriv = 2.0f / sqrtf(REF_PI_F) * expf(-approx * approx);
        approx -= err / deriv;
    }
    return approx;
}

static inline float ref_scalar_f32_exp2(float a) {
    return exp2f(a);
}

static inline float ref_scalar_f32_frac(float a) {
    return a - floorf(a);
}

static inline float ref_scalar_f32_i0(float x) {
    float ax = fabsf(x);
    if (ax < 3.75f) {
        float y = x / 3.75f;
        y *= y;
        return 1.0f + y * (3.5156229f + y * (3.0899424f + y * (1.2067492f
            + y * (0.2659732f + y * (0.0360768f + y * 0.0045813f)))));
    }
    float y = 3.75f / ax;
    return (expf(ax) / sqrtf(ax)) * (0.39894228f + y * (0.01328592f
        + y * (0.00225319f + y * (-0.00157565f + y * (0.00916281f
        + y * (-0.02057706f + y * (0.02635537f
        + y * (-0.01647633f + y * 0.00392377f))))))));
}

static inline float ref_scalar_f32_lgamma(float a) {
    return lgammaf(a);
}

static inline float ref_scalar_f32_logit(float a) {
    if (isnan(a)) {
        return a;
    }
    if (a == 0.0f) {
        return -INFINITY;
    }
    if (a == 1.0f) {
        return INFINITY;
    }
    if (a < 0.0f || a > 1.0f) {
        return NAN;
    }
    return logf(a / (1.0f - a));
}

static inline float ref_scalar_f32_isinf(float a) {
    return isinf(a) ? 1.0f : 0.0f;
}

static inline float ref_scalar_f32_nan_to_num(float a) {
    if (isnan(a)) {
        return 0.0f;
    }
    if (isinf(a)) {
        return signbit(a) ? -FLT_MAX : FLT_MAX;
    }
    return a;
}

static inline float ref_scalar_f32_positive(float a) {
    return a;
}

static inline float ref_scalar_f32_rad2deg(float a) {
    return a * (180.0f / REF_PI_F);
}

static inline float ref_scalar_f32_real(float a) {
    return a;
}

static inline float ref_scalar_f32_sgn(float a) {
    if (isnan(a)) {
        return 0.0f;
    }
    if (a > 0.0f) {
        return 1.0f;
    }
    if (a < 0.0f) {
        return -1.0f;
    }
    return 0.0f;
}

static inline float ref_scalar_f32_sinc(float a) {
    if (a == 0.0f) {
        return 1.0f;
    }
    float x = REF_PI_F * a;
    return sinf(x) / x;
}

static inline float ref_scalar_f32_square(float a) {
    return a * a;
}

#endif
