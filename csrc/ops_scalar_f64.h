#ifndef REF_BACKEND_OPS_SCALAR_F64_H
#define REF_BACKEND_OPS_SCALAR_F64_H

#include <float.h>
#include <math.h>

#ifndef REF_PI_D
#define REF_PI_D 3.14159265358979323846
#endif

static inline double ref_scalar_f64_abs(double a) {
    return fabs(a);
}

static inline double ref_scalar_f64_absolute(double a) {
    return ref_scalar_f64_abs(a);
}

static inline double ref_scalar_f64_add(double a, double b) {
    return a + b;
}

static inline double ref_scalar_f64_sub(double a, double b) {
    return a - b;
}

static inline double ref_scalar_f64_mul(double a, double b) {
    return a * b;
}

static inline double ref_scalar_f64_div(double a, double b) {
    return a / b;
}

static inline double ref_scalar_f64_maximum(double a, double b) {
    return fmax(a, b);
}

static inline double ref_scalar_f64_minimum(double a, double b) {
    return fmin(a, b);
}

static inline double ref_scalar_f64_le(double a, double b) {
    return a <= b ? 1.0 : 0.0;
}

static inline double ref_scalar_f64_lt(double a, double b) {
    return a < b ? 1.0 : 0.0;
}

static inline double ref_scalar_f64_ge(double a, double b) {
    return a >= b ? 1.0 : 0.0;
}

static inline double ref_scalar_f64_gt(double a, double b) {
    return a > b ? 1.0 : 0.0;
}

static inline double ref_scalar_f64_eq(double a, double b) {
    return a == b ? 1.0 : 0.0;
}

static inline double ref_scalar_f64_ne(double a, double b) {
    return a != b ? 1.0 : 0.0;
}

static inline double ref_scalar_f64_logical_or(double a, double b) {
    return (a != 0.0 || b != 0.0) ? 1.0 : 0.0;
}

static inline double ref_scalar_f64_logical_and(double a, double b) {
    return (a != 0.0 && b != 0.0) ? 1.0 : 0.0;
}

static inline double ref_scalar_f64_logical_xor(double a, double b) {
    return ((a != 0.0) != (b != 0.0)) ? 1.0 : 0.0;
}

static inline double ref_scalar_f64_logical_not(double a) {
    return a == 0.0 ? 1.0 : 0.0;
}

static inline double ref_scalar_f64_fmax(double a, double b) {
    return fmax(a, b);
}

static inline double ref_scalar_f64_fmin(double a, double b) {
    return fmin(a, b);
}

static inline double ref_scalar_f64_copysign(double a, double b) {
    return copysign(a, b);
}

static inline double ref_scalar_f64_hypot(double a, double b) {
    return hypot(a, b);
}

static inline double ref_scalar_f64_atan2(double a, double b) {
    return atan2(a, b);
}

static inline double ref_scalar_f64_pow(double a, double b) {
    return pow(a, b);
}

static inline double ref_scalar_f64_fmod(double a, double b) {
    return fmod(a, b);
}

static inline double ref_scalar_f64_remainder(double a, double b) {
    if (isnan(a) || isnan(b)) {
        return NAN;
    }
    if (b == 0.0) {
        return NAN;
    }
    double mod = fmod(a, b);
    if (mod == 0.0) {
        return mod;
    }
    if ((mod < 0.0) != (b < 0.0)) {
        mod += b;
    }
    return mod;
}

static inline double ref_scalar_f64_floor_divide(double a, double b) {
    return floor(a / b);
}

static inline double ref_scalar_f64_logaddexp(double a, double b) {
    if (isnan(a) || isnan(b)) {
        return NAN;
    }
    double max_val = fmax(a, b);
    double min_val = fmin(a, b);
    if (max_val == -INFINITY) {
        return -INFINITY;
    }
    return max_val + log1p(exp(min_val - max_val));
}

static inline double ref_scalar_f64_logaddexp2(double a, double b) {
    if (isnan(a) || isnan(b)) {
        return NAN;
    }
    double max_val = fmax(a, b);
    double min_val = fmin(a, b);
    if (max_val == -INFINITY) {
        return -INFINITY;
    }
    return max_val + log2(1.0 + exp2(min_val - max_val));
}

static inline double ref_scalar_f64_nextafter(double a, double b) {
    return nextafter(a, b);
}

static inline double ref_scalar_f64_xlogy(double a, double b) {
    if (isnan(a) || isnan(b)) {
        return NAN;
    }
    if (a == 0.0) {
        return 0.0;
    }
    return a * log(b);
}

static inline double ref_scalar_f64_heaviside(double a, double b) {
    if (a > 0.0) {
        return 1.0;
    }
    if (a == 0.0) {
        return b;
    }
    return 0.0;
}

static inline double ref_scalar_f64_ldexp(double a, double b) {
    return a * exp2(b);
}

static inline double ref_scalar_f64_clamp_min(double a, double b) {
    return fmax(a, b);
}

static inline double ref_scalar_f64_clamp_max(double a, double b) {
    return fmin(a, b);
}

static inline double ref_scalar_f64_neg(double a) {
    return -a;
}

static inline double ref_scalar_f64_reciprocal(double a) {
    return 1.0 / a;
}

static inline double ref_scalar_f64_relu(double a) {
    return a > 0.0 ? a : 0.0;
}

static inline double ref_scalar_f64_ceil(double a) {
    return ceil(a);
}

static inline double ref_scalar_f64_floor(double a) {
    return floor(a);
}

static inline double ref_scalar_f64_sin(double a) {
    return sin(a);
}

static inline double ref_scalar_f64_cos(double a) {
    return cos(a);
}

static inline double ref_scalar_f64_sqrt(double a) {
    return sqrt(a);
}

static inline double ref_scalar_f64_cbrt(double a) {
    return cbrtf(a);
}

static inline double ref_scalar_f64_exp(double a) {
    return exp(a);
}

static inline double ref_scalar_f64_tanh(double a) {
    return tanh(a);
}

static inline double ref_scalar_f64_log(double a) {
    return log(a);
}

static inline double ref_scalar_f64_acos(double a) {
    return acos(a);
}

static inline double ref_scalar_f64_arccos(double a) {
    return ref_scalar_f64_acos(a);
}

static inline double ref_scalar_f64_acosh(double a) {
    return acosh(a);
}

static inline double ref_scalar_f64_asin(double a) {
    return asin(a);
}

static inline double ref_scalar_f64_arcsin(double a) {
    return ref_scalar_f64_asin(a);
}

static inline double ref_scalar_f64_asinh(double a) {
    return asinh(a);
}

static inline double ref_scalar_f64_arcsinh(double a) {
    return ref_scalar_f64_asinh(a);
}

static inline double ref_scalar_f64_atan(double a) {
    return atan(a);
}

static inline double ref_scalar_f64_arctan(double a) {
    return ref_scalar_f64_atan(a);
}

static inline double ref_scalar_f64_atanh(double a) {
    return atanh(a);
}

static inline double ref_scalar_f64_cosh(double a) {
    return cosh(a);
}

static inline double ref_scalar_f64_sinh(double a) {
    return sinh(a);
}

static inline double ref_scalar_f64_tan(double a) {
    return tan(a);
}

static inline double ref_scalar_f64_erf(double a) {
    return erf(a);
}

static inline double ref_scalar_f64_erfc(double a) {
    return erfc(a);
}

static inline double ref_scalar_f64_expm1(double a) {
    return expm1(a);
}

static inline double ref_scalar_f64_log1p(double a) {
    return log1p(a);
}

static inline double ref_scalar_f64_log2(double a) {
    return log2(a);
}

static inline double ref_scalar_f64_log10(double a) {
    return log10(a);
}

static inline double ref_scalar_f64_isfinite(double a) {
    return isfinite(a) ? 1.0 : 0.0;
}

static inline double ref_scalar_f64_rsqrt(double a) {
    return 1.0 / sqrt(a);
}

static inline double ref_scalar_f64_sigmoid(double a) {
    return 1.0 / (1.0 + exp(-a));
}

static inline double ref_scalar_f64_log_sigmoid(double a) {
    if (a >= 0.0) {
        return -log1p(exp(-a));
    }
    return a - log1p(exp(a));
}

static inline double ref_scalar_f64_gelu(double a) {
    const double inv_sqrt2 = 0.7071067811865475;
    return 0.5 * a * (1.0 + erf(a * inv_sqrt2));
}

static inline double ref_scalar_f64_elu(double a) {
    const double alpha = 1.0;
    const double scale = 1.0;
    const double input_scale = 1.0;
    if (a > 0.0) {
        return scale * a;
    }
    return scale * alpha * (exp(input_scale * a) - 1.0);
}

static inline double ref_scalar_f64_leaky_relu(double a) {
    const double negative_slope = 0.01;
    return a > 0.0 ? a : negative_slope * a;
}

static inline double ref_scalar_f64_softplus(double a) {
    const double beta = 1.0;
    const double threshold = 20.0;
    if (beta * a > threshold) {
        return a;
    }
    return log1p(exp(beta * a)) / beta;
}

static inline double ref_scalar_f64_silu(double a) {
    return a / (1.0 + exp(-a));
}

static inline double ref_scalar_f64_mish(double a) {
    if (a > 20.0) {
        return a;
    }
    if (a < -20.0) {
        double exp_a = exp(a);
        return a * exp_a;
    }
    double softplus = log1p(exp(a));
    return a * tanh(softplus);
  }
  
static inline double ref_scalar_f64_selu(double a) {
    const double alpha = 1.6732632423543772848170429916717;
    const double scale = 1.0507009873554804934193349852946;
    if (a > 0.0) {
        return scale * a;
    }
    return scale * alpha * (exp(a) - 1.0);
}

static inline double ref_scalar_f64_relu6(double a) {
    return fmin(6.0, fmax(0.0, a));
}

static inline double ref_scalar_f64_hardsigmoid(double a) {
    double shifted = a + 3.0;
    double clamped = fmin(6.0, fmax(0.0, shifted));
    return clamped / 6.0;
}

static inline double ref_scalar_f64_hardswish(double a) {
    double shifted = a + 3.0;
    double clamped = fmin(6.0, fmax(0.0, shifted));
    return a * clamped / 6.0;
}

static inline double ref_scalar_f64_sign(double a) {
    if (isnan(a)) {
        return a;
    }
    if (a > 0.0) {
        return 1.0;
    }
    if (a < 0.0) {
        return -1.0;
    }
    return 0.0;
}

static inline double ref_scalar_f64_round(double a) {
    return round(a);
}

static inline double ref_scalar_f64_trunc(double a) {
    return trunc(a);
}

static inline double ref_scalar_f64_angle(double a) {
    if (isnan(a)) {
        return a;
    }
    return a < 0.0 ? REF_PI_D : 0.0;
}

static inline double ref_scalar_f64_conj(double a) {
    return a;
}

static inline double ref_scalar_f64_conj_physical(double a) {
    return a;
}

static inline double ref_scalar_f64_deg2rad(double a) {
    return a * (REF_PI_D / 180.0);
}

static inline double ref_scalar_f64_digamma(double x) {
    if (isnan(x) || isinf(x)) {
        return x;
    }
    if (x <= 0.0) {
        double frac = x - floor(x);
        if (frac == 0.0) {
            return NAN;
        }
        return ref_scalar_f64_digamma(1.0 - x) - REF_PI_D / tan(REF_PI_D * x);
    }
    double result = 0.0;
    while (x < 10.0) {
        result -= 1.0 / x;
        x += 1.0;
    }
    double inv = 1.0 / x;
    double inv2 = inv * inv;
    result += log(x) - 0.5 * inv
        - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0
        - inv2 * (1.0 / 252.0 - inv2 * (1.0 / 240.0
        - inv2 * (1.0 / 132.0 - inv2 * (691.0 / 32760.0))))));
    return result;
}

static inline double ref_scalar_f64_erfinv(double x) {
    if (isnan(x)) {
        return x;
    }
    if (x <= -1.0) {
        return x == -1.0 ? -INFINITY : NAN;
    }
    if (x >= 1.0) {
        return x == 1.0 ? INFINITY : NAN;
    }
    if (x == 0.0) {
        return 0.0;
    }
    double a = 0.147;
    double ln = log(1.0 - x * x);
    double term = 2.0 / (REF_PI_D * a) + ln / 2.0;
    double inner = term * term - ln / a;
    double approx = sqrt(fmax(0.0, sqrt(inner) - term));
    if (x < 0.0) {
        approx = -approx;
    }
    for (int i = 0; i < 2; ++i) {
        double err = erf(approx) - x;
        double deriv = 2.0 / sqrt(REF_PI_D) * exp(-approx * approx);
        approx -= err / deriv;
    }
    return approx;
}

static inline double ref_scalar_f64_exp2(double a) {
    return exp2(a);
}

static inline double ref_scalar_f64_frac(double a) {
    return a - trunc(a);
}

static inline double ref_scalar_f64_i0(double x) {
    double ax = fabs(x);
    if (ax < 3.75) {
        double y = x / 3.75;
        y *= y;
        return 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492
            + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))));
    }
    double y = 3.75 / ax;
    return (exp(ax) / sqrt(ax)) * (0.39894228 + y * (0.01328592
        + y * (0.00225319 + y * (-0.00157565 + y * (0.00916281
        + y * (-0.02057706 + y * (0.02635537
        + y * (-0.01647633 + y * 0.00392377))))))));
}

static inline double ref_scalar_f64_lgamma(double a) {
    return lgamma(a);
}

static inline double ref_scalar_f64_logit(double a) {
    if (isnan(a)) {
        return a;
    }
    if (a == 0.0) {
        return -INFINITY;
    }
    if (a == 1.0) {
        return INFINITY;
    }
    if (a < 0.0 || a > 1.0) {
        return NAN;
    }
    return log(a / (1.0 - a));
}

static inline double ref_scalar_f64_isnan(double a) {
    return isnan(a) ? 1.0 : 0.0;
}

static inline double ref_scalar_f64_isinf(double a) {
    return isinf(a) ? 1.0 : 0.0;
}

static inline double ref_scalar_f64_isneginf(double a) {
    return (isinf(a) && signbit(a)) ? 1.0 : 0.0;
}

static inline double ref_scalar_f64_isposin(double a) {
    return (isinf(a) && !signbit(a)) ? 1.0 : 0.0;
}

static inline double ref_scalar_f64_nan_to_num(double a) {
    if (isnan(a)) {
        return 0.0;
    }
    if (isinf(a)) {
        return signbit(a) ? -FLT_MAX : FLT_MAX;
    }
    return a;
}

static inline double ref_scalar_f64_positive(double a) {
    return a;
}

static inline double ref_scalar_f64_rad2deg(double a) {
    return a * (180.0 / REF_PI_D);
}

static inline double ref_scalar_f64_real(double a) {
    return a;
}

static inline double ref_scalar_f64_sgn(double a) {
    if (isnan(a)) {
        return 0.0;
    }
    if (a > 0.0) {
        return 1.0;
    }
    if (a < 0.0) {
        return -1.0;
    }
    return 0.0;
}

static inline double ref_scalar_f64_sinc(double a) {
    if (a == 0.0) {
        return 1.0;
    }
    double x = REF_PI_D * a;
    return sin(x) / x;
}

static inline double ref_scalar_f64_square(double a) {
    return a * a;
}

#endif
