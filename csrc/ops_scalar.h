#ifndef REF_BACKEND_OPS_SCALAR_H
#define REF_BACKEND_OPS_SCALAR_H

#include <math.h>

static inline float ref_scalar_abs(float a) {
    return fabsf(a);
}

static inline float ref_scalar_add(float a, float b) {
    return a + b;
}

static inline float ref_scalar_sub(float a, float b) {
    return a - b;
}

static inline float ref_scalar_mul(float a, float b) {
    return a * b;
}

static inline float ref_scalar_div(float a, float b) {
    return a / b;
}

static inline float ref_scalar_maximum(float a, float b) {
    return fmaxf(a, b);
}

static inline float ref_scalar_minimum(float a, float b) {
    return fminf(a, b);
}

static inline float ref_scalar_neg(float a) {
    return -a;
}

static inline float ref_scalar_reciprocal(float a) {
    return 1.0f / a;
}

static inline float ref_scalar_relu(float a) {
    return a > 0.0f ? a : 0.0f;
}

static inline float ref_scalar_ceil(float a) {
    return ceilf(a);
}

static inline float ref_scalar_floor(float a) {
    return floorf(a);
}

static inline float ref_scalar_sin(float a) {
    return sinf(a);
}

static inline float ref_scalar_cos(float a) {
    return cosf(a);
}

static inline float ref_scalar_sqrt(float a) {
    return sqrtf(a);
}

static inline float ref_scalar_exp(float a) {
    return expf(a);
}

static inline float ref_scalar_tanh(float a) {
    return tanhf(a);
}

static inline float ref_scalar_log(float a) {
    return logf(a);
}

#endif
