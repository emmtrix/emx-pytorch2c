#ifndef REF_BACKEND_OPS_SCALAR_I32_H
#define REF_BACKEND_OPS_SCALAR_I32_H

#include <limits.h>
#include <stdint.h>

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

#endif
