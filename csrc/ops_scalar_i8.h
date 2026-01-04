#ifndef REF_BACKEND_OPS_SCALAR_I8_H
#define REF_BACKEND_OPS_SCALAR_I8_H

#include <limits.h>
#include <stdint.h>

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

#endif
