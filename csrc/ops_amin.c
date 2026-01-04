#include "ops_utils.h"

#include <math.h>
#include <stdio.h>

static void write_errorf(
    char *err_msg,
    size_t err_cap,
    const char *op_name,
    const char *suffix
) {
    char msg[128];
    snprintf(msg, sizeof(msg), "%s %s", op_name, suffix);
    write_error(err_msg, err_cap, msg);
}

static int update_min(float *acc, float value, int *initialized) {
    if (!*initialized) {
        *acc = value;
        *initialized = 1;
        return isnan(value);
    }
    if (isnan(value)) {
        *acc = value;
        return 1;
    }
    if (value < *acc) {
        *acc = value;
    }
    return 0;
}

static void reduce_f32(
    const float *a_data,
    int64_t total,
    float *out_data
) {
    float acc = a_data[0];
    if (isnan(acc)) {
        out_data[0] = acc;
        return;
    }
    for (int64_t i = 1; i < total; ++i) {
        float value = a_data[i];
        if (isnan(value)) {
            out_data[0] = value;
            return;
        }
        if (value < acc) {
            acc = value;
        }
    }
    out_data[0] = acc;
}

static void reduce_f32_strided(
    const float *a_data,
    const int64_t sizes[8],
    const int64_t a_strides[8],
    float *out_data
) {
    float acc = 0.0f;
    int initialized = 0;
    for (int64_t i0 = 0; i0 < sizes[0]; ++i0) {
        for (int64_t i1 = 0; i1 < sizes[1]; ++i1) {
            for (int64_t i2 = 0; i2 < sizes[2]; ++i2) {
                for (int64_t i3 = 0; i3 < sizes[3]; ++i3) {
                    for (int64_t i4 = 0; i4 < sizes[4]; ++i4) {
                        for (int64_t i5 = 0; i5 < sizes[5]; ++i5) {
                            for (int64_t i6 = 0; i6 < sizes[6]; ++i6) {
                                for (int64_t i7 = 0; i7 < sizes[7]; ++i7) {
                                    int64_t a_offset = i0 * a_strides[0]
                                        + i1 * a_strides[1]
                                        + i2 * a_strides[2]
                                        + i3 * a_strides[3]
                                        + i4 * a_strides[4]
                                        + i5 * a_strides[5]
                                        + i6 * a_strides[6]
                                        + i7 * a_strides[7];
                                    if (update_min(&acc, a_data[a_offset], &initialized)) {
                                        out_data[0] = acc;
                                        return;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out_data[0] = acc;
}

int ref_run_amin(const RefOpCall *call, char *err_msg, size_t err_cap) {
    if (call->n_inputs != 1 || call->n_outputs != 1) {
        write_errorf(err_msg, err_cap, "amin", "expects exactly 1 input and 1 output");
        return 1;
    }
    RefTensorView *a = &call->inputs[0];
    RefTensorView *out = &call->outputs[0];

    if (a->dtype != REF_F32 || out->dtype != REF_F32) {
        write_errorf(err_msg, err_cap, "amin", "supports only REF_F32 dtype");
        return 2;
    }
    if (out->ndim != 0) {
        write_errorf(err_msg, err_cap, "amin", "requires output to be a scalar tensor");
        return 3;
    }
    if (a->ndim > 8) {
        write_errorf(err_msg, err_cap, "amin", "supports at most 8 dimensions");
        return 4;
    }

    int64_t total = numel(a);
    if (total == 0) {
        write_errorf(
            err_msg,
            err_cap,
            "amin",
            "requires input.numel() > 0 when dim is not specified"
        );
        return 5;
    }

    float *a_data = (float *)a->data;
    float *out_data = (float *)out->data;
    if (is_contiguous(a)) {
        reduce_f32(a_data, total, out_data);
        return 0;
    }

    int64_t sizes_8[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t a_strides_8[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int32_t offset = 8 - a->ndim;
    for (int32_t d = 0; d < a->ndim; ++d) {
        int32_t dst = offset + d;
        sizes_8[dst] = a->sizes[d];
        a_strides_8[dst] = a->strides[d];
    }
    reduce_f32_strided(a_data, sizes_8, a_strides_8, out_data);
    return 0;
}
