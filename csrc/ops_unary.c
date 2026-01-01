#include "ops_unary.h"
#include "ops_utils.h"

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

static void unary_f32(
    const float *a_data,
    float *out_data,
    int64_t total,
    RefUnaryF32Op op
) {
    for (int64_t i = 0; i < total; ++i) {
        out_data[i] = op(a_data[i]);
    }
}

static void unary_f32_strided(
    const float *a_data,
    float *out_data,
    const int64_t sizes[8],
    const int64_t a_strides[8],
    const int64_t out_strides[8],
    RefUnaryF32Op op
) {
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
                                    int64_t out_offset = i0 * out_strides[0]
                                        + i1 * out_strides[1]
                                        + i2 * out_strides[2]
                                        + i3 * out_strides[3]
                                        + i4 * out_strides[4]
                                        + i5 * out_strides[5]
                                        + i6 * out_strides[6]
                                        + i7 * out_strides[7];
                                    out_data[out_offset] = op(a_data[a_offset]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

static int check_same_shape(const RefTensorView *a, const RefTensorView *out) {
    if (a->ndim != out->ndim) {
        return 0;
    }
    for (int32_t i = 0; i < a->ndim; ++i) {
        if (a->sizes[i] != out->sizes[i]) {
            return 0;
        }
    }
    return 1;
}

int ref_run_unary_f32(
    const RefOpCall *call,
    char *err_msg,
    size_t err_cap,
    RefUnaryF32Op op,
    const char *op_name
) {
    if (call->n_inputs != 1 || call->n_outputs != 1) {
        write_errorf(err_msg, err_cap, op_name, "expects exactly 1 input and 1 output");
        return 1;
    }
    RefTensorView *a = &call->inputs[0];
    RefTensorView *out = &call->outputs[0];

    if (a->dtype != REF_F32 || out->dtype != REF_F32) {
        write_errorf(err_msg, err_cap, op_name, "supports only REF_F32 dtype");
        return 2;
    }
    if (!check_same_shape(a, out)) {
        write_errorf(err_msg, err_cap, op_name, "requires input and output to have identical shapes");
        return 3;
    }
    if (a->ndim > 8) {
        write_errorf(err_msg, err_cap, op_name, "supports at most 8 dimensions");
        return 4;
    }

    int64_t total = numel(a);
    float *a_data = (float *)a->data;
    float *out_data = (float *)out->data;
    if (is_contiguous(a) && is_contiguous(out)) {
        unary_f32(a_data, out_data, total, op);
        return 0;
    }
    int64_t sizes_8[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t a_strides_8[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t out_strides_8[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int32_t offset = 8 - a->ndim;
    for (int32_t d = 0; d < a->ndim; ++d) {
        int32_t dst = offset + d;
        sizes_8[dst] = a->sizes[d];
        a_strides_8[dst] = a->strides[d];
        out_strides_8[dst] = out->strides[d];
    }
    unary_f32_strided(
        a_data,
        out_data,
        sizes_8,
        a_strides_8,
        out_strides_8,
        op
    );
    return 0;
}
