#include "ref_backend.h"

#include <stdio.h>
#include <string.h>

static void write_error(char *err_msg, size_t err_cap, const char *msg) {
    if (err_msg == NULL || err_cap == 0) {
        return;
    }
    strncpy(err_msg, msg, err_cap - 1);
    err_msg[err_cap - 1] = '\0';
}

static int is_contiguous(const RefTensorView *tensor) {
    if (tensor->ndim <= 0) {
        return 1;
    }
    int64_t expected = 1;
    for (int32_t i = tensor->ndim - 1; i >= 0; --i) {
        if (tensor->strides[i] != expected) {
            return 0;
        }
        expected *= tensor->sizes[i];
    }
    return 1;
}

static void matmul_f32(const float *a_data, const float *b_data, float *out_data, int64_t m, int64_t k,
                       int64_t n) {
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int64_t kk = 0; kk < k; ++kk) {
                acc += a_data[i * k + kk] * b_data[kk * n + j];
            }
            out_data[i * n + j] = acc;
        }
    }
}

int ref_run_matmul(const RefOpCall *call, char *err_msg, size_t err_cap) {
    if (call->n_inputs != 2 || call->n_outputs != 1) {
        write_error(err_msg, err_cap, "matmul expects exactly 2 inputs and 1 output");
        return 1;
    }
    RefTensorView *a = &call->inputs[0];
    RefTensorView *b = &call->inputs[1];
    RefTensorView *out = &call->outputs[0];

    if (a->dtype != REF_F32 || b->dtype != REF_F32 || out->dtype != REF_F32) {
        write_error(err_msg, err_cap, "matmul supports only REF_F32 dtype");
        return 2;
    }

    if (a->ndim != 2 || b->ndim != 2 || out->ndim != 2) {
        write_error(err_msg, err_cap, "matmul requires 2D inputs and output");
        return 3;
    }

    int64_t m = a->sizes[0];
    int64_t k = a->sizes[1];
    if (b->sizes[0] != k) {
        write_error(err_msg, err_cap, "matmul requires inner dimensions to match");
        return 4;
    }
    int64_t n = b->sizes[1];
    if (out->sizes[0] != m || out->sizes[1] != n) {
        write_error(err_msg, err_cap, "matmul requires output shape (m, n)");
        return 5;
    }

    if (!is_contiguous(a) || !is_contiguous(b) || !is_contiguous(out)) {
        write_error(err_msg, err_cap, "matmul requires contiguous tensors");
        return 6;
    }

    float *a_data = (float *)a->data;
    float *b_data = (float *)b->data;
    float *out_data = (float *)out->data;
    matmul_f32(a_data, b_data, out_data, m, k, n);
    return 0;
}
