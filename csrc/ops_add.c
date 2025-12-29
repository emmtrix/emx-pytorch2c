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

static int64_t numel(const RefTensorView *tensor) {
    if (tensor->ndim <= 0) {
        return 1;
    }
    int64_t total = 1;
    for (int32_t i = 0; i < tensor->ndim; ++i) {
        total *= tensor->sizes[i];
    }
    return total;
}

static int check_same_shape(const RefTensorView *a, const RefTensorView *b) {
    if (a->ndim != b->ndim) {
        return 0;
    }
    for (int32_t i = 0; i < a->ndim; ++i) {
        if (a->sizes[i] != b->sizes[i]) {
            return 0;
        }
    }
    return 1;
}

int ref_run_add(const RefOpCall *call, char *err_msg, size_t err_cap) {
    if (call->n_inputs != 2 || call->n_outputs != 1) {
        write_error(err_msg, err_cap, "add expects exactly 2 inputs and 1 output");
        return 1;
    }
    RefTensorView *a = &call->inputs[0];
    RefTensorView *b = &call->inputs[1];
    RefTensorView *out = &call->outputs[0];

    if (a->dtype != REF_F32 || b->dtype != REF_F32 || out->dtype != REF_F32) {
        write_error(err_msg, err_cap, "add supports only REF_F32 dtype");
        return 2;
    }

    if (!check_same_shape(a, b) || !check_same_shape(a, out)) {
        write_error(err_msg, err_cap, "add requires inputs and output to have identical shapes");
        return 3;
    }

    if (!is_contiguous(a) || !is_contiguous(b) || !is_contiguous(out)) {
        write_error(err_msg, err_cap, "add requires contiguous tensors");
        return 4;
    }

    int64_t total = numel(a);
    float *a_data = (float *)a->data;
    float *b_data = (float *)b->data;
    float *out_data = (float *)out->data;
    for (int64_t i = 0; i < total; ++i) {
        out_data[i] = a_data[i] + b_data[i];
    }
    return 0;
}
