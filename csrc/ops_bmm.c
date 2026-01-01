#include "ops_utils.h"

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

static void bmm_f32(const float *a_data, const float *b_data, float *out_data, int64_t batch, int64_t m,
                    int64_t k, int64_t n) {
    int64_t a_batch_stride = m * k;
    int64_t b_batch_stride = k * n;
    int64_t out_batch_stride = m * n;
    for (int64_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
        matmul_f32(
            a_data + batch_idx * a_batch_stride,
            b_data + batch_idx * b_batch_stride,
            out_data + batch_idx * out_batch_stride,
            m,
            k,
            n);
    }
}

int ref_run_bmm(const RefOpCall *call, char *err_msg, size_t err_cap) {
    if (call->n_inputs != 2 || call->n_outputs != 1) {
        write_error(err_msg, err_cap, "bmm expects exactly 2 inputs and 1 output");
        return 1;
    }
    RefTensorView *a = &call->inputs[0];
    RefTensorView *b = &call->inputs[1];
    RefTensorView *out = &call->outputs[0];

    if (a->dtype != REF_F32 || b->dtype != REF_F32 || out->dtype != REF_F32) {
        write_error(err_msg, err_cap, "bmm supports only REF_F32 dtype");
        return 2;
    }

    if (a->ndim != 3 || b->ndim != 3 || out->ndim != 3) {
        write_error(err_msg, err_cap, "bmm requires 3D inputs and output");
        return 3;
    }

    int64_t batch = a->sizes[0];
    int64_t m = a->sizes[1];
    int64_t k = a->sizes[2];
    if (b->sizes[0] != batch) {
        write_error(err_msg, err_cap, "bmm requires batch dimensions to match");
        return 4;
    }
    if (b->sizes[1] != k) {
        write_error(err_msg, err_cap, "bmm requires inner dimensions to match");
        return 5;
    }
    int64_t n = b->sizes[2];
    if (out->sizes[0] != batch || out->sizes[1] != m || out->sizes[2] != n) {
        write_error(err_msg, err_cap, "bmm requires output shape (batch, m, n)");
        return 6;
    }

    if (!is_contiguous(a) || !is_contiguous(b) || !is_contiguous(out)) {
        write_error(err_msg, err_cap, "bmm requires contiguous tensors");
        return 7;
    }

    float *a_data = (float *)a->data;
    float *b_data = (float *)b->data;
    float *out_data = (float *)out->data;
    bmm_f32(a_data, b_data, out_data, batch, m, k, n);
    return 0;
}
