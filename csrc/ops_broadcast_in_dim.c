#include "ops_utils.h"

#include <stdlib.h>

static void broadcast_in_dim_f32(const float *a_data, float *out_data, int32_t out_ndim,
                                 const int64_t *a_sizes, const int64_t *a_strides,
                                 const int64_t *out_sizes, const int32_t *out_to_in,
                                 int64_t total) {
    int64_t out_index[out_ndim];

    for (int64_t linear = 0; linear < total; ++linear) {
        int64_t remaining = linear;
        for (int32_t dim = out_ndim - 1; dim >= 0; --dim) {
            int64_t size = out_sizes[dim];
            out_index[dim] = remaining % size;
            remaining /= size;
        }

        int64_t a_offset = 0;
        for (int32_t out_dim = 0; out_dim < out_ndim; ++out_dim) {
            int32_t in_dim = out_to_in[out_dim];
            if (in_dim == -1) {
                continue;
            }
            int64_t in_size = a_sizes[in_dim];
            int64_t in_index = (in_size == 1) ? 0 : out_index[out_dim];
            a_offset += in_index * a_strides[in_dim];
        }
        out_data[linear] = a_data[a_offset];
    }
}

int ref_run_broadcast_in_dim(const RefOpCall *call, char *err_msg, size_t err_cap) {
    if (call->n_inputs != 1 || call->n_outputs != 1) {
        write_error(err_msg, err_cap, "broadcast_in_dim expects 1 input and 1 output");
        return 1;
    }
    RefTensorView *a = &call->inputs[0];
    RefTensorView *out = &call->outputs[0];

    if (a->dtype != REF_F32 || out->dtype != REF_F32) {
        write_error(err_msg, err_cap, "broadcast_in_dim supports only REF_F32 dtype");
        return 2;
    }

    if (!is_contiguous(a) || !is_contiguous(out)) {
        write_error(err_msg, err_cap, "broadcast_in_dim requires contiguous tensors");
        return 3;
    }

    if (call->params == NULL) {
        write_error(err_msg, err_cap, "broadcast_in_dim missing params");
        return 4;
    }

    RefBroadcastInDimParams *params = (RefBroadcastInDimParams *)call->params;
    if (params->n_dims != a->ndim) {
        write_error(err_msg, err_cap, "broadcast_in_dim params must match input rank");
        return 5;
    }
    if (out->ndim < a->ndim) {
        write_error(err_msg, err_cap, "broadcast_in_dim requires output rank >= input rank");
        return 6;
    }

    int32_t out_ndim = out->ndim;
    int32_t *out_to_in = (int32_t *)malloc(sizeof(int32_t) * out_ndim);
    if (out_to_in == NULL) {
        write_error(err_msg, err_cap, "broadcast_in_dim allocation failed");
        return 7;
    }
    for (int32_t i = 0; i < out_ndim; ++i) {
        out_to_in[i] = -1;
    }

    int32_t last_dim = -1;
    for (int32_t in_dim = 0; in_dim < params->n_dims; ++in_dim) {
        int32_t out_dim = params->broadcast_dimensions[in_dim];
        if (out_dim < 0 || out_dim >= out_ndim) {
            free(out_to_in);
            write_error(err_msg, err_cap, "broadcast_in_dim has out-of-range dimensions");
            return 8;
        }
        if (out_dim <= last_dim) {
            free(out_to_in);
            write_error(err_msg, err_cap,
                        "broadcast_in_dim expects strictly increasing broadcast_dimensions");
            return 9;
        }
        if (out_to_in[out_dim] != -1) {
            free(out_to_in);
            write_error(err_msg, err_cap, "broadcast_in_dim requires unique dimensions");
            return 10;
        }
        out_to_in[out_dim] = in_dim;
        last_dim = out_dim;

        int64_t in_size = a->sizes[in_dim];
        int64_t out_size = out->sizes[out_dim];
        if (in_size != out_size && in_size != 1) {
            free(out_to_in);
            write_error(err_msg, err_cap,
                        "broadcast_in_dim requires broadcast-compatible shapes");
            return 11;
        }
    }

    int64_t total = numel(out);
    broadcast_in_dim_f32((const float *)a->data, (float *)out->data, out_ndim, a->sizes,
                         a->strides, out->sizes, out_to_in, total);

    free(out_to_in);
    return 0;
}
