#include "ops_utils.h"

#include <math.h>
#include <stdlib.h>

static float sigmoid_f32(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static int64_t input_offset(int64_t t, int64_t b, int64_t input_size, int64_t batch,
                            int64_t seq_len, int batch_first) {
    if (batch_first) {
        return (b * seq_len + t) * input_size;
    }
    return (t * batch + b) * input_size;
}

static int64_t output_offset(int64_t t, int64_t b, int64_t hidden_size, int64_t batch,
                             int64_t seq_len, int batch_first) {
    if (batch_first) {
        return (b * seq_len + t) * hidden_size;
    }
    return (t * batch + b) * hidden_size;
}

int ref_run_lstm(const RefOpCall *call, char *err_msg, size_t err_cap) {
    if (call->n_inputs != 7 || call->n_outputs != 3) {
        write_error(err_msg, err_cap, "lstm expects 7 inputs and 3 outputs");
        return 1;
    }
    if (call->params == NULL) {
        write_error(err_msg, err_cap, "lstm requires params");
        return 2;
    }
    const RefLstmParams *params = (const RefLstmParams *)call->params;
    if (params->has_biases != 1) {
        write_error(err_msg, err_cap, "lstm supports only has_biases=True");
        return 3;
    }
    if (params->num_layers != 1) {
        write_error(err_msg, err_cap, "lstm supports only num_layers=1");
        return 4;
    }
    if (params->dropout != 0.0f) {
        write_error(err_msg, err_cap, "lstm supports only dropout=0");
        return 5;
    }
    if (params->train != 0) {
        write_error(err_msg, err_cap, "lstm supports only train=False");
        return 6;
    }
    if (params->bidirectional != 0) {
        write_error(err_msg, err_cap, "lstm supports only bidirectional=False");
        return 7;
    }

    RefTensorView *input = &call->inputs[0];
    RefTensorView *h0 = &call->inputs[1];
    RefTensorView *c0 = &call->inputs[2];
    RefTensorView *w_ih = &call->inputs[3];
    RefTensorView *w_hh = &call->inputs[4];
    RefTensorView *b_ih = &call->inputs[5];
    RefTensorView *b_hh = &call->inputs[6];
    RefTensorView *out = &call->outputs[0];
    RefTensorView *h_n = &call->outputs[1];
    RefTensorView *c_n = &call->outputs[2];

    if (input->dtype != REF_F32 || h0->dtype != REF_F32 || c0->dtype != REF_F32 ||
        w_ih->dtype != REF_F32 || w_hh->dtype != REF_F32 || b_ih->dtype != REF_F32 ||
        b_hh->dtype != REF_F32 || out->dtype != REF_F32 || h_n->dtype != REF_F32 ||
        c_n->dtype != REF_F32) {
        write_error(err_msg, err_cap, "lstm supports only REF_F32 dtype");
        return 8;
    }
    if (input->ndim != 3) {
        write_error(err_msg, err_cap, "lstm requires input to be 3D");
        return 9;
    }
    if (h0->ndim != 3 || c0->ndim != 3) {
        write_error(err_msg, err_cap, "lstm requires h0 and c0 to be 3D");
        return 10;
    }
    if (w_ih->ndim != 2 || w_hh->ndim != 2) {
        write_error(err_msg, err_cap, "lstm requires weight_ih and weight_hh to be 2D");
        return 11;
    }
    if (b_ih->ndim != 1 || b_hh->ndim != 1) {
        write_error(err_msg, err_cap, "lstm requires bias_ih and bias_hh to be 1D");
        return 12;
    }

    int64_t batch = params->batch_first ? input->sizes[0] : input->sizes[1];
    int64_t seq_len = params->batch_first ? input->sizes[1] : input->sizes[0];
    int64_t input_size = input->sizes[2];
    int64_t hidden_size = w_hh->sizes[1];
    int64_t gate_size = 4 * hidden_size;

    if (w_ih->sizes[0] != gate_size || w_hh->sizes[0] != gate_size) {
        write_error(err_msg, err_cap, "lstm requires weight_ih and weight_hh to have 4 * hidden_size rows");
        return 13;
    }
    if (w_ih->sizes[1] != input_size) {
        write_error(err_msg, err_cap, "lstm requires input_size to match weight_ih");
        return 14;
    }
    if (w_hh->sizes[1] != hidden_size) {
        write_error(err_msg, err_cap, "lstm requires hidden_size to match weight_hh");
        return 15;
    }
    if (b_ih->sizes[0] != gate_size || b_hh->sizes[0] != gate_size) {
        write_error(err_msg, err_cap, "lstm requires bias_ih and bias_hh to have 4 * hidden_size elements");
        return 16;
    }
    if (h0->sizes[0] != 1 || c0->sizes[0] != 1 || h0->sizes[1] != batch ||
        c0->sizes[1] != batch || h0->sizes[2] != hidden_size || c0->sizes[2] != hidden_size) {
        write_error(err_msg, err_cap, "lstm requires h0 and c0 shape (1, batch, hidden_size)");
        return 17;
    }
    if (params->batch_first) {
        if (out->sizes[0] != batch || out->sizes[1] != seq_len ||
            out->sizes[2] != hidden_size) {
            write_error(err_msg, err_cap, "lstm requires output shape (batch, seq_len, hidden_size)");
            return 18;
        }
    } else {
        if (out->sizes[0] != seq_len || out->sizes[1] != batch ||
            out->sizes[2] != hidden_size) {
            write_error(err_msg, err_cap, "lstm requires output shape (seq_len, batch, hidden_size)");
            return 19;
        }
    }
    if (h_n->sizes[0] != 1 || c_n->sizes[0] != 1 || h_n->sizes[1] != batch ||
        c_n->sizes[1] != batch || h_n->sizes[2] != hidden_size || c_n->sizes[2] != hidden_size) {
        write_error(err_msg, err_cap, "lstm requires h_n and c_n shape (1, batch, hidden_size)");
        return 20;
    }
    if (!is_contiguous(input) || !is_contiguous(h0) || !is_contiguous(c0) ||
        !is_contiguous(w_ih) || !is_contiguous(w_hh) || !is_contiguous(b_ih) ||
        !is_contiguous(b_hh) || !is_contiguous(out) || !is_contiguous(h_n) ||
        !is_contiguous(c_n)) {
        write_error(err_msg, err_cap, "lstm requires contiguous tensors");
        return 21;
    }

    float *input_data = (float *)input->data;
    float *h0_data = (float *)h0->data;
    float *c0_data = (float *)c0->data;
    float *w_ih_data = (float *)w_ih->data;
    float *w_hh_data = (float *)w_hh->data;
    float *b_ih_data = (float *)b_ih->data;
    float *b_hh_data = (float *)b_hh->data;
    float *out_data = (float *)out->data;
    float *h_n_data = (float *)h_n->data;
    float *c_n_data = (float *)c_n->data;

    int64_t state_size = batch * hidden_size;
    float *h_prev = (float *)malloc(sizeof(float) * state_size);
    float *c_prev = (float *)malloc(sizeof(float) * state_size);
    float *h_next = (float *)malloc(sizeof(float) * state_size);
    float *c_next = (float *)malloc(sizeof(float) * state_size);
    if (h_prev == NULL || c_prev == NULL || h_next == NULL || c_next == NULL) {
        free(h_prev);
        free(c_prev);
        free(h_next);
        free(c_next);
        write_error(err_msg, err_cap, "lstm out of memory");
        return 22;
    }

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < hidden_size; ++h) {
            h_prev[b * hidden_size + h] = h0_data[b * hidden_size + h];
            c_prev[b * hidden_size + h] = c0_data[b * hidden_size + h];
        }
    }

    if (seq_len == 0) {
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t h = 0; h < hidden_size; ++h) {
                h_n_data[b * hidden_size + h] = h_prev[b * hidden_size + h];
                c_n_data[b * hidden_size + h] = c_prev[b * hidden_size + h];
            }
        }
        free(h_prev);
        free(c_prev);
        free(h_next);
        free(c_next);
        return 0;
    }

    for (int64_t t = 0; t < seq_len; ++t) {
        for (int64_t b = 0; b < batch; ++b) {
            const float *x_ptr = input_data + input_offset(t, b, input_size, batch, seq_len, params->batch_first);
            for (int64_t h = 0; h < hidden_size; ++h) {
                float gate_i = b_ih_data[h] + b_hh_data[h];
                float gate_f = b_ih_data[hidden_size + h] + b_hh_data[hidden_size + h];
                float gate_g = b_ih_data[2 * hidden_size + h] + b_hh_data[2 * hidden_size + h];
                float gate_o = b_ih_data[3 * hidden_size + h] + b_hh_data[3 * hidden_size + h];

                const float *w_ih_i = w_ih_data + h * input_size;
                const float *w_ih_f = w_ih_data + (hidden_size + h) * input_size;
                const float *w_ih_g = w_ih_data + (2 * hidden_size + h) * input_size;
                const float *w_ih_o = w_ih_data + (3 * hidden_size + h) * input_size;
                for (int64_t i = 0; i < input_size; ++i) {
                    float x_val = x_ptr[i];
                    gate_i += w_ih_i[i] * x_val;
                    gate_f += w_ih_f[i] * x_val;
                    gate_g += w_ih_g[i] * x_val;
                    gate_o += w_ih_o[i] * x_val;
                }

                const float *w_hh_i = w_hh_data + h * hidden_size;
                const float *w_hh_f = w_hh_data + (hidden_size + h) * hidden_size;
                const float *w_hh_g = w_hh_data + (2 * hidden_size + h) * hidden_size;
                const float *w_hh_o = w_hh_data + (3 * hidden_size + h) * hidden_size;
                const float *h_prev_ptr = h_prev + b * hidden_size;
                for (int64_t i = 0; i < hidden_size; ++i) {
                    float h_val = h_prev_ptr[i];
                    gate_i += w_hh_i[i] * h_val;
                    gate_f += w_hh_f[i] * h_val;
                    gate_g += w_hh_g[i] * h_val;
                    gate_o += w_hh_o[i] * h_val;
                }

                gate_i = sigmoid_f32(gate_i);
                gate_f = sigmoid_f32(gate_f);
                gate_g = tanhf(gate_g);
                gate_o = sigmoid_f32(gate_o);

                float c_val = gate_f * c_prev[b * hidden_size + h] + gate_i * gate_g;
                float h_val = gate_o * tanhf(c_val);
                c_next[b * hidden_size + h] = c_val;
                h_next[b * hidden_size + h] = h_val;
            }
        }

        for (int64_t b = 0; b < batch; ++b) {
            int64_t out_base = output_offset(t, b, hidden_size, batch, seq_len, params->batch_first);
            for (int64_t h = 0; h < hidden_size; ++h) {
                out_data[out_base + h] = h_next[b * hidden_size + h];
            }
        }

        float *tmp = h_prev;
        h_prev = h_next;
        h_next = tmp;
        tmp = c_prev;
        c_prev = c_next;
        c_next = tmp;
    }

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < hidden_size; ++h) {
            h_n_data[b * hidden_size + h] = h_prev[b * hidden_size + h];
            c_n_data[b * hidden_size + h] = c_prev[b * hidden_size + h];
        }
    }

    free(h_prev);
    free(c_prev);
    free(h_next);
    free(c_next);
    return 0;
}
