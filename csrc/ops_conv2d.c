#include "ops_utils.h"

typedef struct RefConv2dParams {
    int64_t stride_h;
    int64_t stride_w;
    int64_t padding_h;
    int64_t padding_w;
    int64_t dilation_h;
    int64_t dilation_w;
    int64_t groups;
} RefConv2dParams;

static void conv2d_f32(
    const float *input,
    const float *weight,
    float *output,
    int64_t batch,
    int64_t in_channels,
    int64_t in_h,
    int64_t in_w,
    int64_t out_channels,
    int64_t k_h,
    int64_t k_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dil_h,
    int64_t dil_w,
    int64_t groups) {
    int64_t out_h = (in_h + 2 * pad_h - dil_h * (k_h - 1) - 1) / stride_h + 1;
    int64_t out_w = (in_w + 2 * pad_w - dil_w * (k_w - 1) - 1) / stride_w + 1;
    int64_t in_per_group = in_channels / groups;
    int64_t out_per_group = out_channels / groups;

    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t oc = 0; oc < out_channels; ++oc) {
            int64_t group = oc / out_per_group;
            for (int64_t oh = 0; oh < out_h; ++oh) {
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    float acc = 0.0f;
                    int64_t in_h_base = oh * stride_h - pad_h;
                    int64_t in_w_base = ow * stride_w - pad_w;
                    for (int64_t ic = 0; ic < in_per_group; ++ic) {
                        int64_t in_c = group * in_per_group + ic;
                        for (int64_t kh = 0; kh < k_h; ++kh) {
                            int64_t in_h_idx = in_h_base + kh * dil_h;
                            if (in_h_idx < 0 || in_h_idx >= in_h) {
                                continue;
                            }
                            for (int64_t kw = 0; kw < k_w; ++kw) {
                                int64_t in_w_idx = in_w_base + kw * dil_w;
                                if (in_w_idx < 0 || in_w_idx >= in_w) {
                                    continue;
                                }
                                int64_t input_offset =
                                    ((n * in_channels + in_c) * in_h + in_h_idx) * in_w + in_w_idx;
                                int64_t weight_offset =
                                    ((oc * in_per_group + ic) * k_h + kh) * k_w + kw;
                                acc += input[input_offset] * weight[weight_offset];
                            }
                        }
                    }
                    int64_t out_offset = ((n * out_channels + oc) * out_h + oh) * out_w + ow;
                    output[out_offset] = acc;
                }
            }
        }
    }
}

int ref_run_conv2d(const RefOpCall *call, char *err_msg, size_t err_cap) {
    if (call->n_inputs != 2 || call->n_outputs != 1) {
        write_error(err_msg, err_cap, "conv2d expects exactly 2 inputs and 1 output");
        return 1;
    }
    if (call->params == NULL) {
        write_error(err_msg, err_cap, "conv2d expects parameters");
        return 2;
    }

    RefTensorView *input = &call->inputs[0];
    RefTensorView *weight = &call->inputs[1];
    RefTensorView *out = &call->outputs[0];
    RefConv2dParams *params = (RefConv2dParams *)call->params;

    if (input->dtype != REF_F32 || weight->dtype != REF_F32 || out->dtype != REF_F32) {
        write_error(err_msg, err_cap, "conv2d supports only REF_F32 dtype");
        return 3;
    }

    if (input->ndim != 4 || weight->ndim != 4 || out->ndim != 4) {
        write_error(err_msg, err_cap, "conv2d requires 4D input, weight, and output");
        return 4;
    }

    if (!is_contiguous(input) || !is_contiguous(weight) || !is_contiguous(out)) {
        write_error(err_msg, err_cap, "conv2d requires contiguous tensors");
        return 5;
    }

    int64_t stride_h = params->stride_h;
    int64_t stride_w = params->stride_w;
    int64_t pad_h = params->padding_h;
    int64_t pad_w = params->padding_w;
    int64_t dil_h = params->dilation_h;
    int64_t dil_w = params->dilation_w;
    int64_t groups = params->groups;

    if (stride_h <= 0 || stride_w <= 0 || dil_h <= 0 || dil_w <= 0 || pad_h < 0 || pad_w < 0) {
        write_error(
            err_msg,
            err_cap,
            "conv2d expects stride and dilation to be positive and padding to be non-negative");
        return 6;
    }

    if (groups <= 0) {
        write_error(err_msg, err_cap, "conv2d requires positive groups");
        return 7;
    }

    int64_t batch = input->sizes[0];
    int64_t in_channels = input->sizes[1];
    int64_t in_h = input->sizes[2];
    int64_t in_w = input->sizes[3];

    int64_t out_channels = weight->sizes[0];
    int64_t weight_in_channels = weight->sizes[1];
    int64_t k_h = weight->sizes[2];
    int64_t k_w = weight->sizes[3];

    if (in_channels != weight_in_channels * groups) {
        write_error(err_msg, err_cap, "conv2d requires input channels to match weight channels * groups");
        return 8;
    }

    if (out_channels % groups != 0) {
        write_error(err_msg, err_cap, "conv2d requires output channels to be divisible by groups");
        return 9;
    }

    int64_t numerator_h = in_h + 2 * pad_h - dil_h * (k_h - 1) - 1;
    int64_t numerator_w = in_w + 2 * pad_w - dil_w * (k_w - 1) - 1;
    if (numerator_h < 0 || numerator_w < 0) {
        write_error(err_msg, err_cap, "conv2d requires output shape (N, C_out, H_out, W_out)");
        return 10;
    }
    int64_t out_h = numerator_h / stride_h + 1;
    int64_t out_w = numerator_w / stride_w + 1;

    if (out->sizes[0] != batch || out->sizes[1] != out_channels || out->sizes[2] != out_h ||
        out->sizes[3] != out_w) {
        write_error(err_msg, err_cap, "conv2d requires output shape (N, C_out, H_out, W_out)");
        return 11;
    }

    float *input_data = (float *)input->data;
    float *weight_data = (float *)weight->data;
    float *out_data = (float *)out->data;

    conv2d_f32(
        input_data,
        weight_data,
        out_data,
        batch,
        in_channels,
        in_h,
        in_w,
        out_channels,
        k_h,
        k_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        groups);
    return 0;
}
