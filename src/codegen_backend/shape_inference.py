from __future__ import annotations

import math
import numbers
from fractions import Fraction
from typing import Sequence, Tuple

from c_ref_backend.cffi_bindings import RefBackendError


def compute_arange_size(
    start: float | int | bool,
    end: float | int | bool,
    step: float | int | bool,
) -> int:
    if step == 0:
        raise RefBackendError("codegen arange expects step to be non-zero")
    if all(
        isinstance(value, numbers.Integral) for value in (start, end, step)
    ):
        start_value = int(start)
        end_value = int(end)
        step_value = int(step)
        if step_value == 0:
            raise RefBackendError("codegen arange expects step to be non-zero")
        if step_value > 0 and end_value <= start_value:
            return 0
        if step_value < 0 and end_value >= start_value:
            return 0
        delta = end_value - start_value
        size = int(math.ceil(Fraction(delta, step_value)))
        return max(size, 0)
    start_value = float(start)
    end_value = float(end)
    step_value = float(step)
    delta = (end_value - start_value) / step_value
    size = int(math.ceil(delta))
    return max(size, 0)


def infer_diagonal_output_shape(
    input_shape: Sequence[int], offset: int, dim1: int, dim2: int
) -> Tuple[int, ...]:
    size1 = input_shape[dim1]
    size2 = input_shape[dim2]
    if offset >= 0:
        diag_len = min(size1, size2 - offset)
    else:
        diag_len = min(size1 + offset, size2)
    diag_len = max(0, diag_len)
    output_dims = [
        size
        for index, size in enumerate(input_shape)
        if index not in (dim1, dim2)
    ]
    output_dims.append(diag_len)
    return tuple(output_dims)


def infer_reduction_output_shape(
    input_shape: Sequence[int],
    reduction_dims: Tuple[int, ...],
    keepdim: bool,
    *,
    reduce_all: bool,
) -> Tuple[int, ...]:
    if reduce_all:
        return ()
    if not reduction_dims:
        return tuple(input_shape)
    if keepdim:
        output_shape = list(input_shape)
        for dim in reduction_dims:
            output_shape[dim] = 1
        return tuple(output_shape)
    return tuple(
        size for dim, size in enumerate(input_shape) if dim not in reduction_dims
    )


def unpack_conv2d_input_shape(
    input_shape: Sequence[int],
) -> Tuple[bool, int, int, int, int]:
    if len(input_shape) == 4:
        batch, in_channels, in_h, in_w = input_shape
        return True, batch, in_channels, in_h, in_w
    if len(input_shape) == 3:
        in_channels, in_h, in_w = input_shape
        return False, 1, in_channels, in_h, in_w
    raise RefBackendError("codegen conv2d requires 3D or 4D input tensors")


def conv2d_output_shape_from_shapes(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    groups: int,
) -> Tuple[int, ...]:
    has_batch, batch, in_channels, in_h, in_w = unpack_conv2d_input_shape(
        input_shape
    )
    out_channels, weight_in_channels, kernel_h, kernel_w = weight_shape
    if in_channels != weight_in_channels * groups:
        raise RefBackendError(
            "codegen conv2d requires input channels to match weight channels * groups"
        )
    if out_channels % groups != 0:
        raise RefBackendError(
            "codegen conv2d requires output channels to be divisible by groups"
        )
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    numerator_h = in_h + 2 * pad_h - dil_h * (kernel_h - 1) - 1
    numerator_w = in_w + 2 * pad_w - dil_w * (kernel_w - 1) - 1
    if numerator_h < 0 or numerator_w < 0:
        raise RefBackendError(
            "codegen conv2d requires output shape (N, C_out, H_out, W_out)"
        )
    out_h = numerator_h // stride_h + 1
    out_w = numerator_w // stride_w + 1
    if has_batch:
        return batch, out_channels, out_h, out_w
    return out_channels, out_h, out_w


def conv2d_transposed_output_shape_from_shapes(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    output_padding: Tuple[int, int],
    groups: int,
) -> Tuple[int, ...]:
    has_batch, batch, in_channels, in_h, in_w = unpack_conv2d_input_shape(
        input_shape
    )
    weight_in_channels, weight_out_channels, kernel_h, kernel_w = weight_shape
    if in_channels != weight_in_channels:
        raise RefBackendError(
            "codegen conv2d requires input channels to match weight channels"
        )
    if in_channels % groups != 0:
        raise RefBackendError(
            "codegen conv2d requires input channels to be divisible by groups"
        )
    out_channels = weight_out_channels * groups
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    out_pad_h, out_pad_w = output_padding
    out_h = (
        (in_h - 1) * stride_h
        - 2 * pad_h
        + dil_h * (kernel_h - 1)
        + out_pad_h
        + 1
    )
    out_w = (
        (in_w - 1) * stride_w
        - 2 * pad_w
        + dil_w * (kernel_w - 1)
        + out_pad_w
        + 1
    )
    if out_h <= 0 or out_w <= 0:
        raise RefBackendError(
            "codegen conv2d requires output shape (N, C_out, H_out, W_out)"
        )
    if has_batch:
        return batch, out_channels, out_h, out_w
    return out_channels, out_h, out_w


def conv2d_same_padding(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    stride: Tuple[int, int],
    dilation: Tuple[int, int],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    _, _, in_h, in_w = unpack_conv2d_input_shape(input_shape)[1:]
    _, _, kernel_h, kernel_w = weight_shape
    stride_h, stride_w = stride
    dil_h, dil_w = dilation
    out_h = math.ceil(in_h / stride_h)
    out_w = math.ceil(in_w / stride_w)
    pad_h = max(
        (out_h - 1) * stride_h + (dil_h * (kernel_h - 1) + 1) - in_h,
        0,
    )
    pad_w = max(
        (out_w - 1) * stride_w + (dil_w * (kernel_w - 1) + 1) - in_w,
        0,
    )
    pad_top = pad_h // 2
    pad_left = pad_w // 2
    return (pad_top, pad_left), (out_h, out_w)


def conv2d_validate_channels(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    groups: int,
) -> Tuple[bool, int]:
    has_batch, _, in_channels, _, _ = unpack_conv2d_input_shape(input_shape)
    out_channels, weight_in_channels, _, _ = weight_shape
    if in_channels != weight_in_channels * groups:
        raise RefBackendError(
            "codegen conv2d requires input channels to match weight channels * groups"
        )
    if out_channels % groups != 0:
        raise RefBackendError(
            "codegen conv2d requires output channels to be divisible by groups"
        )
    return has_batch, out_channels


def conv1d_validate_channels(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    groups: int,
) -> Tuple[int, int]:
    batch, in_channels, _ = input_shape
    out_channels, weight_in_channels, _ = weight_shape
    if in_channels != weight_in_channels * groups:
        raise RefBackendError(
            "codegen conv1d requires input channels to match weight channels * groups"
        )
    if out_channels % groups != 0:
        raise RefBackendError(
            "codegen conv1d requires output channels to be divisible by groups"
        )
    return batch, out_channels


def conv1d_same_padding(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    stride: int,
    dilation: int,
) -> Tuple[int, int]:
    _, _, in_l = input_shape
    _, _, kernel_l = weight_shape
    out_l = math.ceil(in_l / stride)
    pad_l = max(
        (out_l - 1) * stride + (dilation * (kernel_l - 1) + 1) - in_l,
        0,
    )
    pad_left = pad_l // 2
    return pad_left, out_l


def conv1d_output_shape_from_shapes(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
) -> Tuple[int, int, int]:
    batch, out_channels = conv1d_validate_channels(
        input_shape, weight_shape, groups
    )
    in_l = input_shape[2]
    kernel_l = weight_shape[2]
    numerator = in_l + 2 * padding - dilation * (kernel_l - 1) - 1
    if numerator < 0:
        raise RefBackendError(
            "codegen conv1d requires output shape (N, C_out, L_out)"
        )
    out_l = numerator // stride + 1
    return batch, out_channels, out_l


def pool1d_output_shape_from_shapes(
    input_shape: Sequence[int],
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    ceil_mode: bool,
) -> Tuple[int, int, int]:
    batch, channels, in_l = input_shape
    numerator = in_l + 2 * padding - dilation * (kernel_size - 1) - 1
    if numerator < 0:
        raise RefBackendError(
            "codegen pool1d requires output shape (N, C, L_out)"
        )
    if ceil_mode:
        out_l = (numerator + stride - 1) // stride + 1
        if (out_l - 1) * stride >= in_l + padding:
            out_l -= 1
    else:
        out_l = numerator // stride + 1
    return batch, channels, out_l


def pool2d_output_shape_from_shapes(
    input_shape: Sequence[int],
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    ceil_mode: bool = False,
) -> Tuple[int, int, int, int]:
    batch, channels, in_h, in_w = input_shape
    k_h, k_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    numerator_h = in_h + 2 * pad_h - dil_h * (k_h - 1) - 1
    numerator_w = in_w + 2 * pad_w - dil_w * (k_w - 1) - 1
    if numerator_h < 0 or numerator_w < 0:
        raise RefBackendError(
            "codegen pool2d requires output shape (N, C, H_out, W_out)"
        )
    if ceil_mode:
        out_h = (numerator_h + stride_h - 1) // stride_h + 1
        out_w = (numerator_w + stride_w - 1) // stride_w + 1
        if (out_h - 1) * stride_h >= in_h + pad_h:
            out_h -= 1
        if (out_w - 1) * stride_w >= in_w + pad_w:
            out_w -= 1
    else:
        out_h = numerator_h // stride_h + 1
        out_w = numerator_w // stride_w + 1
    return batch, channels, out_h, out_w


def pool3d_output_shape_from_shapes(
    input_shape: Sequence[int],
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
) -> Tuple[int, int, int, int, int]:
    batch, channels, in_d, in_h, in_w = input_shape
    k_d, k_h, k_w = kernel_size
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    dil_d, dil_h, dil_w = dilation
    numerator_d = in_d + 2 * pad_d - dil_d * (k_d - 1) - 1
    numerator_h = in_h + 2 * pad_h - dil_h * (k_h - 1) - 1
    numerator_w = in_w + 2 * pad_w - dil_w * (k_w - 1) - 1
    if numerator_d < 0 or numerator_h < 0 or numerator_w < 0:
        raise RefBackendError(
            "codegen pool3d requires output shape (N, C, D_out, H_out, W_out)"
        )
    out_d = numerator_d // stride_d + 1
    out_h = numerator_h // stride_h + 1
    out_w = numerator_w // stride_w + 1
    return batch, channels, out_d, out_h, out_w
