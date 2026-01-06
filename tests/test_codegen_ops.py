import copy
import operator

import pytest
import torch
from codegen_backend import codegen_generic_backend
from torch.testing._internal.common_device_type import instantiate_device_type_tests, ops
from torch.testing._internal.common_methods_invocations import SampleInput, op_db
from torch.testing._internal.common_utils import TestCase


def _flatten_tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        tensors = []
        for item in value:
            tensors.extend(_flatten_tensors(item))
        return tensors
    return []


def _extract_tensors(sample):
    tensors = _flatten_tensors(sample.input)
    for arg in sample.args:
        tensors.extend(_flatten_tensors(arg))
    return tensors


def _update_sample(sample, updated_tensors):
    tensor_iter = iter(updated_tensors)
    if isinstance(sample.input, (list, tuple)):
        new_input = type(sample.input)(
            next(tensor_iter) if isinstance(item, torch.Tensor) else item
            for item in sample.input
        )
    else:
        new_input = next(tensor_iter)
    new_args = []
    for arg in sample.args:
        if isinstance(arg, torch.Tensor):
            new_args.append(next(tensor_iter))
        elif isinstance(arg, (list, tuple)):
            new_args.append(
                type(arg)(
                    next(tensor_iter) if isinstance(item, torch.Tensor) else item
                    for item in arg
                )
            )
        else:
            new_args.append(arg)
    return SampleInput(new_input, args=tuple(new_args), kwargs=sample.kwargs)


def _addmv_sample_filter(sample):
    tensors = _extract_tensors(sample)
    if len(tensors) != 3:
        return False
    input_tensor, mat, vec = tensors
    if input_tensor.ndim != 1 or mat.ndim != 2 or vec.ndim != 1:
        return False
    if mat.shape[1] != vec.shape[0]:
        return False
    expected_shape = (mat.shape[0],)
    return input_tensor.shape == expected_shape


def _addr_sample_filter(sample):
    tensors = _extract_tensors(sample)
    if len(tensors) != 3:
        return False
    input_tensor, vec1, vec2 = tensors
    if input_tensor.ndim != 2 or vec1.ndim != 1 or vec2.ndim != 1:
        return False
    expected_shape = (vec1.shape[0], vec2.shape[0])
    return input_tensor.shape == expected_shape


def _all_same_shape(tensors):
    if not tensors:
        return True
    shape = tensors[0].shape
    return all(tensor.shape == shape for tensor in tensors[1:])


def _broadcast_shapes(*shapes):
    if not shapes:
        return ()
    max_len = max(len(shape) for shape in shapes)
    output_shape = []
    for dim in range(1, max_len + 1):
        sizes = [
            shape[-dim] if dim <= len(shape) else 1
            for shape in shapes
        ]
        max_size = max(sizes)
        if any(size not in (1, max_size) for size in sizes):
            raise ValueError("shapes are not broadcastable")
        output_shape.append(max_size)
    return tuple(reversed(output_shape))


def _broadcastable_sample_filter(sample):
    tensors = _extract_tensors(sample)
    if not tensors:
        return False
    try:
        _broadcast_shapes(*(tensor.shape for tensor in tensors))
    except ValueError:
        return False
    return True


def _concat_sample_filter(sample):
    if not isinstance(sample.input, (list, tuple)):
        return False
    tensors = _extract_tensors(sample)
    if not tensors:
        return False
    dim = sample.kwargs.get("dim", 0)
    if sample.args:
        dim = sample.args[0]
    if not isinstance(dim, int):
        return False
    rank = tensors[0].ndim
    if rank == 0:
        return False
    if dim < 0:
        dim += rank
    if dim < 0 or dim >= rank:
        return False
    for tensor in tensors:
        if tensor.ndim != rank:
            return False
    base_shape = tensors[0].shape
    for tensor in tensors[1:]:
        for dim_index, size in enumerate(tensor.shape):
            if dim_index == dim:
                continue
            if size != base_shape[dim_index]:
                return False
    return True


def _var_dim_sample_filter(sample):
    return "correction" not in sample.kwargs


def _norm_dim_sample_filter(sample):
    return len(sample.args) >= 2


def _cumsum_sample_filter(sample):
    if not isinstance(sample.input, torch.Tensor):
        return False
    dim = sample.args[0] if sample.args else sample.kwargs.get("dim")
    dtype = None
    if len(sample.args) > 1:
        dtype = sample.args[1]
    if "dtype" in sample.kwargs:
        dtype = sample.kwargs["dtype"]
    if not isinstance(dim, int):
        return False
    rank = sample.input.ndim
    if rank == 0:
        if dim not in (-1, 0):
            return False
    else:
        if dim < 0:
            dim += rank
        if dim < 0 or dim >= rank:
            return False
    if dtype is not None and dtype is not sample.input.dtype:
        return False
    return True


def _resize_sample_filter(sample):
    if not isinstance(sample.input, torch.Tensor):
        return False
    if sample.kwargs.get("memory_format") is not None:
        return False
    size_value = None
    if sample.args:
        size_value = sample.args[0]
    else:
        size_value = sample.kwargs.get("size")
    if size_value is None:
        return False
    if isinstance(size_value, torch.Size):
        size_value = tuple(size_value)
    if not isinstance(size_value, (list, tuple)):
        return False
    try:
        size_tuple = tuple(int(operator.index(item)) for item in size_value)
    except TypeError:
        return False
    return size_tuple == tuple(sample.input.shape)


def _normalize_conv2d_param(value):
    if isinstance(value, int):
        return (value, value)
    if (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(isinstance(item, int) for item in value)
    ):
        return (value[0], value[1])
    return None


def _normalize_pool2d_param(value):
    if isinstance(value, int):
        return (value, value)
    if (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(isinstance(item, int) for item in value)
    ):
        return (value[0], value[1])
    return None


def _normalize_pool1d_param(value):
    if isinstance(value, int):
        return value
    if (
        isinstance(value, (tuple, list))
        and len(value) == 1
        and all(isinstance(item, int) for item in value)
    ):
        return value[0]
    return None


def _convolution_sample_filter(sample):
    if not isinstance(sample.input, torch.Tensor):
        return False
    if not sample.args:
        return False
    args = list(sample.args)
    weight = args[0] if len(args) > 0 else sample.kwargs.get("weight")
    if not isinstance(weight, torch.Tensor):
        return False
    stride = sample.kwargs.get("stride", args[2] if len(args) > 2 else 1)
    padding = sample.kwargs.get("padding", args[3] if len(args) > 3 else 0)
    dilation = sample.kwargs.get("dilation", args[4] if len(args) > 4 else 1)
    transposed = sample.kwargs.get(
        "transposed", args[5] if len(args) > 5 else False
    )
    output_padding = sample.kwargs.get(
        "output_padding", args[6] if len(args) > 6 else 0
    )
    groups = sample.kwargs.get("groups", args[7] if len(args) > 7 else 1)
    if transposed:
        return False
    output_padding_pair = _normalize_conv2d_param(output_padding)
    if output_padding_pair is None or output_padding_pair != (0, 0):
        return False
    stride_pair = _normalize_conv2d_param(stride)
    padding_pair = _normalize_conv2d_param(padding)
    dilation_pair = _normalize_conv2d_param(dilation)
    if stride_pair is None or padding_pair is None or dilation_pair is None:
        return False
    if (
        stride_pair[0] <= 0
        or stride_pair[1] <= 0
        or dilation_pair[0] <= 0
        or dilation_pair[1] <= 0
        or padding_pair[0] < 0
        or padding_pair[1] < 0
    ):
        return False
    if not isinstance(groups, int) or groups <= 0:
        return False
    if sample.input.ndim != 4 or weight.ndim != 4:
        return False
    if not sample.input.is_contiguous() or not weight.is_contiguous():
        return False
    if sample.input.shape[1] != weight.shape[1] * groups:
        return False
    if weight.shape[0] % groups != 0:
        return False
    return True


def _max_pool1d_sample_filter(sample):
    if not isinstance(sample.input, torch.Tensor):
        return False
    if sample.input.ndim != 3:
        return False
    if not sample.input.is_contiguous():
        return False
    args = list(sample.args)
    kernel_size = args[0] if len(args) > 0 else sample.kwargs.get("kernel_size")
    stride = args[1] if len(args) > 1 else sample.kwargs.get("stride")
    padding = args[2] if len(args) > 2 else sample.kwargs.get("padding", 0)
    dilation = args[3] if len(args) > 3 else sample.kwargs.get("dilation", 1)
    ceil_mode = args[4] if len(args) > 4 else sample.kwargs.get("ceil_mode", False)
    kernel_value = _normalize_pool1d_param(kernel_size)
    if kernel_value is None:
        return False
    if stride is None:
        stride_value = kernel_value
    else:
        stride_value = _normalize_pool1d_param(stride)
        if stride_value is None:
            return False
    padding_value = _normalize_pool1d_param(padding)
    dilation_value = _normalize_pool1d_param(dilation)
    if padding_value is None or dilation_value is None:
        return False
    if (
        kernel_value <= 0
        or stride_value <= 0
        or dilation_value <= 0
        or padding_value < 0
    ):
        return False
    if ceil_mode:
        return False
    return True


def _avg_pool1d_sample_filter(sample):
    if not isinstance(sample.input, torch.Tensor):
        return False
    if sample.input.ndim != 3:
        return False
    if not sample.input.is_contiguous():
        return False
    args = list(sample.args)
    kernel_size = args[0] if len(args) > 0 else sample.kwargs.get("kernel_size")
    stride = args[1] if len(args) > 1 else sample.kwargs.get("stride")
    padding = args[2] if len(args) > 2 else sample.kwargs.get("padding", 0)
    ceil_mode = args[3] if len(args) > 3 else sample.kwargs.get("ceil_mode", False)
    count_include_pad = (
        args[4] if len(args) > 4 else sample.kwargs.get("count_include_pad", False)
    )
    divisor_override = (
        args[5] if len(args) > 5 else sample.kwargs.get("divisor_override", None)
    )
    kernel_value = _normalize_pool1d_param(kernel_size)
    if kernel_value is None:
        return False
    if stride is None:
        stride_value = kernel_value
    else:
        stride_value = _normalize_pool1d_param(stride)
        if stride_value is None:
            return False
    padding_value = _normalize_pool1d_param(padding)
    if padding_value is None:
        return False
    if kernel_value <= 0 or stride_value <= 0 or padding_value < 0:
        return False
    if ceil_mode:
        return False
    if not isinstance(count_include_pad, bool):
        return False
    if divisor_override is not None and (
        not isinstance(divisor_override, int) or divisor_override <= 0
    ):
        return False
    return True


def _max_pool2d_sample_filter(sample):
    if not isinstance(sample.input, torch.Tensor):
        return False
    if sample.input.ndim != 4:
        return False
    if not sample.input.is_contiguous():
        return False
    args = list(sample.args)
    kernel_size = args[0] if len(args) > 0 else sample.kwargs.get("kernel_size")
    stride = args[1] if len(args) > 1 else sample.kwargs.get("stride")
    padding = args[2] if len(args) > 2 else sample.kwargs.get("padding", 0)
    dilation = args[3] if len(args) > 3 else sample.kwargs.get("dilation", 1)
    ceil_mode = args[4] if len(args) > 4 else sample.kwargs.get("ceil_mode", False)
    kernel_pair = _normalize_pool2d_param(kernel_size)
    if kernel_pair is None:
        return False
    if stride is None:
        stride_pair = kernel_pair
    else:
        stride_pair = _normalize_pool2d_param(stride)
        if stride_pair is None:
            return False
    padding_pair = _normalize_pool2d_param(padding)
    dilation_pair = _normalize_pool2d_param(dilation)
    if padding_pair is None or dilation_pair is None:
        return False
    if (
        kernel_pair[0] <= 0
        or kernel_pair[1] <= 0
        or stride_pair[0] <= 0
        or stride_pair[1] <= 0
        or dilation_pair[0] <= 0
        or dilation_pair[1] <= 0
        or padding_pair[0] < 0
        or padding_pair[1] < 0
    ):
        return False
    if ceil_mode:
        return False
    return True


def _avg_pool2d_sample_filter(sample):
    if not isinstance(sample.input, torch.Tensor):
        return False
    if sample.input.ndim != 4:
        return False
    if not sample.input.is_contiguous():
        return False
    args = list(sample.args)
    kernel_size = args[0] if len(args) > 0 else sample.kwargs.get("kernel_size")
    stride = args[1] if len(args) > 1 else sample.kwargs.get("stride")
    padding = args[2] if len(args) > 2 else sample.kwargs.get("padding", 0)
    ceil_mode = args[3] if len(args) > 3 else sample.kwargs.get("ceil_mode", False)
    count_include_pad = (
        args[4] if len(args) > 4 else sample.kwargs.get("count_include_pad", False)
    )
    divisor_override = (
        args[5] if len(args) > 5 else sample.kwargs.get("divisor_override", None)
    )
    kernel_pair = _normalize_pool2d_param(kernel_size)
    if kernel_pair is None:
        return False
    if stride is None:
        stride_pair = kernel_pair
    else:
        stride_pair = _normalize_pool2d_param(stride)
        if stride_pair is None:
            return False
    padding_pair = _normalize_pool2d_param(padding)
    if padding_pair is None:
        return False
    if (
        kernel_pair[0] <= 0
        or kernel_pair[1] <= 0
        or stride_pair[0] <= 0
        or stride_pair[1] <= 0
        or padding_pair[0] < 0
        or padding_pair[1] < 0
    ):
        return False
    if ceil_mode:
        return False
    if not isinstance(count_include_pad, bool):
        return False
    if divisor_override is not None and (
        not isinstance(divisor_override, int) or divisor_override <= 0
    ):
        return False
    return True


def _sample_matches_constraints(sample, dtype, constraints):
    tensors = _extract_tensors(sample)
    if not tensors:
        return False
    max_ndim = constraints["max_ndim"]
    if max_ndim is not None and any(tensor.ndim > max_ndim for tensor in tensors):
        return False
    if not all(tensor.dtype is dtype for tensor in tensors):
        return False
    if constraints["requires_same_shape"] and not _all_same_shape(tensors):
        return False
    if constraints["requires_contiguous"] and any(
        not tensor.is_contiguous() for tensor in tensors
    ):
        return False
    sample_filter = constraints["sample_filter"]
    if sample_filter is not None and not sample_filter(sample):
        return False
    return True


def _iter_supported_samples(op, device, dtype, constraints):
    for sample in op.sample_inputs(device, dtype):
        if sample.kwargs and not constraints["allow_kwargs"]:
            continue
        if not constraints["allow_non_tensor_args"] and any(
            not isinstance(arg, torch.Tensor) for arg in sample.args
        ):
            continue
        if not _sample_matches_constraints(sample, dtype, constraints):
            continue
        yield sample

        if constraints["allow_noncontiguous"]:
            tensors = _extract_tensors(sample)
            if all(tensor.ndim >= 2 for tensor in tensors):
                transposed = [tensor.transpose(0, 1) for tensor in tensors]
                updated = _update_sample(sample, transposed)
                if _sample_matches_constraints(updated, dtype, constraints):
                    yield updated

            if all(tensor.ndim >= 1 and tensor.size(-1) > 1 for tensor in tensors):
                sliced = [tensor[..., ::2] for tensor in tensors]
                updated = _update_sample(sample, sliced)
                if _sample_matches_constraints(updated, dtype, constraints):
                    yield updated


CODEGEN_ATEN_OPS = [
    torch.ops.aten.abs.default,
    torch.ops.aten.absolute.default,
    torch.ops.aten.acos.default,
    torch.ops.aten.acosh.default,
    torch.ops.aten.add.Tensor,
    torch.ops.aten.add.Scalar,
    torch.ops.aten.all.default,
    torch.ops.aten.angle.default,
    torch.ops.aten.any.default,
    torch.ops.aten.any.dim,
    torch.ops.aten.any.dims,
    torch.ops.aten.argmax.default,
    torch.ops.aten.argmin.default,
    torch.ops.aten.amax.default,
    torch.ops.aten.amin.default,
    torch.ops.aten.asin.default,
    torch.ops.aten.asinh.default,
    torch.ops.aten.atan.default,
    torch.ops.aten.atan2.default,
    torch.ops.aten.atanh.default,
    torch.ops.aten.arccos.default,
    torch.ops.aten.arcsin.default,
    torch.ops.aten.arcsinh.default,
    torch.ops.aten.arctan.default,
    torch.ops.aten.bitwise_and.Tensor,
    torch.ops.aten.bitwise_and.Scalar,
    torch.ops.aten.bitwise_left_shift.Tensor,
    torch.ops.aten.bitwise_not.default,
    torch.ops.aten.bitwise_or.Tensor,
    torch.ops.aten.bitwise_or.Scalar,
    torch.ops.aten.bitwise_right_shift.Tensor,
    torch.ops.aten.bitwise_xor.Tensor,
    torch.ops.aten.bitwise_xor.Scalar,
    torch.ops.aten.bmm.default,
    torch.ops.aten.cat.default,
    torch.ops.aten.ceil.default,
    torch.ops.aten.clamp.default,
    torch.ops.aten.clamp_max.Tensor,
    torch.ops.aten.clamp_min.Tensor,
    torch.ops.aten.clone.default,
    torch.ops.aten.conj.default,
    torch.ops.aten.conj_physical.default,
    torch.ops.aten.copysign.Tensor,
    torch.ops.aten.copysign.Scalar,
    torch.ops.aten.conv1d.default,
    torch.ops.aten.conv2d.default,
    torch.ops.aten.convolution.default,
    torch.ops.aten.avg_pool1d.default,
    torch.ops.aten.avg_pool2d.default,
    torch.ops.aten.cos.default,
    torch.ops.aten.cosh.default,
    torch.ops.aten.cumsum.default,
    torch.ops.aten.deg2rad.default,
    torch.ops.aten.digamma.default,
    torch.ops.aten.diagonal.default,
    torch.ops.aten.div.Tensor,
    torch.ops.aten.div.Scalar,
    torch.ops.aten.embedding.default,
    torch.ops.aten.erf.default,
    torch.ops.aten.erfc.default,
    torch.ops.aten.erfinv.default,
    torch.ops.aten.exp.default,
    torch.ops.aten.exp2.default,
    torch.ops.aten.expm1.default,
    torch.ops.aten.fill.Scalar,
    torch.ops.aten.flip.default,
    torch.ops.aten.floor.default,
    torch.ops.aten.floor_divide.default,
    torch.ops.aten.floor_divide.Scalar,
    torch.ops.aten.fmax.default,
    torch.ops.aten.fmin.default,
    torch.ops.aten.fmod.Tensor,
    torch.ops.aten.fmod.Scalar,
    torch.ops.aten.frac.default,
    torch.ops.aten.heaviside.default,
    torch.ops.aten.hypot.default,
    torch.ops.aten.i0.default,
    torch.ops.aten.isfinite.default,
    torch.ops.aten.isinf.default,
    torch.ops.aten.isnan.default,
    torch.ops.aten.isneginf.default,
    torch.ops.aten.isposinf.default,
    torch.ops.aten.ldexp.Tensor,
    torch.ops.aten.lt.Tensor,
    torch.ops.aten.lt.Scalar,
    torch.ops.aten.le.Tensor,
    torch.ops.aten.le.Scalar,
    torch.ops.aten.gt.Tensor,
    torch.ops.aten.gt.Scalar,
    torch.ops.aten.ge.Tensor,
    torch.ops.aten.ge.Scalar,
    torch.ops.aten.eq.Tensor,
    torch.ops.aten.eq.Scalar,
    torch.ops.aten.ne.Tensor,
    torch.ops.aten.ne.Scalar,
    torch.ops.aten.logical_and.default,
    torch.ops.aten.logical_not.default,
    torch.ops.aten.logical_or.default,
    torch.ops.aten.logical_xor.default,
    torch.ops.aten.lgamma.default,
    torch.ops.aten.log.default,
    torch.ops.aten.log10.default,
    torch.ops.aten.log1p.default,
    torch.ops.aten.log2.default,
    torch.ops.aten.logaddexp.default,
    torch.ops.aten.logaddexp2.default,
    torch.ops.aten.log_sigmoid.default,
    torch.ops.aten.gelu.default,
    torch.ops.aten.elu.default,
    torch.ops.aten.leaky_relu.default,
    torch.ops.aten.softplus.default,
    torch.ops.aten.log_softmax.int,
    torch.ops.aten.logit.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.addbmm.default,
    torch.ops.aten.addmv.default,
    torch.ops.aten.addr.default,
    torch.ops.aten.mm.default,
    torch.ops.aten.matmul.default,
    torch.ops.aten.max_pool1d.default,
    torch.ops.aten.max_pool2d.default,
    torch.ops.aten.maximum.default,
    torch.ops.aten.minimum.default,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.mul.Scalar,
    torch.ops.aten.mean.default,
    torch.ops.aten.mean.dim,
    torch.ops.aten.std.default,
    torch.ops.aten.var.dim,
    torch.ops.aten.norm.ScalarOpt_dim,
    torch.ops.aten.nan_to_num.default,
    torch.ops.aten.neg.default,
    torch.ops.aten.nextafter.default,
    torch.ops.aten.positive.default,
    torch.ops.aten.pow.Tensor_Tensor,
    torch.ops.aten.pow.Scalar,
    torch.ops.aten.pow.Tensor_Scalar,
    torch.ops.aten.prod.default,
    torch.ops.aten.rad2deg.default,
    torch.ops.aten.real.default,
    torch.ops.aten.reciprocal.default,
    torch.ops.aten.relu.default,
    torch.ops.aten.remainder.Tensor,
    torch.ops.aten.remainder.Scalar,
    torch.ops.aten.round.default,
    torch.ops.aten.rsqrt.default,
    torch.ops.aten.selu.default,
    torch.ops.aten.sgn.default,
    torch.ops.aten.sigmoid.default,
    torch.ops.aten.sign.default,
    torch.ops.aten.softmax.int,
    torch.ops.aten.hardsigmoid.default,
    torch.ops.aten.hardtanh.default,
    torch.ops.aten.hardswish.default,
    torch.ops.aten.sin.default,
    torch.ops.aten.sinc.default,
    torch.ops.aten.sinh.default,
    torch.ops.aten.sqrt.default,
    torch.ops.aten.square.default,
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.sub.Scalar,
    torch.ops.aten.sum.default,
    torch.ops.aten.sum.dim_IntList,
    torch.ops.aten.tan.default,
    torch.ops.aten.tanh.default,
    torch.ops.aten.relu6.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.resize_.default,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.select.int,
    torch.ops.aten.narrow.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.expand.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.trunc.default,
    torch.ops.aten.xlogy.Tensor,
    torch.ops.aten.where.self,
    torch.ops.aten.where.Scalar,
]
CODEGEN_EXTRA_ATEN_OPS = [
    torch.ops.aten._softmax.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten.adaptive_avg_pool1d.default,
    torch.ops.aten._adaptive_avg_pool2d.default,
    torch.ops.aten._native_batch_norm_legit_no_training.default,
    torch.ops.aten._pdist_forward.default,
]
aten_cbrt = getattr(torch.ops.aten, "cbrt", None)
if aten_cbrt is not None:
    CODEGEN_ATEN_OPS.append(aten_cbrt.default)
INPLACE_ATEN_OPS = [
    torch.ops.aten.add_.Tensor,
    torch.ops.aten.abs_.default,
    torch.ops.aten.absolute_.default,
    torch.ops.aten.acos_.default,
    torch.ops.aten.arccos_.default,
    torch.ops.aten.acosh_.default,
    torch.ops.aten.arccosh_.default,
    torch.ops.aten.asin_.default,
    torch.ops.aten.arcsin_.default,
    torch.ops.aten.asinh_.default,
    torch.ops.aten.arcsinh_.default,
    torch.ops.aten.atan_.default,
    torch.ops.aten.arctan_.default,
    torch.ops.aten.atan2_.default,
    torch.ops.aten.atanh_.default,
    torch.ops.aten.bitwise_and_.Tensor,
    torch.ops.aten.bitwise_left_shift_.Tensor,
    torch.ops.aten.bitwise_not_.default,
    torch.ops.aten.bitwise_or_.Tensor,
    torch.ops.aten.bitwise_right_shift_.Tensor,
    torch.ops.aten.bitwise_xor_.Tensor,
    torch.ops.aten.ceil_.default,
    torch.ops.aten.clamp_.default,
    torch.ops.aten.clamp_max_.Tensor,
    torch.ops.aten.clamp_min_.Tensor,
    torch.ops.aten.conj_physical_.default,
    torch.ops.aten.copysign_.Tensor,
    torch.ops.aten.cos_.default,
    torch.ops.aten.cosh_.default,
    torch.ops.aten.deg2rad_.default,
    torch.ops.aten.digamma_.default,
    torch.ops.aten.div_.Tensor,
    torch.ops.aten.erf_.default,
    torch.ops.aten.erfc_.default,
    torch.ops.aten.erfinv_.default,
    torch.ops.aten.exp_.default,
    torch.ops.aten.exp2_.default,
    torch.ops.aten.expm1_.default,
    torch.ops.aten.fill_.Scalar,
    torch.ops.aten.floor_.default,
    torch.ops.aten.floor_divide_.Tensor,
    torch.ops.aten.fmod_.Tensor,
    torch.ops.aten.frac_.default,
    torch.ops.aten.heaviside_.default,
    torch.ops.aten.hypot_.default,
    torch.ops.aten.i0_.default,
    torch.ops.aten.ldexp_.default,
    torch.ops.aten.lgamma_.default,
    torch.ops.aten.log_.default,
    torch.ops.aten.log10_.default,
    torch.ops.aten.log1p_.default,
    torch.ops.aten.log2_.default,
    torch.ops.aten.logit_.default,
    torch.ops.aten.mish_.default,
    torch.ops.aten.logical_and_.default,
    torch.ops.aten.logical_not_.default,
    torch.ops.aten.logical_or_.default,
    torch.ops.aten.logical_xor_.default,
    torch.ops.aten.mul_.Tensor,
    torch.ops.aten.nan_to_num_.default,
    torch.ops.aten.neg_.default,
    torch.ops.aten.nextafter_.default,
    torch.ops.aten.pow_.Tensor,
    torch.ops.aten.pow_.Scalar,
    torch.ops.aten.rad2deg_.default,
    torch.ops.aten.reciprocal_.default,
    torch.ops.aten.relu_.default,
    torch.ops.aten.remainder_.Tensor,
    torch.ops.aten.round_.default,
    torch.ops.aten.rsqrt_.default,
    torch.ops.aten.sgn_.default,
    torch.ops.aten.sigmoid_.default,
    torch.ops.aten.sign_.default,
    torch.ops.aten.hardswish_.default,
    torch.ops.aten.hardtanh_.default,
    torch.ops.aten.sin_.default,
    torch.ops.aten.sinc_.default,
    torch.ops.aten.silu_.default,
    torch.ops.aten.sinh_.default,
    torch.ops.aten.sqrt_.default,
    torch.ops.aten.square_.default,
    torch.ops.aten.sub_.Tensor,
    torch.ops.aten.tan_.default,
    torch.ops.aten.tanh_.default,
    torch.ops.aten.trunc_.default,
    torch.ops.aten.xlogy_.Tensor,
]
aten_cbrt_inplace = getattr(torch.ops.aten, "cbrt_", None)
if aten_cbrt_inplace is not None:
    INPLACE_ATEN_OPS.append(aten_cbrt_inplace.default)
aten_gelu_inplace = getattr(torch.ops.aten, "gelu_", None)
if aten_gelu_inplace is not None:
    INPLACE_ATEN_OPS.append(aten_gelu_inplace.default)
aten_elu_inplace = getattr(torch.ops.aten, "elu_", None)
if aten_elu_inplace is not None:
    INPLACE_ATEN_OPS.append(aten_elu_inplace.default)
aten_leaky_relu_inplace = getattr(torch.ops.aten, "leaky_relu_", None)
if aten_leaky_relu_inplace is not None:
    INPLACE_ATEN_OPS.append(aten_leaky_relu_inplace.default)


def _lookup_opinfo(aten_name, variant_test_name):
    matches = [
        op
        for op in op_db
        if op.aten_name == aten_name and op.variant_test_name == variant_test_name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one OpInfo entry for {aten_name} "
            f"variant '{variant_test_name}'; found {len(matches)}"
        )
    return matches[0]


def _clone_opinfo(opinfo, *, name, aten_name):
    cloned = copy.deepcopy(opinfo)
    cloned.name = name
    cloned.aten_name = aten_name
    return cloned


def _clone_scalar_opinfo(aten_name, variant_test_name=""):
    return _clone_opinfo(
        _lookup_opinfo(aten_name, variant_test_name),
        name=f"{aten_name}_scalar",
        aten_name=aten_name,
    )


def _clone_tensor_scalar_opinfo(aten_name, variant_test_name=""):
    return _clone_opinfo(
        _lookup_opinfo(aten_name, variant_test_name),
        name=f"{aten_name}_tensor_scalar",
        aten_name=aten_name,
    )


CODEGEN_OPINFO_OVERRIDES = {
    torch.ops.aten.div.Tensor: _lookup_opinfo("div", "no_rounding_mode"),
    torch.ops.aten.div.Scalar: _clone_scalar_opinfo("div", "no_rounding_mode"),
    torch.ops.aten.hardsigmoid.default: _lookup_opinfo(
        "nn.functional.hardsigmoid", ""
    ),
    torch.ops.aten.add.Scalar: _clone_scalar_opinfo("add", ""),
    torch.ops.aten.bitwise_and.Scalar: _clone_scalar_opinfo("bitwise_and", ""),
    torch.ops.aten.bitwise_or.Scalar: _clone_scalar_opinfo("bitwise_or", ""),
    torch.ops.aten.bitwise_xor.Scalar: _clone_scalar_opinfo("bitwise_xor", ""),
    torch.ops.aten.copysign.Scalar: _clone_scalar_opinfo("copysign", ""),
    torch.ops.aten.eq.Scalar: _clone_scalar_opinfo("eq", ""),
    torch.ops.aten.floor_divide.Scalar: _clone_scalar_opinfo("floor_divide", ""),
    torch.ops.aten.fmod.Scalar: _clone_scalar_opinfo("fmod", ""),
    torch.ops.aten.ge.Scalar: _clone_scalar_opinfo("ge", ""),
    torch.ops.aten.gt.Scalar: _clone_scalar_opinfo("gt", ""),
    torch.ops.aten.le.Scalar: _clone_scalar_opinfo("le", ""),
    torch.ops.aten.lt.Scalar: _clone_scalar_opinfo("lt", ""),
    torch.ops.aten.mul.Scalar: _clone_scalar_opinfo("mul", ""),
    torch.ops.aten.ne.Scalar: _clone_scalar_opinfo("ne", ""),
    torch.ops.aten.pow.Scalar: _clone_scalar_opinfo("pow", ""),
    torch.ops.aten.pow.Tensor_Scalar: _clone_tensor_scalar_opinfo("pow", ""),
    torch.ops.aten.remainder.Scalar: _clone_scalar_opinfo("remainder", ""),
    torch.ops.aten.sub.Scalar: _clone_scalar_opinfo("sub", ""),
    torch.ops.aten.where.Scalar: _clone_scalar_opinfo("where", ""),
    torch.ops.aten.elu.default: _lookup_opinfo("nn.functional.elu", ""),
    torch.ops.aten.softplus.default: _lookup_opinfo("nn.functional.softplus", ""),
    torch.ops.aten.round.default: _lookup_opinfo("round", ""),
    torch.ops.aten.selu.default: _lookup_opinfo("nn.functional.selu", ""),
    torch.ops.aten.std.default: _lookup_opinfo("std", ""),
    torch.ops.aten.var.dim: _lookup_opinfo("var", ""),
    torch.ops.aten.norm.ScalarOpt_dim: _lookup_opinfo("norm", ""),
    torch.ops.aten.softmax.int: _lookup_opinfo("softmax", ""),
    torch.ops.aten.log_softmax.int: _lookup_opinfo("log_softmax", ""),
    torch.ops.aten.embedding.default: _lookup_opinfo(
        "nn.functional.embedding", ""
    ),
    torch.ops.aten.addmm.default: _lookup_opinfo("addmm", ""),
    torch.ops.aten.addbmm.default: _lookup_opinfo("addbmm", ""),
    torch.ops.aten.addmv.default: _lookup_opinfo("addmv", ""),
    torch.ops.aten.addr.default: _lookup_opinfo("addr", ""),
    torch.ops.aten.convolution.default: _clone_opinfo(
        _lookup_opinfo("conv2d", ""),
        name="aten.convolution",
        aten_name="convolution",
    ),
}


def _find_opinfo_for_overload(aten_overload):
    if aten_overload in CODEGEN_OPINFO_OVERRIDES:
        return CODEGEN_OPINFO_OVERRIDES[aten_overload]
    base_name = aten_overload._schema.name.split("::")[-1]
    packet = getattr(torch.ops.aten, base_name, None)
    if packet is None or aten_overload.overloadpacket is not packet:
        raise RuntimeError(
            f"Unsupported overload packet for {aten_overload}: expected aten.{base_name}"
        )
    candidates = [op for op in op_db if op.aten_name == base_name]
    if not candidates:
        candidates = [op for op in op_db if op.name == base_name]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one OpInfo entry for {aten_overload}; "
            f"found {len(candidates)}"
        )
    return candidates[0]


ALIASED_CODEGEN_OPS = {
    torch.ops.aten.absolute.default,
    torch.ops.aten.arccos.default,
    torch.ops.aten.arcsin.default,
    torch.ops.aten.arcsinh.default,
    torch.ops.aten.arctan.default,
    torch.ops.aten.any.dim,
    torch.ops.aten.any.dims,
    torch.ops.aten.mean.dim,
    torch.ops.aten.sum.dim_IntList,
}
CODEGEN_OPS_UNDER_TEST = [
    (aten_overload, _find_opinfo_for_overload(aten_overload))
    for aten_overload in CODEGEN_ATEN_OPS
    if aten_overload not in ALIASED_CODEGEN_OPS
]
CODEGEN_OPINFO_LIST = [opinfo for _, opinfo in CODEGEN_OPS_UNDER_TEST]
CODEGEN_OP_TEST_CONFIG = {
    torch.ops.aten.embedding.default: {
        "allowed_dtypes": (torch.float32,),
    },
    torch.ops.aten.copysign.Tensor: {
        "allowed_dtypes": (torch.float32,),
    },
    torch.ops.aten.copysign.Scalar: {
        "allowed_dtypes": (torch.float32,),
    },
    torch.ops.aten.bitwise_and.Scalar: {
        "allowed_dtypes": (torch.int8, torch.int32, torch.bool),
    },
    torch.ops.aten.bitwise_left_shift.Tensor: {
        "allowed_dtypes": (torch.int8, torch.int32),
    },
    torch.ops.aten.bitwise_left_shift_.Tensor: {
        "allowed_dtypes": (torch.int8, torch.int32),
    },
    torch.ops.aten.bitwise_right_shift.Tensor: {
        "allowed_dtypes": (torch.int8, torch.int32),
    },
    torch.ops.aten.bitwise_right_shift_.Tensor: {
        "allowed_dtypes": (torch.int8, torch.int32),
    },
    torch.ops.aten.logical_and.default: {
        "allowed_dtypes": (torch.float32, torch.int8, torch.int32),
    },
    torch.ops.aten.logical_or.default: {
        "allowed_dtypes": (torch.float32, torch.int8, torch.int32),
    },
    torch.ops.aten.logical_xor.default: {
        "allowed_dtypes": (torch.float32, torch.int8, torch.int32),
    },
    torch.ops.aten.clamp.default: {
        "allowed_dtypes": (torch.float32, torch.int8, torch.int32),
    },
    torch.ops.aten.clamp_.default: {
        "allowed_dtypes": (torch.float32, torch.int8, torch.int32),
    },
    torch.ops.aten.where.self: {
        "sample_filter": _broadcastable_sample_filter,
    },
    torch.ops.aten.where.Scalar: {
        "sample_filter": _broadcastable_sample_filter,
    },
    torch.ops.aten.argmax.default: {
        "allowed_dtypes": (torch.float32, torch.int8, torch.int32),
    },
    torch.ops.aten.argmin.default: {
        "allowed_dtypes": (torch.float32, torch.int8, torch.int32),
    },
    torch.ops.aten.softmax.int: {
        "allowed_dtypes": (torch.float32,),
    },
    torch.ops.aten.log_softmax.int: {
        "allowed_dtypes": (torch.float32,),
    },
    torch.ops.aten.log_sigmoid.default: {
        "allowed_dtypes": (torch.float32,),
    },
    torch.ops.aten.gelu.default: {
        "allowed_dtypes": (torch.float32,),
    },
    torch.ops.aten.elu.default: {
        "allowed_dtypes": (torch.float32,),
    },
    torch.ops.aten.leaky_relu.default: {
        "allowed_dtypes": (torch.float32,),
    },
    torch.ops.aten.softplus.default: {
        "allowed_dtypes": (torch.float32,),
    },
    torch.ops.aten.digamma.default: {
        "allowed_dtypes": (torch.float32,),
        "rtol": 3e-5,
        "atol": 0.0,
    },
    torch.ops.aten.mish_.default: {
        "allowed_dtypes": (torch.float32,),
    },
    torch.ops.aten.hardsigmoid.default: {
        "allowed_dtypes": (torch.float32,),
    },
    torch.ops.aten.var.dim: {
        "allowed_dtypes": (torch.float32,),
        "sample_filter": _var_dim_sample_filter,
    },
    torch.ops.aten.norm.ScalarOpt_dim: {
        "allowed_dtypes": (torch.float32,),
        "sample_filter": _norm_dim_sample_filter,
    },
    torch.ops.aten.view.default: {
        "requires_contiguous": True,
    },
    torch.ops.aten.cat.default: {
        "expand_input_list": True,
    },
    torch.ops.aten.relu6.default: {
        "allowed_dtypes": (torch.float32, torch.int8, torch.int32),
    },
    torch.ops.aten.selu.default: {
        "allowed_dtypes": (torch.float32,),
    },
    torch.ops.aten.conv2d.default: {
        "allowed_dtypes": (torch.float32,),
    },
    torch.ops.aten.convolution.default: {
        "allowed_dtypes": (torch.float32,),
        "sample_filter": _convolution_sample_filter,
    },
    torch.ops.aten.avg_pool1d.default: {
        "allowed_dtypes": (torch.float32,),
        "sample_filter": _avg_pool1d_sample_filter,
    },
    torch.ops.aten.avg_pool2d.default: {
        "allowed_dtypes": (torch.float32,),
        "sample_filter": _avg_pool2d_sample_filter,
    },
    torch.ops.aten.max_pool1d.default: {
        "allowed_dtypes": (torch.float32,),
        "sample_filter": _max_pool1d_sample_filter,
    },
    torch.ops.aten.max_pool2d.default: {
        "allowed_dtypes": (torch.float32,),
        "sample_filter": _max_pool2d_sample_filter,
    },
    torch.ops.aten.conv1d.default: {
        "allowed_dtypes": (torch.float32,),
    },
    torch.ops.aten.resize_.default: {
        "sample_filter": _resize_sample_filter,
    },
    torch.ops.aten.cumsum.default: {
        "allowed_dtypes": (torch.float32, torch.int8, torch.int32),
        "sample_filter": _cumsum_sample_filter,
    },
    torch.ops.aten.addmm.default: {
        "allowed_dtypes": (torch.float32,),
    },
    torch.ops.aten.addbmm.default: {
        "allowed_dtypes": (torch.float32,),
        "rtol": 2e-4,
        "atol": 2e-5,
    },
    torch.ops.aten.addmv.default: {
        "allowed_dtypes": (torch.float32,),
        "sample_filter": _addmv_sample_filter,
    },
    torch.ops.aten.addr.default: {
        "equal_nan": True,
        "sample_filter": _addr_sample_filter,
    },
}
DEFAULT_CONSTRAINTS = {
    "allowed_dtypes": (torch.float32, torch.int8, torch.int32, torch.bool),
    "allow_noncontiguous": True,
    "allow_non_tensor_args": True,
    "allow_kwargs": True,
    "expand_input_list": False,
    "max_ndim": None,
    "requires_same_shape": False,
    "requires_contiguous": False,
    "sample_filter": None,
    "rtol": None,
    "atol": None,
    "equal_nan": False,
}


def _constraints_for_codegen(aten_overload):
    constraints = DEFAULT_CONSTRAINTS.copy()
    constraints.update(CODEGEN_OP_TEST_CONFIG.get(aten_overload, {}))
    return constraints


def _sample_to_inputs(sample, constraints):
    inputs = []
    if constraints["expand_input_list"] and isinstance(sample.input, (list, tuple)):
        inputs.extend(sample.input)
    else:
        inputs.append(sample.input)
    inputs.extend(sample.args)
    kwargs = sample.kwargs if constraints["allow_kwargs"] else {}
    return tuple(inputs), kwargs


def _normalize_dim_argument(dim: object) -> object:
    if dim is None:
        return None
    if isinstance(dim, (tuple, list)):
        return tuple(int(operator.index(item)) for item in dim)
    return int(operator.index(dim))


def _compile_codegen_op(aten_overload):
    if aten_overload in {
        torch.ops.aten.amax.default,
        torch.ops.aten.amin.default,
    }:
        def compiled_fn(*args: torch.Tensor, **kwargs) -> torch.Tensor:
            args = list(args)
            if len(args) > 1:
                args[1] = _normalize_dim_argument(args[1])
            if "dim" in kwargs:
                kwargs["dim"] = _normalize_dim_argument(kwargs["dim"])
            return aten_overload(*args, **kwargs)
    elif aten_overload is torch.ops.aten.cat.default:
        def compiled_fn(*args: torch.Tensor, **kwargs) -> torch.Tensor:
            tensors = list(args)
            dim = None
            if tensors and not isinstance(tensors[-1], torch.Tensor):
                dim = tensors.pop()
            if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
                tensors = list(tensors[0])
            if dim is None:
                return aten_overload(tensors, **kwargs)
            if "dim" in kwargs:
                raise TypeError("cat got multiple values for argument 'dim'")
            return aten_overload(tensors, dim, **kwargs)
    else:
        def compiled_fn(*args: torch.Tensor, **kwargs) -> torch.Tensor:
            return aten_overload(*args, **kwargs)

    return torch.compile(compiled_fn, backend=codegen_generic_backend)


def _compile_codegen_inplace_op(aten_overload):
    def compiled_fn(lhs: torch.Tensor, rhs: torch.Tensor | None = None) -> torch.Tensor:
        if rhs is None:
            return aten_overload(lhs)
        return aten_overload(lhs, rhs)

    return torch.compile(compiled_fn, backend=codegen_generic_backend)


def _sanitize_inplace_inputs(
    aten_overload: torch._ops.OpOverload, lhs: torch.Tensor, rhs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    name = aten_overload._schema.name.split("::")[-1]
    unit_range_ops = {"acos_", "arccos_", "asin_", "arcsin_", "atanh_", "erfinv_"}
    ge1_ops = {"acosh_", "arccosh_"}
    positive_ops = {"digamma_", "lgamma_", "log_", "log10_", "log2_", "rsqrt_", "sqrt_"}
    log1p_ops = {"log1p_"}
    logit_ops = {"logit_"}
    reciprocal_ops = {"reciprocal_"}
    pow_ops = {"pow_"}
    rhs_nonzero_ops = {"div_", "floor_divide_", "fmod_", "remainder_"}
    xlogy_ops = {"xlogy_"}
    ldexp_ops = {"ldexp_"}

    if name in unit_range_ops:
        lhs = lhs.tanh()
    if name in ge1_ops:
        lhs = lhs.abs() + 1
    if name in positive_ops:
        lhs = lhs.abs() + 0.1
    if name in log1p_ops:
        lhs = lhs.abs()
    if name in logit_ops:
        lhs = lhs.sigmoid().clamp(1e-4, 1 - 1e-4)
    if name in reciprocal_ops:
        lhs = lhs.sign() * (lhs.abs() + 0.1)
    if name in pow_ops:
        lhs = lhs.abs() + 0.1
    if name in rhs_nonzero_ops:
        rhs = rhs.sign() * (rhs.abs() + 0.1)
    if name in xlogy_ops:
        rhs = rhs.abs() + 0.1
    if name in ldexp_ops:
        rhs = rhs.round()

    return lhs, rhs


def _reference_for_dtype(
    aten_overload: torch._ops.OpOverload,
    inputs: tuple[object, ...],
    kwargs: dict[str, object],
    dtype: torch.dtype,
) -> torch.Tensor:
    if dtype not in (torch.int8, torch.int32, torch.bool):
        return aten_overload(*inputs, **kwargs)
    try:
        expected = aten_overload(*inputs, **kwargs)
    except Exception:
        float_inputs = tuple(
            arg.to(torch.float32) if isinstance(arg, torch.Tensor) else arg
            for arg in inputs
        )
        expected = aten_overload(*float_inputs, **kwargs)
    return expected


class TestCodegenOpInfo(TestCase):
    @ops(CODEGEN_OPINFO_LIST)
    def test_codegen_backend_matches_eager(self, device, dtype, op):
        aten_overload = next(
            (
                overload
                for overload, opinfo in CODEGEN_OPS_UNDER_TEST
                if opinfo is op
            ),
            None,
        )
        if aten_overload is None:
            raise RuntimeError(f"Missing overload mapping for OpInfo {op.name}")
        constraints = _constraints_for_codegen(aten_overload)
        allowed_dtypes = constraints["allowed_dtypes"]
        if allowed_dtypes is not None and dtype not in allowed_dtypes:
            pytest.skip("dtype not supported by test constraints")
        compiled = _compile_codegen_op(aten_overload)
        for sample in _iter_supported_samples(op, device, dtype, constraints):
            inputs, kwargs = _sample_to_inputs(sample, constraints)
            try:
                expected = _reference_for_dtype(aten_overload, inputs, kwargs, dtype)
            except Exception:
                continue
            result = compiled(*inputs, **kwargs)
            if result.dtype is not expected.dtype:
                expected = expected.to(result.dtype)
            compare_kwargs = {"equal_nan": dtype in (torch.int8, torch.int32)}
            if constraints.get("equal_nan"):
                compare_kwargs["equal_nan"] = True
            if constraints["rtol"] is not None or constraints["atol"] is not None:
                compare_kwargs["rtol"] = constraints["rtol"] or 0.0
                compare_kwargs["atol"] = constraints["atol"] or 0.0
            torch.testing.assert_close(result, expected, **compare_kwargs)


class TestCodegenAliasedOps(TestCase):
    def test_codegen_arccosh_matches_eager(self):
        aten_overload = torch.ops.aten.arccosh.default
        compiled = _compile_codegen_op(aten_overload)
        for dtype in (torch.float32,):
            inputs = (torch.rand(2, 3, dtype=dtype) + 1.0,)
            expected = aten_overload(*inputs)
            result = compiled(*inputs)
            torch.testing.assert_close(result, expected)

    def test_codegen_aliases_match_eager(self):
        aliased_ops = [
            torch.ops.aten.absolute.default,
            torch.ops.aten.arccos.default,
            torch.ops.aten.arcsin.default,
            torch.ops.aten.arcsinh.default,
            torch.ops.aten.arctan.default,
        ]
        for aten_overload in aliased_ops:
            compiled = _compile_codegen_op(aten_overload)
            for dtype in (torch.float32,):
                if aten_overload is torch.ops.aten.absolute.default:
                    inputs = (torch.randn(2, 3, dtype=dtype),)
                elif aten_overload in {
                    torch.ops.aten.arccos.default,
                    torch.ops.aten.arcsin.default,
                }:
                    inputs = (torch.rand(2, 3, dtype=dtype) * 2 - 1,)
                else:
                    inputs = (torch.randn(2, 3, dtype=dtype),)
                expected = aten_overload(*inputs)
                result = compiled(*inputs)
                torch.testing.assert_close(result, expected)

    def test_codegen_any_overloads_match_eager(self):
        tensor = torch.tensor([[False, True, False], [False, False, False]])

        for aten_overload, dims in (
            (torch.ops.aten.any.dim, 1),
            (torch.ops.aten.any.dims, (0, 1)),
        ):
            compiled = _compile_codegen_op(aten_overload)
            expected = aten_overload(tensor, dims, False)
            result = compiled(tensor, dims, False)
            torch.testing.assert_close(result, expected)

    def test_codegen_sum_dim_overload_matches_eager(self):
        tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        aten_overload = torch.ops.aten.sum.dim_IntList
        compiled = _compile_codegen_op(aten_overload)
        dims = (1,)
        expected = aten_overload(tensor, dims, False)
        result = compiled(tensor, dims, False)
        torch.testing.assert_close(result, expected)


class TestCodegenAdditionalOps(TestCase):
    def test_codegen__softmax_matches_eager(self):
        aten_overload = torch.ops.aten._softmax.default
        compiled = _compile_codegen_op(aten_overload)
        inputs = (torch.rand(2, 3, dtype=torch.float32), 1, False)
        expected = aten_overload(*inputs)
        result = compiled(*inputs)
        torch.testing.assert_close(result, expected)

    def test_codegen__to_copy_matches_eager(self):
        aten_overload = torch.ops.aten._to_copy.default
        compiled = _compile_codegen_op(aten_overload)
        inputs = (torch.rand(2, 3, dtype=torch.float32),)
        expected = aten_overload(*inputs, non_blocking=False)
        result = compiled(*inputs, non_blocking=False)
        torch.testing.assert_close(result, expected)

    def test_codegen_adaptive_avg_pool1d_matches_eager(self):
        aten_overload = torch.ops.aten.adaptive_avg_pool1d.default
        compiled = _compile_codegen_op(aten_overload)
        inputs = (torch.rand(2, 3, 6, dtype=torch.float32), 3)
        expected = aten_overload(*inputs)
        result = compiled(*inputs)
        torch.testing.assert_close(result, expected)

    def test_codegen_adaptive_avg_pool2d_matches_eager(self):
        aten_overload = torch.ops.aten._adaptive_avg_pool2d.default
        compiled = _compile_codegen_op(aten_overload)
        inputs = (torch.rand(2, 3, 6, 8, dtype=torch.float32), (3, 4))
        expected = aten_overload(*inputs)
        result = compiled(*inputs)
        torch.testing.assert_close(result, expected)

    def test_codegen_native_batch_norm_no_training_matches_eager(self):
        def compiled_fn(
            input_tensor: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
            running_mean: torch.Tensor,
            running_var: torch.Tensor,
            momentum: float,
            eps: float,
        ) -> torch.Tensor:
            return torch.ops.aten._native_batch_norm_legit_no_training.default(
                input_tensor,
                weight,
                bias,
                running_mean,
                running_var,
                momentum,
                eps,
            )[0]

        compiled = torch.compile(compiled_fn, backend=codegen_generic_backend)
        input_tensor = torch.rand(2, 3, 4, dtype=torch.float32)
        weight = torch.rand(3, dtype=torch.float32)
        bias = torch.rand(3, dtype=torch.float32)
        running_mean = torch.rand(3, dtype=torch.float32)
        running_var = torch.rand(3, dtype=torch.float32) + 1.0
        expected = torch.ops.aten._native_batch_norm_legit_no_training.default(
            input_tensor,
            weight,
            bias,
            running_mean,
            running_var,
            0.1,
            1e-5,
        )[0]
        result = compiled(
            input_tensor, weight, bias, running_mean, running_var, 0.1, 1e-5
        )
        torch.testing.assert_close(result, expected)

    def test_codegen_pdist_forward_matches_eager(self):
        aten_overload = torch.ops.aten._pdist_forward.default
        compiled = _compile_codegen_op(aten_overload)
        inputs = (torch.rand(4, 5, dtype=torch.float32), 2.0)
        expected = aten_overload(*inputs)
        result = compiled(*inputs)
        torch.testing.assert_close(result, expected)


class TestCodegenInplaceOps(TestCase):
    def test_codegen_backend_inplace_ops(self):
        device = torch.device("cpu")
        sample_shapes = (
            ((2, 3), (2, 3)),
        )

        for aten_overload in INPLACE_ATEN_OPS:
            constraints = _constraints_for_codegen(aten_overload)
            allowed_dtypes = constraints["allowed_dtypes"]
            dtype = (
                torch.float32
                if torch.float32 in allowed_dtypes
                else allowed_dtypes[0]
            )
            required_args = [
                arg
                for arg in aten_overload._schema.arguments
                if not arg.kwarg_only and not arg.has_default_value()
            ]
            requires_rhs = len(required_args) > 1
            compiled = _compile_codegen_inplace_op(aten_overload)
            for lhs_shape, rhs_shape in sample_shapes:
                op_name = aten_overload._schema.name.split("::")[-1]
                if dtype in (torch.int8, torch.int32):
                    if op_name in {"bitwise_left_shift_", "bitwise_right_shift_"}:
                        low, high = 0, 3
                    else:
                        low, high = -5, 5
                    lhs = torch.randint(
                        low, high, lhs_shape, device=device, dtype=dtype
                    )
                    rhs = torch.randint(
                        low, high, rhs_shape, device=device, dtype=dtype
                    )
                elif dtype is torch.bool:
                    lhs = torch.randint(
                        0, 2, lhs_shape, device=device, dtype=dtype
                    )
                    rhs = torch.randint(
                        0, 2, rhs_shape, device=device, dtype=dtype
                    )
                else:
                    lhs = torch.randn(lhs_shape, device=device, dtype=dtype)
                    rhs = torch.randn(rhs_shape, device=device, dtype=dtype)
                lhs, rhs = _sanitize_inplace_inputs(aten_overload, lhs, rhs)

                expected = lhs.clone()
                if requires_rhs:
                    args = (expected, rhs)
                else:
                    args = (expected,)
                try:
                    expected_result = aten_overload(*args)
                except Exception:
                    continue

                compiled_lhs = lhs.clone()
                if requires_rhs:
                    compiled_args = (compiled_lhs, rhs)
                else:
                    compiled_args = (compiled_lhs,)
                compiled_result = compiled(*compiled_args)

                torch.testing.assert_close(compiled_result, expected_result)
                torch.testing.assert_close(compiled_lhs, expected)


class TestCodegenReductionOps(TestCase):
    def test_codegen_sum_default_cases(self):
        cases = [
            torch.randn(2, 3, dtype=torch.float32),
            torch.randn(4, dtype=torch.float32),
            torch.tensor(3.5, dtype=torch.float32),
            torch.randn(0, 3, dtype=torch.float32),
            torch.randn(2, 3, dtype=torch.float32).t(),
        ]
        for tensor in cases:
            compiled = _compile_codegen_op(torch.ops.aten.sum.default)
            result = compiled(tensor)
            expected = torch.ops.aten.sum.default(tensor)
            assert result.shape == torch.Size([])
            torch.testing.assert_close(result, expected)

    def test_codegen_prod_default_cases(self):
        cases = [
            torch.randn(2, 3, dtype=torch.float32),
            torch.randn(4, dtype=torch.float32),
            torch.tensor(3.5, dtype=torch.float32),
            torch.randn(0, 3, dtype=torch.float32),
            torch.randn(2, 3, dtype=torch.float32).t(),
        ]
        for tensor in cases:
            compiled = _compile_codegen_op(torch.ops.aten.prod.default)
            result = compiled(tensor)
            expected = torch.ops.aten.prod.default(tensor)
            assert result.shape == torch.Size([])
            torch.testing.assert_close(result, expected)

    def test_codegen_mean_default_cases(self):
        cases = [
            torch.randn(2, 3, dtype=torch.float32),
            torch.randn(4, dtype=torch.float32),
            torch.tensor(3.5, dtype=torch.float32),
            torch.randn(0, 3, dtype=torch.float32),
            torch.randn(2, 3, dtype=torch.float32).t(),
        ]
        for tensor in cases:
            compiled = _compile_codegen_op(torch.ops.aten.mean.default)
            result = compiled(tensor)
            expected = torch.ops.aten.mean.default(tensor)
            assert result.shape == torch.Size([])
            torch.testing.assert_close(result, expected, equal_nan=True)

    def test_codegen_std_default_cases(self):
        cases = [
            torch.randn(2, 3, dtype=torch.float32),
            torch.randn(4, dtype=torch.float32),
            torch.tensor(3.5, dtype=torch.float32),
            torch.randn(0, 3, dtype=torch.float32),
            torch.randn(2, 3, dtype=torch.float32).t(),
        ]
        compiled = _compile_codegen_op(torch.ops.aten.std.default)
        for tensor in cases:
            result = compiled(tensor)
            expected = torch.ops.aten.std.default(tensor)
            assert result.shape == torch.Size([])
            torch.testing.assert_close(result, expected, equal_nan=True)

        for tensor in cases:
            result = compiled(tensor, False)
            expected = torch.ops.aten.std.default(tensor, False)
            assert result.shape == torch.Size([])
            torch.testing.assert_close(result, expected, equal_nan=True)


instantiate_device_type_tests(TestCodegenOpInfo, globals(), only_for="cpu")
