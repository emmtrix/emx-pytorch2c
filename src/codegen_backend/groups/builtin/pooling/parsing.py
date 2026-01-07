from __future__ import annotations

import numbers
from typing import Tuple

import torch
import torch.fx

from codegen_backend.analysis_helpers import (
    error_kwarg_specified_once,
    normalize_param,
)
from codegen_backend.errors import CodegenBackendError
from codegen_backend.param_normalize import normalize_int_or_pair, normalize_int_or_tuple


def parse_max_pool1d_args(
    node: torch.fx.Node,
) -> Tuple[torch.fx.Node, object, object, object, object, object]:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) < 2 or len(args) > 6:
        raise CodegenBackendError("codegen max_pool1d expects pooling arguments")
    input_arg = args[0]
    kernel_size = args[1]
    stride = None
    padding = 0
    dilation = 1
    ceil_mode = False
    remaining = args[2:]
    if len(remaining) >= 1:
        stride = remaining[0]
    if len(remaining) >= 2:
        padding = remaining[1]
    if len(remaining) >= 3:
        dilation = remaining[2]
    if len(remaining) >= 4:
        ceil_mode = remaining[3]
    if kwargs:
        extra = set(kwargs) - {
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "ceil_mode",
        }
        if extra:
            raise CodegenBackendError(
                f"codegen max_pool1d got unexpected kwargs: {sorted(extra)}"
            )
        if "kernel_size" in kwargs:
            kernel_size = kwargs["kernel_size"]
        if "stride" in kwargs:
            stride = kwargs["stride"]
        if "padding" in kwargs:
            padding = kwargs["padding"]
        if "dilation" in kwargs:
            dilation = kwargs["dilation"]
        if "ceil_mode" in kwargs:
            ceil_mode = kwargs["ceil_mode"]
    return input_arg, kernel_size, stride, padding, dilation, ceil_mode


def parse_avg_pool1d_args(
    node: torch.fx.Node,
) -> Tuple[torch.fx.Node, object, object, object, object, object, object]:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) < 2 or len(args) > 7:
        raise CodegenBackendError("codegen avg_pool1d expects pooling arguments")
    input_arg = args[0]
    kernel_size = args[1]
    stride = None
    padding = 0
    ceil_mode = False
    count_include_pad = False
    divisor_override = None
    remaining = args[2:]
    if len(remaining) >= 1:
        stride = remaining[0]
    if len(remaining) >= 2:
        padding = remaining[1]
    if len(remaining) >= 3:
        ceil_mode = remaining[2]
    if len(remaining) >= 4:
        count_include_pad = remaining[3]
    if len(remaining) >= 5:
        divisor_override = remaining[4]
    if kwargs:
        extra = set(kwargs) - {
            "kernel_size",
            "stride",
            "padding",
            "ceil_mode",
            "count_include_pad",
            "divisor_override",
        }
        if extra:
            raise CodegenBackendError(
                f"codegen avg_pool1d got unexpected kwargs: {sorted(extra)}"
            )
        if "kernel_size" in kwargs:
            kernel_size = kwargs["kernel_size"]
        if "stride" in kwargs:
            stride = kwargs["stride"]
        if "padding" in kwargs:
            padding = kwargs["padding"]
        if "ceil_mode" in kwargs:
            ceil_mode = kwargs["ceil_mode"]
        if "count_include_pad" in kwargs:
            count_include_pad = kwargs["count_include_pad"]
        if "divisor_override" in kwargs:
            divisor_override = kwargs["divisor_override"]
    return (
        input_arg,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )


def parse_adaptive_avg_pool1d_args(
    node: torch.fx.Node,
) -> Tuple[torch.fx.Node, object]:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) < 2 or len(args) > 2:
        raise CodegenBackendError(
            "codegen adaptive_avg_pool1d expects input and output_size"
        )
    input_arg = args[0]
    output_size = args[1]
    if kwargs:
        if "output_size" in kwargs:
            if len(args) > 1:
                raise error_kwarg_specified_once(
                    "adaptive_avg_pool1d", "output_size"
                )
            output_size = kwargs["output_size"]
        extra = set(kwargs) - {"output_size"}
        if extra:
            raise CodegenBackendError(
                "codegen adaptive_avg_pool1d got unexpected kwargs: "
                f"{sorted(extra)}"
            )
    return input_arg, output_size


def parse_adaptive_avg_pool2d_args(
    node: torch.fx.Node,
) -> Tuple[torch.fx.Node, object]:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) < 2 or len(args) > 2:
        raise CodegenBackendError(
            "codegen adaptive_avg_pool2d expects input and output_size"
        )
    input_arg = args[0]
    output_size = args[1]
    if kwargs:
        if "output_size" in kwargs:
            if len(args) > 1:
                raise error_kwarg_specified_once(
                    "adaptive_avg_pool2d", "output_size"
                )
            output_size = kwargs["output_size"]
        extra = set(kwargs) - {"output_size"}
        if extra:
            raise CodegenBackendError(
                "codegen adaptive_avg_pool2d got unexpected kwargs: "
                f"{sorted(extra)}"
            )
    return input_arg, output_size


def parse_adaptive_avg_pool3d_args(
    node: torch.fx.Node,
) -> Tuple[torch.fx.Node, object]:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) < 2 or len(args) > 2:
        raise CodegenBackendError(
            "codegen adaptive_avg_pool3d expects input and output_size"
        )
    input_arg = args[0]
    output_size = args[1]
    if kwargs:
        if "output_size" in kwargs:
            if len(args) > 1:
                raise error_kwarg_specified_once(
                    "adaptive_avg_pool3d", "output_size"
                )
            output_size = kwargs["output_size"]
        extra = set(kwargs) - {"output_size"}
        if extra:
            raise CodegenBackendError(
                "codegen adaptive_avg_pool3d got unexpected kwargs: "
                f"{sorted(extra)}"
            )
    return input_arg, output_size


def parse_adaptive_avg_pool2d_backward_args(
    node: torch.fx.Node,
) -> Tuple[torch.fx.Node, torch.fx.Node]:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) != 2:
        raise CodegenBackendError(
            "codegen adaptive_avg_pool2d_backward expects grad_output and input"
        )
    if kwargs:
        raise CodegenBackendError(
            "codegen adaptive_avg_pool2d_backward expects no keyword arguments"
        )
    grad_output = args[0]
    input_arg = args[1]
    return grad_output, input_arg


def parse_max_pool2d_args(
    node: torch.fx.Node,
) -> Tuple[torch.fx.Node, object, object, object, object, object]:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) > 6:
        raise CodegenBackendError("codegen max_pool2d expects pooling arguments")
    input_arg = args[0] if len(args) > 0 else None
    kernel_size = None
    stride = None
    padding = 0
    dilation = 1
    ceil_mode = False
    remaining = args[1:]
    has_kernel_size = len(remaining) >= 1
    has_stride = len(remaining) >= 2
    has_padding = len(remaining) >= 3
    has_dilation = len(remaining) >= 4
    has_ceil_mode = len(remaining) >= 5
    if has_kernel_size:
        kernel_size = remaining[0]
    if has_stride:
        stride = remaining[1]
    if has_padding:
        padding = remaining[2]
    if has_dilation:
        dilation = remaining[3]
    if has_ceil_mode:
        ceil_mode = remaining[4]
    if kwargs:
        extra = set(kwargs) - {
            "input",
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "ceil_mode",
        }
        if extra:
            raise CodegenBackendError(
                f"codegen max_pool2d got unexpected kwargs: {sorted(extra)}"
            )
        if "input" in kwargs:
            if input_arg is not None:
                raise error_kwarg_specified_once("max_pool2d", "input")
            input_arg = kwargs["input"]
        if "kernel_size" in kwargs:
            if has_kernel_size:
                raise error_kwarg_specified_once("max_pool2d", "kernel_size")
            kernel_size = kwargs["kernel_size"]
        if "stride" in kwargs:
            if has_stride:
                raise error_kwarg_specified_once("max_pool2d", "stride")
            stride = kwargs["stride"]
        if "padding" in kwargs:
            if has_padding:
                raise error_kwarg_specified_once("max_pool2d", "padding")
            padding = kwargs["padding"]
        if "dilation" in kwargs:
            if has_dilation:
                raise error_kwarg_specified_once("max_pool2d", "dilation")
            dilation = kwargs["dilation"]
        if "ceil_mode" in kwargs:
            if has_ceil_mode:
                raise error_kwarg_specified_once("max_pool2d", "ceil_mode")
            ceil_mode = kwargs["ceil_mode"]
    if input_arg is None or kernel_size is None:
        raise CodegenBackendError("codegen max_pool2d expects pooling arguments")
    return input_arg, kernel_size, stride, padding, dilation, ceil_mode


def parse_avg_pool2d_args(
    node: torch.fx.Node,
) -> Tuple[torch.fx.Node, object, object, object, object, object, object, object]:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) < 2 or len(args) > 7:
        raise CodegenBackendError("codegen avg_pool2d expects pooling arguments")
    input_arg = args[0]
    kernel_size = args[1]
    stride = None
    padding = 0
    ceil_mode = False
    count_include_pad = True
    divisor_override = None
    remaining = args[2:]
    if len(remaining) >= 1:
        stride = remaining[0]
    if len(remaining) >= 2:
        padding = remaining[1]
    if len(remaining) >= 3:
        ceil_mode = remaining[2]
    if len(remaining) >= 4:
        count_include_pad = remaining[3]
    if len(remaining) >= 5:
        divisor_override = remaining[4]
    if kwargs:
        extra = set(kwargs) - {
            "kernel_size",
            "stride",
            "padding",
            "ceil_mode",
            "count_include_pad",
            "divisor_override",
        }
        if extra:
            raise CodegenBackendError(
                f"codegen avg_pool2d got unexpected kwargs: {sorted(extra)}"
            )
        if "kernel_size" in kwargs:
            kernel_size = kwargs["kernel_size"]
        if "stride" in kwargs:
            stride = kwargs["stride"]
        if "padding" in kwargs:
            padding = kwargs["padding"]
        if "ceil_mode" in kwargs:
            ceil_mode = kwargs["ceil_mode"]
        if "count_include_pad" in kwargs:
            count_include_pad = kwargs["count_include_pad"]
        if "divisor_override" in kwargs:
            divisor_override = kwargs["divisor_override"]
    if isinstance(kernel_size, torch.fx.Node):
        raise CodegenBackendError("codegen avg_pool2d expects kernel_size to be a list")
    if kernel_size is None:
        raise CodegenBackendError(
            "codegen avg_pool2d expects kernel_size and stride"
        )
    kernel_value = normalize_param(kernel_size)
    if stride is None:
        stride_value = kernel_value
    elif isinstance(stride, torch.fx.Node):
        raise CodegenBackendError("codegen avg_pool2d expects stride to be a list")
    else:
        stride_value = normalize_param(stride)
    if isinstance(padding, torch.fx.Node):
        raise CodegenBackendError("codegen avg_pool2d expects padding to be a list")
    padding_value = normalize_param(padding)
    if kernel_value[0] <= 0 or kernel_value[1] <= 0:
        raise CodegenBackendError("codegen avg_pool2d expects positive kernel sizes")
    if stride_value[0] <= 0 or stride_value[1] <= 0:
        raise CodegenBackendError(
            "codegen avg_pool2d expects positive kernel and stride"
        )
    if isinstance(count_include_pad, torch.fx.Node):
        raise CodegenBackendError(
            "codegen avg_pool2d expects count_include_pad to be a bool"
        )
    if not isinstance(count_include_pad, bool):
        raise CodegenBackendError(
            "codegen avg_pool2d expects count_include_pad to be a bool"
        )
    if divisor_override is not None:
        if isinstance(divisor_override, torch.fx.Node):
            raise CodegenBackendError(
                "codegen avg_pool2d expects divisor_override to be a number"
            )
        if not isinstance(divisor_override, numbers.Integral):
            raise CodegenBackendError(
                "codegen avg_pool2d expects divisor_override to be an int"
            )
    return (
        input_arg,
        kernel_value,
        stride_value,
        padding_value,
        ceil_mode,
        count_include_pad,
        divisor_override,
        None,
    )


__all__ = [
    "parse_adaptive_avg_pool1d_args",
    "parse_adaptive_avg_pool2d_args",
    "parse_adaptive_avg_pool2d_backward_args",
    "parse_adaptive_avg_pool3d_args",
    "parse_avg_pool1d_args",
    "parse_avg_pool2d_args",
    "parse_max_pool1d_args",
    "parse_max_pool2d_args",
]
