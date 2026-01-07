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
    if isinstance(kernel_size, torch.fx.Node) or isinstance(
        padding, torch.fx.Node
    ):
        raise CodegenBackendError(
            "codegen max_pool1d expects constant kernel, padding, and ceil_mode"
        )
    if stride is None:
        stride = kernel_size
    if isinstance(stride, torch.fx.Node) or isinstance(
        dilation, torch.fx.Node
    ):
        raise CodegenBackendError(
            "codegen max_pool1d expects constant stride, padding, and dilation"
        )
    if isinstance(ceil_mode, torch.fx.Node):
        raise CodegenBackendError(
            "codegen max_pool1d expects ceil_mode to be a bool"
        )
    if ceil_mode:
        raise CodegenBackendError(
            "codegen max_pool1d expects ceil_mode to be False"
        )
    padding_value = normalize_param(
        normalize_int_or_tuple, "padding", padding, 1
    )
    if padding_value[0] < 0:
        raise CodegenBackendError(
            "codegen max_pool1d expects padding to be non-negative"
        )
    kernel_value = normalize_param(
        normalize_int_or_tuple, "kernel_size", kernel_size, 1
    )
    stride_value = normalize_param(
        normalize_int_or_tuple, "stride", stride, 1
    )
    dilation_value = normalize_param(
        normalize_int_or_tuple, "dilation", dilation, 1
    )
    if kernel_value[0] <= 0 or stride_value[0] <= 0 or dilation_value[0] <= 0:
        raise CodegenBackendError(
            "codegen max_pool1d expects positive kernel, stride, and dilation"
        )
    return input_arg, kernel_value, stride_value, padding_value, dilation_value, ceil_mode


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
    if isinstance(kernel_size, torch.fx.Node) or isinstance(
        padding, torch.fx.Node
    ):
        raise CodegenBackendError(
            "codegen avg_pool1d expects constant kernel, padding, and ceil_mode"
        )
    if stride is None:
        stride = kernel_size
    if isinstance(stride, torch.fx.Node):
        raise CodegenBackendError(
            "codegen avg_pool1d expects constant stride, padding, and dilation"
        )
    if isinstance(ceil_mode, torch.fx.Node):
        raise CodegenBackendError(
            "codegen avg_pool1d expects ceil_mode to be a bool"
        )
    if ceil_mode:
        raise CodegenBackendError(
            "codegen avg_pool1d expects ceil_mode to be False"
        )
    padding_value = normalize_param(
        normalize_int_or_tuple, "padding", padding, 1
    )
    if padding_value[0] < 0:
        raise CodegenBackendError(
            "codegen avg_pool1d expects padding to be non-negative"
        )
    kernel_value = normalize_param(
        normalize_int_or_tuple, "kernel_size", kernel_size, 1
    )
    stride_value = normalize_param(
        normalize_int_or_tuple, "stride", stride, 1
    )
    if kernel_value[0] <= 0 or stride_value[0] <= 0:
        raise CodegenBackendError(
            "codegen avg_pool1d expects positive kernel and stride"
        )
    if isinstance(count_include_pad, bool):
        count_include_pad_value = count_include_pad
    elif isinstance(count_include_pad, numbers.Integral):
        count_include_pad_value = bool(count_include_pad)
    else:
        raise CodegenBackendError(
            "codegen avg_pool1d expects count_include_pad to be a bool"
        )
    if divisor_override is not None:
        if isinstance(divisor_override, torch.fx.Node):
            raise CodegenBackendError(
                "codegen avg_pool1d expects divisor_override to be a number"
            )
        if not isinstance(divisor_override, numbers.Integral):
            raise CodegenBackendError(
                "codegen avg_pool1d expects divisor_override to be an int"
            )
    return (
        input_arg,
        kernel_value,
        stride_value,
        padding_value,
        ceil_mode,
        count_include_pad_value,
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
        extra = set(kwargs)
        if extra:
            raise CodegenBackendError(
                f"codegen adaptive_avg_pool1d got unexpected kwargs: {sorted(extra)}"
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
        extra = set(kwargs)
        if extra:
            raise CodegenBackendError(
                f"codegen adaptive_avg_pool2d got unexpected kwargs: {sorted(extra)}"
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
        extra = set(kwargs)
        if extra:
            raise CodegenBackendError(
                f"codegen adaptive_avg_pool3d got unexpected kwargs: {sorted(extra)}"
            )
    return input_arg, output_size


def parse_adaptive_avg_pool2d_backward_args(
    node: torch.fx.Node,
) -> Tuple[torch.fx.Node, torch.fx.Node]:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) < 2 or len(args) > 2:
        raise CodegenBackendError(
            "codegen adaptive_avg_pool2d_backward expects grad_output and input"
        )
    grad_output = args[0]
    input_arg = args[1]
    if kwargs:
        extra = set(kwargs)
        if extra:
            raise CodegenBackendError(
                f"codegen adaptive_avg_pool2d_backward got unexpected kwargs: {sorted(extra)}"
            )
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
    if len(args) < 2 or len(args) > 8:
        raise CodegenBackendError("codegen avg_pool2d expects pooling arguments")
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
    if isinstance(kernel_size, torch.fx.Node) or isinstance(
        padding, torch.fx.Node
    ):
        raise CodegenBackendError(
            "codegen avg_pool2d expects constant kernel, padding, and ceil_mode"
        )
    if stride is None:
        stride = kernel_size
    if isinstance(stride, torch.fx.Node):
        raise CodegenBackendError(
            "codegen avg_pool2d expects constant stride, padding, and dilation"
        )
    if isinstance(ceil_mode, torch.fx.Node):
        raise CodegenBackendError(
            "codegen avg_pool2d expects ceil_mode to be a bool"
        )
    if ceil_mode:
        raise CodegenBackendError(
            "codegen avg_pool2d expects ceil_mode to be False"
        )
    padding_value = normalize_param(
        normalize_int_or_pair, "padding", padding
    )
    if padding_value[0] < 0 or padding_value[1] < 0:
        raise CodegenBackendError(
            "codegen avg_pool2d expects padding to be non-negative"
        )
    kernel_value = normalize_param(
        normalize_int_or_pair, "kernel_size", kernel_size
    )
    stride_value = normalize_param(
        normalize_int_or_pair, "stride", stride
    )
    if (
        kernel_value[0] <= 0
        or kernel_value[1] <= 0
        or stride_value[0] <= 0
        or stride_value[1] <= 0
    ):
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
