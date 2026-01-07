from __future__ import annotations

import torch.fx

from codegen_backend.errors import CodegenBackendError


def parse_conv2d_args(
    node: torch.fx.Node,
) -> tuple[
    torch.fx.Node,
    torch.fx.Node,
    object,
    object,
    object,
    object,
    object,
    object,
    object,
]:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) < 2 or len(args) > 9:
        raise CodegenBackendError("codegen conv2d expects convolution arguments")
    input_arg = args[0]
    weight_arg = args[1]
    bias = None
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    transposed = False
    output_padding: object = (0, 0)
    remaining = args[2:]
    if len(args) <= 7:
        if len(remaining) >= 1:
            bias = remaining[0]
        if len(remaining) >= 2:
            stride = remaining[1]
        if len(remaining) >= 3:
            padding = remaining[2]
        if len(remaining) >= 4:
            dilation = remaining[3]
        if len(remaining) >= 5:
            groups = remaining[4]
    elif len(args) in {8, 9}:
        if len(args) == 8:
            (
                stride,
                padding,
                dilation,
                transposed,
                output_padding,
                groups,
            ) = remaining
        else:
            (
                bias,
                stride,
                padding,
                dilation,
                transposed,
                output_padding,
                groups,
            ) = remaining

    if kwargs:
        extra = set(kwargs) - {
            "bias",
            "stride",
            "padding",
            "dilation",
            "groups",
            "transposed",
            "output_padding",
        }
        if extra:
            raise CodegenBackendError(
                f"codegen conv2d got unexpected kwargs: {sorted(extra)}"
            )
        if "bias" in kwargs:
            bias = kwargs["bias"]
        if "stride" in kwargs:
            stride = kwargs["stride"]
        if "padding" in kwargs:
            padding = kwargs["padding"]
        if "dilation" in kwargs:
            dilation = kwargs["dilation"]
        if "groups" in kwargs:
            groups = kwargs["groups"]
        if "transposed" in kwargs:
            transposed = kwargs["transposed"]
        if "output_padding" in kwargs:
            output_padding = kwargs["output_padding"]

    return (
        input_arg,
        weight_arg,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )


def parse_conv1d_args(
    node: torch.fx.Node,
) -> tuple[
    torch.fx.Node,
    torch.fx.Node,
    object,
    object,
    object,
    object,
    object,
]:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) < 2 or len(args) > 7:
        raise CodegenBackendError("codegen conv1d expects convolution arguments")
    input_arg = args[0]
    weight_arg = args[1]
    bias = None
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    remaining = args[2:]
    if len(remaining) >= 1:
        bias = remaining[0]
    if len(remaining) >= 2:
        stride = remaining[1]
    if len(remaining) >= 3:
        padding = remaining[2]
    if len(remaining) >= 4:
        dilation = remaining[3]
    if len(remaining) >= 5:
        groups = remaining[4]

    if kwargs:
        extra = set(kwargs) - {"bias", "stride", "padding", "dilation", "groups"}
        if extra:
            raise CodegenBackendError(
                f"codegen conv1d got unexpected kwargs: {sorted(extra)}"
            )
        if "bias" in kwargs:
            bias = kwargs["bias"]
        if "stride" in kwargs:
            stride = kwargs["stride"]
        if "padding" in kwargs:
            padding = kwargs["padding"]
        if "dilation" in kwargs:
            dilation = kwargs["dilation"]
        if "groups" in kwargs:
            groups = kwargs["groups"]

    return (input_arg, weight_arg, bias, stride, padding, dilation, groups)


def parse_col2im_args(
    node: torch.fx.Node,
) -> tuple[torch.fx.Node, object, object, object, object, object]:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) < 1 or len(args) > 6:
        raise CodegenBackendError(
            "codegen col2im expects input, output_size, kernel_size, dilation, padding, and stride"
        )
    input_arg = args[0] if len(args) >= 1 else None
    output_size = args[1] if len(args) >= 2 else None
    kernel_size = args[2] if len(args) >= 3 else None
    dilation = args[3] if len(args) >= 4 else None
    padding = args[4] if len(args) >= 5 else None
    stride = args[5] if len(args) >= 6 else None
    if kwargs:
        extra = set(kwargs) - {
            "output_size",
            "kernel_size",
            "dilation",
            "padding",
            "stride",
        }
        if extra:
            raise CodegenBackendError(
                f"codegen col2im got unexpected kwargs: {sorted(extra)}"
            )
        if "output_size" in kwargs:
            output_size = kwargs["output_size"]
        if "kernel_size" in kwargs:
            kernel_size = kwargs["kernel_size"]
        if "dilation" in kwargs:
            dilation = kwargs["dilation"]
        if "padding" in kwargs:
            padding = kwargs["padding"]
        if "stride" in kwargs:
            stride = kwargs["stride"]
    if (
        input_arg is None
        or output_size is None
        or kernel_size is None
        or dilation is None
        or padding is None
        or stride is None
    ):
        raise CodegenBackendError(
            "codegen col2im expects input, output_size, kernel_size, dilation, padding, and stride"
        )
    return input_arg, output_size, kernel_size, dilation, padding, stride
