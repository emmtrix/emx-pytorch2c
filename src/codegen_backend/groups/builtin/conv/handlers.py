from __future__ import annotations

from typing import Dict, List, Sequence, TYPE_CHECKING, Tuple

import math
import numbers
import torch
import torch.fx

from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.conv1d import Conv1dEmitter
from codegen_backend.emitters.conv2d import Conv2dEmitter
from codegen_backend.errors import CodegenBackendError
from codegen_backend.graph import _GenericGraph, _OpNode
from codegen_backend.indexing import _contiguous_strides
from codegen_backend.kinds import (
    ConvContext,
    HandlerContextProvider,
    OpKindHandler,
    OpKindHandlerFactory,
    OpNodeBuildResult,
)
from codegen_backend.param_normalize import normalize_int_or_pair, normalize_int_or_tuple, normalize_padding
from codegen_backend.specs import OpKind, _OpSpec
from codegen_backend.analysis_helpers import normalize_param
from codegen_backend.groups.builtin.conv.parsing import (
    parse_conv1d_args,
    parse_conv2d_args,
)

if TYPE_CHECKING:
    from codegen_backend.emitters.registry import KindHandlerRegistration


def _unpack_conv2d_input_shape(
    input_shape: Sequence[int],
) -> Tuple[bool, int, int, int, int]:
    if len(input_shape) == 4:
        batch, in_channels, in_h, in_w = input_shape
        return True, batch, in_channels, in_h, in_w
    if len(input_shape) == 3:
        in_channels, in_h, in_w = input_shape
        return False, 1, in_channels, in_h, in_w
    raise CodegenBackendError("codegen conv2d requires 3D or 4D input tensors")


def _conv2d_output_shape_from_shapes(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    groups: int,
) -> Tuple[int, ...]:
    has_batch, batch, in_channels, in_h, in_w = _unpack_conv2d_input_shape(
        input_shape
    )
    out_channels, weight_in_channels, kernel_h, kernel_w = weight_shape
    if in_channels != weight_in_channels * groups:
        raise CodegenBackendError(
            "codegen conv2d requires input channels to match weight channels * groups"
        )
    if out_channels % groups != 0:
        raise CodegenBackendError(
            "codegen conv2d requires output channels to be divisible by groups"
        )
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    numerator_h = in_h + 2 * pad_h - dil_h * (kernel_h - 1) - 1
    numerator_w = in_w + 2 * pad_w - dil_w * (kernel_w - 1) - 1
    if numerator_h < 0 or numerator_w < 0:
        raise CodegenBackendError(
            "codegen conv2d requires output shape (N, C_out, H_out, W_out)"
        )
    out_h = numerator_h // stride_h + 1
    out_w = numerator_w // stride_w + 1
    if has_batch:
        return batch, out_channels, out_h, out_w
    return out_channels, out_h, out_w


def _conv2d_transposed_output_shape_from_shapes(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    output_padding: Tuple[int, int],
    groups: int,
) -> Tuple[int, ...]:
    has_batch, batch, in_channels, in_h, in_w = _unpack_conv2d_input_shape(
        input_shape
    )
    weight_in_channels, weight_out_channels, kernel_h, kernel_w = weight_shape
    if in_channels != weight_in_channels:
        raise CodegenBackendError(
            "codegen conv2d requires input channels to match weight channels"
        )
    if in_channels % groups != 0:
        raise CodegenBackendError(
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
        raise CodegenBackendError(
            "codegen conv2d requires output shape (N, C_out, H_out, W_out)"
        )
    if has_batch:
        return batch, out_channels, out_h, out_w
    return out_channels, out_h, out_w


def _conv2d_same_padding(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    stride: Tuple[int, int],
    dilation: Tuple[int, int],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    _, _, in_h, in_w = _unpack_conv2d_input_shape(input_shape)[1:]
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


def _conv2d_validate_channels(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    groups: int,
) -> Tuple[bool, int]:
    has_batch, _, in_channels, _, _ = _unpack_conv2d_input_shape(input_shape)
    out_channels, weight_in_channels, _, _ = weight_shape
    if in_channels != weight_in_channels * groups:
        raise CodegenBackendError(
            "codegen conv2d requires input channels to match weight channels * groups"
        )
    if out_channels % groups != 0:
        raise CodegenBackendError(
            "codegen conv2d requires output channels to be divisible by groups"
        )
    return has_batch, out_channels


def _conv1d_validate_channels(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    groups: int,
) -> Tuple[int, int]:
    batch, in_channels, _ = input_shape
    out_channels, weight_in_channels, _ = weight_shape
    if in_channels != weight_in_channels * groups:
        raise CodegenBackendError(
            "codegen conv1d requires input channels to match weight channels * groups"
        )
    if out_channels % groups != 0:
        raise CodegenBackendError(
            "codegen conv1d requires output channels to be divisible by groups"
        )
    return batch, out_channels


def _conv1d_same_padding(
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


def _conv1d_output_shape_from_shapes(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
) -> Tuple[int, int, int]:
    batch, out_channels = _conv1d_validate_channels(
        input_shape, weight_shape, groups
    )
    in_l = input_shape[2]
    kernel_l = weight_shape[2]
    numerator = in_l + 2 * padding - dilation * (kernel_l - 1) - 1
    if numerator < 0:
        raise CodegenBackendError(
            "codegen conv1d requires output shape (N, C_out, L_out)"
        )
    out_l = numerator // stride + 1
    return batch, out_channels, out_l


class Conv1dHandler(OpKindHandler):
    def emit(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        input_node, weight_node, *_ = op_node.inputs
        # Only the input and weight tensors are part of the kernel signature.
        return self._emit_standard(
            node_index,
            op_node,
            graph,
            inputs=(input_node, weight_node),
            params={
                "stride": op_node.p("stride", 1),
                "padding": op_node.p("padding", 0),
                "dilation": op_node.p("dilation", 1),
                "groups": op_node.p("groups", 1),
                "has_bias": bool(op_node.p("has_bias", False)),
            },
        )

    def infer_shapes(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        input_shape, weight_shape = input_shapes[:2]
        stride = op_node.p("stride", 1)
        padding = op_node.p("padding", 0)
        dilation = op_node.p("dilation", 1)
        groups = op_node.p("groups", 1)
        if isinstance(padding, str):
            if padding == "valid":
                padding_value = 0
                return _conv1d_output_shape_from_shapes(
                    input_shape,
                    weight_shape,
                    stride,
                    padding_value,
                    dilation,
                    groups,
                )
            padding_value, out_l = _conv1d_same_padding(
                input_shape,
                weight_shape,
                stride,
                dilation,
            )
            op_node.params["padding"] = padding_value
            batch, out_channels = _conv1d_validate_channels(
                input_shape,
                weight_shape,
                groups,
            )
            return batch, out_channels, out_l
        return _conv1d_output_shape_from_shapes(
            input_shape,
            weight_shape,
            stride,
            padding,
            dilation,
            groups,
        )


class Conv2dHandler(OpKindHandler):
    def emit(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        input_node, weight_node, *_ = op_node.inputs
        # Only the input and weight tensors are part of the kernel signature.
        return self._emit_standard(
            node_index,
            op_node,
            graph,
            inputs=(input_node, weight_node),
            params={
                "transposed": bool(op_node.p("transposed", False)),
                "stride": op_node.p("stride", (1, 1)),
                "padding": op_node.p("padding", (0, 0)),
                "dilation": op_node.p("dilation", (1, 1)),
                "output_padding": op_node.p("output_padding", (0, 0)),
                "groups": op_node.p("groups", 1),
                "has_bias": bool(op_node.p("has_bias", False)),
            },
        )

    def infer_shapes(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        input_shape, weight_shape = input_shapes[:2]
        stride = op_node.p("stride", (1, 1))
        padding = op_node.p("padding", (0, 0))
        dilation = op_node.p("dilation", (1, 1))
        groups = op_node.p("groups", 1)
        transposed = bool(op_node.p("transposed", False))
        output_padding = op_node.p("output_padding", (0, 0))
        if isinstance(padding, str):
            if padding == "valid":
                padding_value = (0, 0)
                return _conv2d_output_shape_from_shapes(
                    input_shape,
                    weight_shape,
                    stride,
                    padding_value,
                    dilation,
                    groups,
                )
            has_batch, out_channels = _conv2d_validate_channels(
                input_shape, weight_shape, groups
            )
            padding_value, (out_h, out_w) = _conv2d_same_padding(
                input_shape, weight_shape, stride, dilation
            )
            op_node.params["padding"] = padding_value
            if has_batch:
                return (input_shape[0], out_channels, out_h, out_w)
            return (out_channels, out_h, out_w)
        if transposed:
            return _conv2d_transposed_output_shape_from_shapes(
                input_shape,
                weight_shape,
                stride,
                padding,
                dilation,
                output_padding,
                groups,
            )
        return _conv2d_output_shape_from_shapes(
            input_shape,
            weight_shape,
            stride,
            padding,
            dilation,
            groups,
        )


class _BackendConv1dHandler(Conv1dHandler):
    def build_op_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType | None,
        shapes: Dict[torch.fx.Node, tuple[int, ...]],
        strides: Dict[torch.fx.Node, tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
        inplace_input: int | None,
    ) -> OpNodeBuildResult | None:
        if dtype_info is None:
            return None
        (
            input_arg,
            weight_arg,
            bias,
            stride,
            padding,
            dilation,
            groups,
        ) = parse_conv1d_args(node)
        if not isinstance(input_arg, torch.fx.Node) or not isinstance(
            weight_arg, torch.fx.Node
        ):
            raise self._ctx.analysis_service.error_expected_tensor("conv1d")
        bias_node = None
        if bias is not None:
            if isinstance(bias, torch.fx.Node):
                bias_node = bias
            else:
                raise CodegenBackendError("codegen conv1d expects bias to be a tensor")
        if isinstance(stride, torch.fx.Node) or isinstance(
            padding, torch.fx.Node
        ) or isinstance(dilation, torch.fx.Node):
            raise CodegenBackendError(
                "codegen conv1d expects constant stride, padding, and dilation"
            )
        if isinstance(groups, torch.fx.Node):
            raise CodegenBackendError("codegen conv1d expects constant groups")
        if dtype_info.torch_dtype is not torch.float32:
            raise CodegenBackendError(
                "codegen conv1d supports only torch.float32 tensors"
            )
        if input_arg not in shapes or weight_arg not in shapes:
            raise self._ctx.analysis_service.error_expected_tensor("conv1d")
        if (
            dtypes[input_arg] is not torch.float32
            or dtypes[weight_arg] is not torch.float32
        ):
            raise CodegenBackendError(
                "codegen conv1d supports only torch.float32 tensors"
            )
        if bias_node is not None:
            if bias_node not in shapes:
                raise self._ctx.analysis_service.error_expected_tensor("conv1d")
            if dtypes[bias_node] is not torch.float32:
                raise CodegenBackendError(
                    "codegen conv1d supports only torch.float32 tensors"
                )
        input_shape = shapes[input_arg]
        weight_shape = shapes[weight_arg]
        if bias_node is not None:
            bias_shape = shapes[bias_node]
            if len(bias_shape) != 1 or bias_shape[0] != weight_shape[0]:
                raise CodegenBackendError(
                    "codegen conv1d expects bias shape to match output channels"
                )
        if len(input_shape) != 3 or len(weight_shape) != 3:
            raise CodegenBackendError(
                "codegen conv1d requires 3D input and weight tensors"
            )
        stride_value = normalize_param(
            normalize_int_or_tuple, "stride", stride, 1
        )[0]
        dilation_value = normalize_param(
            normalize_int_or_tuple, "dilation", dilation, 1
        )[0]
        padding_value = normalize_param(
            normalize_padding, "padding", padding, 1, allow_strings=("same", "valid")
        )
        if not isinstance(padding_value, str):
            padding_value = padding_value[0]
        if stride_value <= 0 or dilation_value <= 0 or (
            not isinstance(padding_value, str) and padding_value < 0
        ):
            raise CodegenBackendError(
                "codegen conv1d expects stride and dilation to be positive and padding to be non-negative"
            )
        if not isinstance(groups, int) or groups <= 0:
            raise CodegenBackendError("codegen conv1d requires positive groups")
        inputs = (input_arg, weight_arg)
        if bias_node is not None:
            inputs = (*inputs, bias_node)
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=list(inputs),
            output_shape=(),
            inplace_input=None,
            params={
                "stride": stride_value,
                "padding": padding_value,
                "dilation": dilation_value,
                "groups": groups,
                "has_bias": bias_node is not None,
            },
        )
        output_shape = self._ctx.analysis_service.infer_output_shape(
            op_node, [input_shape, weight_shape]
        )
        op_node.output_shape = output_shape
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        strides[node] = _contiguous_strides(output_shape)
        return OpNodeBuildResult(op_node)


class _BackendConv2dHandler(Conv2dHandler):
    def build_op_node(
        self,
        node: torch.fx.Node,
        op_spec: _OpSpec,
        dtype_info: _CodegenDType | None,
        shapes: Dict[torch.fx.Node, tuple[int, ...]],
        strides: Dict[torch.fx.Node, tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
        inplace_input: int | None,
    ) -> OpNodeBuildResult | None:
        if dtype_info is None:
            return None
        (
            input_arg,
            weight_arg,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        ) = parse_conv2d_args(node)
        if not isinstance(input_arg, torch.fx.Node) or not isinstance(
            weight_arg, torch.fx.Node
        ):
            raise self._ctx.analysis_service.error_expected_tensor("conv2d")
        bias_node = None
        if bias is not None:
            if isinstance(bias, torch.fx.Node):
                bias_node = bias
            else:
                raise CodegenBackendError("codegen conv2d expects bias to be a tensor")
        if isinstance(stride, torch.fx.Node) or isinstance(
            padding, torch.fx.Node
        ) or isinstance(dilation, torch.fx.Node):
            raise CodegenBackendError(
                "codegen conv2d expects constant stride, padding, and dilation"
            )
        if isinstance(transposed, torch.fx.Node):
            raise CodegenBackendError("codegen conv2d expects constant transposed value")
        if isinstance(output_padding, torch.fx.Node):
            raise CodegenBackendError("codegen conv2d expects constant output_padding")
        if isinstance(groups, torch.fx.Node):
            raise CodegenBackendError("codegen conv2d expects constant groups")
        if dtype_info.torch_dtype is not torch.float32:
            raise CodegenBackendError(
                "codegen conv2d supports only torch.float32 tensors"
            )
        if input_arg not in shapes or weight_arg not in shapes:
            raise self._ctx.analysis_service.error_expected_tensor("conv2d")
        if (
            dtypes[input_arg] is not torch.float32
            or dtypes[weight_arg] is not torch.float32
        ):
            raise CodegenBackendError(
                "codegen conv2d supports only torch.float32 tensors"
            )
        if bias_node is not None:
            if bias_node not in shapes:
                raise self._ctx.analysis_service.error_expected_tensor("conv2d")
            if dtypes[bias_node] is not torch.float32:
                raise CodegenBackendError(
                    "codegen conv2d supports only torch.float32 tensors"
                )
        input_shape = shapes[input_arg]
        weight_shape = shapes[weight_arg]
        if len(weight_shape) != 4:
            raise CodegenBackendError("codegen conv2d requires 4D weight tensors")
        if len(input_shape) not in (3, 4):
            raise CodegenBackendError(
                "codegen conv2d requires 3D or 4D input tensors"
            )
        stride_pair = normalize_param(
            normalize_int_or_pair, "stride", stride
        )
        padding_pair = normalize_param(
            normalize_padding, "padding", padding, 2, allow_strings=("same", "valid")
        )
        dilation_pair = normalize_param(
            normalize_int_or_pair, "dilation", dilation
        )
        if isinstance(transposed, bool):
            transposed_value = transposed
        elif isinstance(transposed, numbers.Integral):
            transposed_value = bool(transposed)
        else:
            raise CodegenBackendError(
                "codegen conv2d expects transposed to be a bool"
            )
        output_padding_pair = normalize_param(
            normalize_int_or_pair, "output_padding", output_padding
        )
        if isinstance(padding_pair, str):
            if padding_pair not in ("same", "valid"):
                raise CodegenBackendError(
                    "codegen conv2d expects padding to be 'same', 'valid', or an int tuple"
                )
            if transposed_value:
                raise CodegenBackendError(
                    "codegen conv2d expects transposed padding to be an int tuple"
                )
        else:
            if padding_pair[0] < 0 or padding_pair[1] < 0:
                raise CodegenBackendError(
                    "codegen conv2d expects padding to be non-negative"
                )
        if stride_pair[0] <= 0 or stride_pair[1] <= 0:
            raise CodegenBackendError("codegen conv2d expects stride to be positive")
        if dilation_pair[0] <= 0 or dilation_pair[1] <= 0:
            raise CodegenBackendError("codegen conv2d expects dilation to be positive")
        if output_padding_pair[0] < 0 or output_padding_pair[1] < 0:
            raise CodegenBackendError(
                "codegen conv2d expects output_padding to be non-negative"
            )
        if transposed_value and not isinstance(padding_pair, str):
            if (
                output_padding_pair[0]
                >= max(stride_pair[0], dilation_pair[0])
                or output_padding_pair[1]
                >= max(stride_pair[1], dilation_pair[1])
            ):
                raise CodegenBackendError(
                    "codegen conv2d expects output_padding to be smaller than stride or dilation"
                )
        if groups <= 0:
            raise CodegenBackendError("codegen conv2d requires positive groups")
        inputs = (input_arg, weight_arg)
        if bias_node is not None:
            inputs = (*inputs, bias_node)
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=list(inputs),
            output_shape=(),
            inplace_input=None,
            params={
                "stride": stride_pair,
                "padding": padding_pair,
                "dilation": dilation_pair,
                "groups": groups,
                "transposed": transposed_value,
                "output_padding": output_padding_pair,
                "has_bias": bias_node is not None,
            },
        )
        output_shape = self._ctx.analysis_service.infer_output_shape(
            op_node, [input_shape, weight_shape]
        )
        op_node.output_shape = output_shape
        if bias_node is not None:
            bias_shape = shapes[bias_node]
            if len(output_shape) == 4:
                out_channels = output_shape[1]
            else:
                out_channels = output_shape[0]
            if len(bias_shape) != 1 or bias_shape[0] != out_channels:
                raise CodegenBackendError(
                    "codegen conv2d expects bias shape to match output channels"
                )
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        strides[node] = _contiguous_strides(output_shape)
        return OpNodeBuildResult(op_node)


def build_handlers(context: ConvContext) -> Dict[OpKind, OpKindHandler]:
    return {
        OpKind.CONV1D: _BackendConv1dHandler(context, Conv1dEmitter()),
        OpKind.CONV2D: _BackendConv2dHandler(context, Conv2dEmitter()),
    }


class ConvKindHandlerFactory:
    def build_handlers(
        self, context_provider: HandlerContextProvider
    ) -> Dict[OpKind, OpKindHandler]:
        return build_handlers(context_provider.conv)


def build_kind_handler_registrations() -> Dict[OpKind, "KindHandlerRegistration"]:
    from codegen_backend.emitters.registry import KindHandlerRegistration

    return {
        OpKind.CONV1D: KindHandlerRegistration(
            _BackendConv1dHandler, Conv1dEmitter
        ),
        OpKind.CONV2D: KindHandlerRegistration(
            _BackendConv2dHandler, Conv2dEmitter
        ),
    }


__all__ = [
    "ConvKindHandlerFactory",
    "build_handlers",
    "build_kind_handler_registrations",
]
