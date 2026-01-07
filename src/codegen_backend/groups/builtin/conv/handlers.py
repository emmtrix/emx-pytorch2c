from __future__ import annotations

from typing import Dict

import numbers
import torch
import torch.fx

from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.conv1d import Conv1dEmitter
from codegen_backend.emitters.conv2d import Conv2dEmitter
from codegen_backend.errors import CodegenBackendError
from codegen_backend.graph import _OpNode
from codegen_backend.indexing import _contiguous_strides
from codegen_backend.kinds import (
    Conv1dHandler,
    Conv2dHandler,
    HandlerContext,
    OpKindHandler,
    OpNodeBuildResult,
)
from codegen_backend.param_normalize import normalize_int_or_pair, normalize_int_or_tuple, normalize_padding
from codegen_backend.specs import OpKind, _OpSpec
from codegen_backend.backend import (
    _error_expected_tensor,
    _infer_output_shape,
    _normalize_param,
    _parse_conv1d_args,
    _parse_conv2d_args,
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
        ) = _parse_conv1d_args(node)
        if not isinstance(input_arg, torch.fx.Node) or not isinstance(
            weight_arg, torch.fx.Node
        ):
            raise _error_expected_tensor("conv1d")
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
            raise _error_expected_tensor("conv1d")
        if (
            dtypes[input_arg] is not torch.float32
            or dtypes[weight_arg] is not torch.float32
        ):
            raise CodegenBackendError(
                "codegen conv1d supports only torch.float32 tensors"
            )
        if bias_node is not None:
            if bias_node not in shapes:
                raise _error_expected_tensor("conv1d")
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
        stride_value = _normalize_param(
            normalize_int_or_tuple, "stride", stride, 1
        )[0]
        dilation_value = _normalize_param(
            normalize_int_or_tuple, "dilation", dilation, 1
        )[0]
        padding_value = _normalize_param(
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
        output_shape = _infer_output_shape(op_node, [input_shape, weight_shape])
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
        ) = _parse_conv2d_args(node)
        if not isinstance(input_arg, torch.fx.Node) or not isinstance(
            weight_arg, torch.fx.Node
        ):
            raise _error_expected_tensor("conv2d")
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
            raise _error_expected_tensor("conv2d")
        if (
            dtypes[input_arg] is not torch.float32
            or dtypes[weight_arg] is not torch.float32
        ):
            raise CodegenBackendError(
                "codegen conv2d supports only torch.float32 tensors"
            )
        if bias_node is not None:
            if bias_node not in shapes:
                raise _error_expected_tensor("conv2d")
            if dtypes[bias_node] is not torch.float32:
                raise CodegenBackendError(
                    "codegen conv2d supports only torch.float32 tensors"
                )
        input_shape = shapes[input_arg]
        weight_shape = shapes[weight_arg]
        if len(weight_shape) != 4:
            raise CodegenBackendError("codegen conv2d requires 4D weight tensors")
        if bias_node is not None:
            bias_shape = shapes[bias_node]
            if len(bias_shape) != 1 or bias_shape[0] != weight_shape[0]:
                raise CodegenBackendError(
                    "codegen conv2d expects bias shape to match output channels"
                )
        if len(input_shape) != 4:
            raise CodegenBackendError("codegen conv2d requires 4D input tensors")
        stride_pair = _normalize_param(
            normalize_int_or_pair, "stride", stride
        )
        padding_pair = _normalize_param(
            normalize_padding, "padding", padding, 2, allow_strings=("same", "valid")
        )
        dilation_pair = _normalize_param(
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
        output_padding_pair = _normalize_param(
            normalize_int_or_pair, "output_padding", output_padding
        )
        if isinstance(padding_pair, str):
            if padding_pair not in ("same", "valid"):
                raise CodegenBackendError(
                    "codegen conv2d expects padding to be 'same', 'valid', or an int tuple"
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
        output_shape = _infer_output_shape(op_node, [input_shape, weight_shape])
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


def build_handlers(context: HandlerContext) -> Dict[OpKind, OpKindHandler]:
    return {
        OpKind.CONV1D: _BackendConv1dHandler(context, Conv1dEmitter()),
        OpKind.CONV2D: _BackendConv2dHandler(context, Conv2dEmitter()),
    }


__all__ = ["build_handlers"]
