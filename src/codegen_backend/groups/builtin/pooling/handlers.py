from __future__ import annotations

from typing import Dict

import numbers
import torch
import torch.fx

from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.pool1d import Pool1dEmitter
from codegen_backend.emitters.pool2d import Pool2dEmitter
from codegen_backend.emitters.pool2d_backward import Pool2dBackwardEmitter
from codegen_backend.emitters.pool3d import Pool3dEmitter
from codegen_backend.errors import CodegenBackendError
from codegen_backend.graph import _OpNode
from codegen_backend.indexing import _contiguous_strides
from codegen_backend.kinds import (
    HandlerContext,
    OpKindHandler,
    OpNodeBuildResult,
    Pool1dHandler,
    Pool2dBackwardHandler,
    Pool2dHandler,
    Pool3dHandler,
)
from codegen_backend.param_normalize import normalize_int_or_pair, normalize_int_or_tuple
from codegen_backend.specs import OpKind, _OpSpec
from codegen_backend.backend import (
    _error_expected_tensor,
    _infer_output_shape,
    _normalize_param,
    _parse_adaptive_avg_pool1d_args,
    _parse_adaptive_avg_pool2d_args,
    _parse_adaptive_avg_pool2d_backward_args,
    _parse_adaptive_avg_pool3d_args,
    _parse_avg_pool1d_args,
    _parse_avg_pool2d_args,
    _parse_max_pool1d_args,
    _parse_max_pool2d_args,
)


class _BackendPool1dHandler(Pool1dHandler):
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
        if op_spec.name == "adaptive_avg_pool1d":
            input_arg, output_size = _parse_adaptive_avg_pool1d_args(node)
            kernel_size = None
            stride = None
            padding = 0
            dilation = 1
            ceil_mode = False
            count_include_pad = False
            divisor_override = None
        elif op_spec.name == "max_pool1d":
            (
                input_arg,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
            ) = _parse_max_pool1d_args(node)
            count_include_pad = False
            divisor_override = None
        else:
            (
                input_arg,
                kernel_size,
                stride,
                padding,
                ceil_mode,
                count_include_pad,
                divisor_override,
            ) = _parse_avg_pool1d_args(node)
            dilation = 1
        if not isinstance(input_arg, torch.fx.Node):
            raise _error_expected_tensor(op_spec.name)
        if input_arg not in shapes:
            raise _error_expected_tensor(op_spec.name)
        if dtype_info.torch_dtype is not torch.float32:
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only torch.float32 tensors"
            )
        if dtypes[input_arg] is not torch.float32:
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only torch.float32 tensors"
            )
        if isinstance(kernel_size, torch.fx.Node) or isinstance(
            padding, torch.fx.Node
        ) or isinstance(ceil_mode, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects constant kernel, padding, and ceil_mode"
            )
        if stride is not None and isinstance(stride, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects constant stride values"
            )
        if isinstance(dilation, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects constant dilation values"
            )
        if isinstance(count_include_pad, torch.fx.Node) or isinstance(
            divisor_override, torch.fx.Node
        ):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects constant pooling options"
            )
        input_shape = shapes[input_arg]
        if len(input_shape) != 3:
            raise CodegenBackendError(
                f"codegen {op_spec.name} requires 3D input tensors"
            )
        if not self._is_contiguous(input_shape, strides[input_arg]):
            raise CodegenBackendError(
                f"codegen {op_spec.name} requires contiguous input tensors"
            )
        if op_spec.name == "adaptive_avg_pool1d":
            if isinstance(output_size, torch.fx.Node):
                raise CodegenBackendError(
                    "codegen adaptive_avg_pool1d expects output_size to be an int"
                )
            if isinstance(output_size, (tuple, list)):
                if len(output_size) != 1:
                    raise CodegenBackendError(
                        "codegen adaptive_avg_pool1d expects a single output size"
                    )
                output_size = output_size[0]
            if not isinstance(output_size, int):
                raise CodegenBackendError(
                    "codegen adaptive_avg_pool1d expects output_size to be an int"
                )
            if output_size <= 0:
                raise CodegenBackendError(
                    "codegen adaptive_avg_pool1d expects output_size to be positive"
                )
            in_l = input_shape[2]
            if in_l % output_size != 0:
                raise CodegenBackendError(
                    "codegen adaptive_avg_pool1d requires input length divisible by output_size"
                )
            kernel_value = in_l // output_size
            stride_value = kernel_value
            padding_value = 0
            dilation_value = 1
        else:
            kernel_value = _normalize_param(
                normalize_int_or_tuple, "kernel_size", kernel_size, 1
            )[0]
            if stride is None:
                stride_value = kernel_value
            else:
                stride_value = _normalize_param(
                    normalize_int_or_tuple, "stride", stride, 1
                )[0]
            padding_value = _normalize_param(
                normalize_int_or_tuple, "padding", padding, 1
            )[0]
            dilation_value = _normalize_param(
                normalize_int_or_tuple, "dilation", dilation, 1
            )[0]
        if (
            kernel_value <= 0
            or stride_value <= 0
            or dilation_value <= 0
            or padding_value < 0
        ):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects positive kernel, stride, and dilation with non-negative padding"
            )
        if ceil_mode and op_spec.name != "max_pool1d":
            raise CodegenBackendError(
                f"codegen {op_spec.name} does not support ceil_mode"
            )
        if isinstance(count_include_pad, bool):
            count_include_pad_value = count_include_pad
        elif isinstance(count_include_pad, numbers.Integral):
            count_include_pad_value = bool(count_include_pad)
        else:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects count_include_pad to be a bool"
            )
        divisor_override_value = divisor_override
        if divisor_override is not None:
            if isinstance(divisor_override, bool) or not isinstance(
                divisor_override, numbers.Integral
            ):
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects divisor_override to be a positive int"
                )
            divisor_override_value = int(divisor_override)
            if divisor_override_value <= 0:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects divisor_override to be a positive int"
                )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg],
            output_shape=(),
            inplace_input=None,
            params={
                "kernel_size": kernel_value,
                "stride": stride_value,
                "padding": padding_value,
                "dilation": dilation_value,
                "ceil_mode": bool(ceil_mode),
                "count_include_pad": count_include_pad_value,
                "divisor_override": divisor_override_value,
            },
        )
        output_shape = _infer_output_shape(op_node, [input_shape])
        op_node.output_shape = output_shape
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        strides[node] = _contiguous_strides(output_shape)
        return OpNodeBuildResult(op_node)

    @staticmethod
    def _is_contiguous(shape: tuple[int, ...], stride: tuple[int, ...]) -> bool:
        from codegen_backend.emitters.base import _is_contiguous

        return _is_contiguous(shape, stride)


class _BackendPool2dHandler(Pool2dHandler):
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
        if op_spec.name == "adaptive_avg_pool2d":
            input_arg, output_size = _parse_adaptive_avg_pool2d_args(node)
            kernel_size = None
            stride = None
            padding = 0
            dilation = 1
            ceil_mode = False
            count_include_pad = False
            divisor_override = None
        elif op_spec.name == "max_pool2d":
            (
                input_arg,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
            ) = _parse_max_pool2d_args(node)
            count_include_pad = False
            divisor_override = None
        else:
            (
                input_arg,
                kernel_size,
                stride,
                padding,
                ceil_mode,
                count_include_pad,
                divisor_override,
            ) = _parse_avg_pool2d_args(node)
            dilation = 1
        if not isinstance(input_arg, torch.fx.Node):
            raise _error_expected_tensor(op_spec.name)
        if input_arg not in shapes:
            raise _error_expected_tensor(op_spec.name)
        if dtype_info.torch_dtype is not torch.float32:
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only torch.float32 tensors"
            )
        if dtypes[input_arg] is not torch.float32:
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only torch.float32 tensors"
            )
        if isinstance(kernel_size, torch.fx.Node) or isinstance(
            padding, torch.fx.Node
        ) or isinstance(ceil_mode, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects constant kernel, padding, and ceil_mode"
            )
        if stride is not None and isinstance(stride, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects constant stride values"
            )
        if isinstance(dilation, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects constant dilation values"
            )
        if isinstance(count_include_pad, torch.fx.Node) or isinstance(
            divisor_override, torch.fx.Node
        ):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects constant pooling options"
            )
        input_shape = shapes[input_arg]
        if len(input_shape) != 4:
            raise CodegenBackendError(
                f"codegen {op_spec.name} requires 4D input tensors"
            )
        if not self._is_contiguous(input_shape, strides[input_arg]):
            raise CodegenBackendError(
                f"codegen {op_spec.name} requires contiguous input tensors"
            )
        if op_spec.name == "adaptive_avg_pool2d":
            if isinstance(output_size, torch.fx.Node):
                raise CodegenBackendError(
                    "codegen adaptive_avg_pool2d expects output_size to be a tuple of ints"
                )
            if isinstance(output_size, torch.Size):
                output_size = tuple(output_size)
            if isinstance(output_size, int):
                output_pair = (output_size, output_size)
            elif isinstance(output_size, (tuple, list)):
                if len(output_size) != 2:
                    raise CodegenBackendError(
                        "codegen adaptive_avg_pool2d expects output_size to have two values"
                    )
                output_pair = tuple(output_size)
            else:
                raise CodegenBackendError(
                    "codegen adaptive_avg_pool2d expects output_size to be a tuple of ints"
                )
            if not all(isinstance(item, int) for item in output_pair):
                raise CodegenBackendError(
                    "codegen adaptive_avg_pool2d expects output_size to be a tuple of ints"
                )
            if output_pair[0] <= 0 or output_pair[1] <= 0:
                raise CodegenBackendError(
                    "codegen adaptive_avg_pool2d expects output_size to be positive"
                )
            in_h, in_w = input_shape[2], input_shape[3]
            if in_h % output_pair[0] != 0 or in_w % output_pair[1] != 0:
                raise CodegenBackendError(
                    "codegen adaptive_avg_pool2d requires input sizes divisible by output_size"
                )
            kernel_pair = (in_h // output_pair[0], in_w // output_pair[1])
            stride_pair = kernel_pair
            padding_pair = (0, 0)
            dilation_pair = (1, 1)
        else:
            kernel_pair = _normalize_param(
                normalize_int_or_pair, "kernel_size", kernel_size
            )
            if stride is None:
                stride_pair = kernel_pair
            else:
                stride_pair = _normalize_param(
                    normalize_int_or_pair, "stride", stride
                )
            padding_pair = _normalize_param(
                normalize_int_or_pair, "padding", padding
            )
            dilation_pair = _normalize_param(
                normalize_int_or_pair, "dilation", dilation
            )
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
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects positive kernel, stride, and dilation with non-negative padding"
            )
        if not isinstance(ceil_mode, bool):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects ceil_mode to be a bool"
            )
        if not isinstance(count_include_pad, bool):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects count_include_pad to be a bool"
            )
        if divisor_override is not None:
            if not isinstance(divisor_override, int) or divisor_override <= 0:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects divisor_override to be a positive int"
                )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg],
            output_shape=(),
            inplace_input=None,
            params={
                "kernel_size": kernel_pair,
                "stride": stride_pair,
                "padding": padding_pair,
                "dilation": dilation_pair,
                "ceil_mode": bool(ceil_mode),
                "count_include_pad": count_include_pad,
                "divisor_override": divisor_override,
            },
        )
        output_shape = _infer_output_shape(op_node, [input_shape])
        op_node.output_shape = output_shape
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        strides[node] = _contiguous_strides(output_shape)
        return OpNodeBuildResult(op_node)

    @staticmethod
    def _is_contiguous(shape: tuple[int, ...], stride: tuple[int, ...]) -> bool:
        from codegen_backend.emitters.base import _is_contiguous

        return _is_contiguous(shape, stride)


class _BackendPool3dHandler(Pool3dHandler):
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
        input_arg, output_size = _parse_adaptive_avg_pool3d_args(node)
        kernel_size = None
        stride = None
        padding = (0, 0, 0)
        dilation = (1, 1, 1)
        ceil_mode = False
        count_include_pad = False
        divisor_override = None
        if not isinstance(input_arg, torch.fx.Node):
            raise _error_expected_tensor(op_spec.name)
        if input_arg not in shapes:
            raise _error_expected_tensor(op_spec.name)
        if dtype_info.torch_dtype is not torch.float32:
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only torch.float32 tensors"
            )
        if dtypes[input_arg] is not torch.float32:
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only torch.float32 tensors"
            )
        if isinstance(kernel_size, torch.fx.Node) or isinstance(
            padding, torch.fx.Node
        ) or isinstance(ceil_mode, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects constant kernel, padding, and ceil_mode"
            )
        if stride is not None and isinstance(stride, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects constant stride values"
            )
        if isinstance(dilation, torch.fx.Node):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects constant dilation values"
            )
        if isinstance(count_include_pad, torch.fx.Node) or isinstance(
            divisor_override, torch.fx.Node
        ):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects constant pooling options"
            )
        input_shape = shapes[input_arg]
        if len(input_shape) != 5:
            raise CodegenBackendError(
                f"codegen {op_spec.name} requires 5D input tensors"
            )
        if not self._is_contiguous(input_shape, strides[input_arg]):
            raise CodegenBackendError(
                f"codegen {op_spec.name} requires contiguous input tensors"
            )
        if isinstance(output_size, torch.fx.Node):
            raise CodegenBackendError(
                "codegen adaptive_avg_pool3d expects output_size to be a tuple of ints"
            )
        if isinstance(output_size, torch.Size):
            output_size = tuple(output_size)
        if isinstance(output_size, int):
            output_triplet = (output_size, output_size, output_size)
        elif isinstance(output_size, (tuple, list)):
            if len(output_size) != 3:
                raise CodegenBackendError(
                    "codegen adaptive_avg_pool3d expects output_size to have three values"
                )
            output_triplet = tuple(output_size)
        else:
            raise CodegenBackendError(
                "codegen adaptive_avg_pool3d expects output_size to be a tuple of ints"
            )
        if not all(isinstance(item, int) for item in output_triplet):
            raise CodegenBackendError(
                "codegen adaptive_avg_pool3d expects output_size to be a tuple of ints"
            )
        if not all(item > 0 for item in output_triplet):
            raise CodegenBackendError(
                "codegen adaptive_avg_pool3d expects output_size to be positive"
            )
        in_d, in_h, in_w = input_shape[2], input_shape[3], input_shape[4]
        if (
            in_d % output_triplet[0] != 0
            or in_h % output_triplet[1] != 0
            or in_w % output_triplet[2] != 0
        ):
            raise CodegenBackendError(
                "codegen adaptive_avg_pool3d requires input sizes divisible by output_size"
            )
        kernel_triplet = (
            in_d // output_triplet[0],
            in_h // output_triplet[1],
            in_w // output_triplet[2],
        )
        stride_triplet = kernel_triplet
        padding_triplet = (0, 0, 0)
        dilation_triplet = (1, 1, 1)
        if (
            kernel_triplet[0] <= 0
            or kernel_triplet[1] <= 0
            or kernel_triplet[2] <= 0
            or stride_triplet[0] <= 0
            or stride_triplet[1] <= 0
            or stride_triplet[2] <= 0
            or dilation_triplet[0] <= 0
            or dilation_triplet[1] <= 0
            or dilation_triplet[2] <= 0
            or padding_triplet[0] < 0
            or padding_triplet[1] < 0
            or padding_triplet[2] < 0
        ):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects positive kernel, stride, and dilation with non-negative padding"
            )
        if not isinstance(ceil_mode, bool):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects ceil_mode to be a bool"
            )
        if not isinstance(count_include_pad, bool):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects count_include_pad to be a bool"
            )
        if divisor_override is not None:
            if not isinstance(divisor_override, int) or divisor_override <= 0:
                raise CodegenBackendError(
                    f"codegen {op_spec.name} expects divisor_override to be a positive int"
                )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg],
            output_shape=(),
            inplace_input=None,
            params={
                "kernel_size": kernel_triplet,
                "stride": stride_triplet,
                "padding": padding_triplet,
                "dilation": dilation_triplet,
                "ceil_mode": bool(ceil_mode),
                "count_include_pad": count_include_pad,
                "divisor_override": divisor_override,
            },
        )
        output_shape = _infer_output_shape(op_node, [input_shape])
        op_node.output_shape = output_shape
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        strides[node] = _contiguous_strides(output_shape)
        return OpNodeBuildResult(op_node)

    @staticmethod
    def _is_contiguous(shape: tuple[int, ...], stride: tuple[int, ...]) -> bool:
        from codegen_backend.emitters.base import _is_contiguous

        return _is_contiguous(shape, stride)


class _BackendPool2dBackwardHandler(Pool2dBackwardHandler):
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
        grad_output, input_arg = _parse_adaptive_avg_pool2d_backward_args(node)
        if not isinstance(grad_output, torch.fx.Node) or not isinstance(
            input_arg, torch.fx.Node
        ):
            raise _error_expected_tensor(op_spec.name)
        if grad_output not in shapes or input_arg not in shapes:
            raise _error_expected_tensor(op_spec.name)
        if dtype_info.torch_dtype is not torch.float32:
            raise CodegenBackendError(
                "codegen adaptive_avg_pool2d_backward supports only torch.float32 tensors"
            )
        if (
            dtypes[grad_output] is not torch.float32
            or dtypes[input_arg] is not torch.float32
        ):
            raise CodegenBackendError(
                "codegen adaptive_avg_pool2d_backward supports only torch.float32 tensors"
            )
        grad_output_shape = shapes[grad_output]
        input_shape = shapes[input_arg]
        if len(grad_output_shape) != 4 or len(input_shape) != 4:
            raise CodegenBackendError(
                "codegen adaptive_avg_pool2d_backward requires 4D input tensors"
            )
        if not self._is_contiguous(grad_output_shape, strides[grad_output]):
            raise CodegenBackendError(
                "codegen adaptive_avg_pool2d_backward requires contiguous grad_output tensors"
            )
        if not self._is_contiguous(input_shape, strides[input_arg]):
            raise CodegenBackendError(
                "codegen adaptive_avg_pool2d_backward requires contiguous input tensors"
            )
        if (
            grad_output_shape[0] != input_shape[0]
            or grad_output_shape[1] != input_shape[1]
        ):
            raise CodegenBackendError(
                "codegen adaptive_avg_pool2d_backward requires matching batch and channel sizes"
            )
        out_h, out_w = grad_output_shape[2], grad_output_shape[3]
        in_h, in_w = input_shape[2], input_shape[3]
        if out_h <= 0 or out_w <= 0:
            raise CodegenBackendError(
                "codegen adaptive_avg_pool2d_backward expects output size to be positive"
            )
        if in_h % out_h != 0 or in_w % out_w != 0:
            raise CodegenBackendError(
                "codegen adaptive_avg_pool2d_backward requires input sizes divisible by output_size"
            )
        kernel_pair = (in_h // out_h, in_w // out_w)
        if kernel_pair[0] <= 0 or kernel_pair[1] <= 0:
            raise CodegenBackendError(
                "codegen adaptive_avg_pool2d_backward expects positive kernel size"
            )
        stride_pair = kernel_pair
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[grad_output, input_arg],
            output_shape=(),
            inplace_input=None,
            params={
                "kernel_size": kernel_pair,
                "stride": stride_pair,
            },
        )
        output_shape = _infer_output_shape(op_node, [grad_output_shape, input_shape])
        op_node.output_shape = output_shape
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        strides[node] = _contiguous_strides(output_shape)
        return OpNodeBuildResult(op_node)

    @staticmethod
    def _is_contiguous(shape: tuple[int, ...], stride: tuple[int, ...]) -> bool:
        from codegen_backend.emitters.base import _is_contiguous

        return _is_contiguous(shape, stride)


def build_handlers(context: HandlerContext) -> Dict[OpKind, OpKindHandler]:
    return {
        OpKind.POOL1D: _BackendPool1dHandler(context, Pool1dEmitter()),
        OpKind.POOL2D: _BackendPool2dHandler(context, Pool2dEmitter()),
        OpKind.POOL3D: _BackendPool3dHandler(context, Pool3dEmitter()),
        OpKind.POOL2D_BACKWARD: _BackendPool2dBackwardHandler(
            context, Pool2dBackwardEmitter()
        ),
    }


__all__ = ["build_handlers"]
