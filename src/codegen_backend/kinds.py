from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from fractions import Fraction
import math
import numbers
from typing import TYPE_CHECKING, Dict, List, Protocol, Sequence, Tuple

from c_ref_backend.cffi_bindings import RefBackendError
from codegen_backend import shape_utils

if TYPE_CHECKING:
    import torch
    from codegen_backend.graph import _GenericGraph, _OpNode
    from codegen_backend.specs import _OpSpec

@dataclass
class KernelEmitRequest:
    node_index: int
    op_node: "_OpNode | None" = None
    op_spec: "_OpSpec | None" = None
    inputs: Sequence["torch.fx.Node"] = field(default_factory=tuple)
    output_shape: Sequence[int] = field(default_factory=tuple)
    output_strides: Sequence[int] | None = None
    input_shapes: Sequence[Sequence[int]] = field(default_factory=tuple)
    input_strides: Sequence[Sequence[int]] = field(default_factory=tuple)
    input_dtypes: Sequence[object] = field(default_factory=tuple)
    dtype: object | None = None
    reduction_dims: Sequence[int] | None = None
    keepdim: bool | None = None
    params: Dict[str, object] = field(default_factory=dict)


class HandlerContext(Protocol):
    def kernel_inputs(self, op_node: "_OpNode") -> List["torch.fx.Node"]: ...

    def emit_kernel(self, kind: str, req: KernelEmitRequest) -> List[str]: ...


class KindHandler(ABC):
    def __init__(self, context: HandlerContext) -> None:
        self._ctx = context

    @abstractmethod
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        raise NotImplementedError


def _make_request(
    node_index: int,
    op_node: _OpNode,
    graph: _GenericGraph,
    inputs: Sequence["torch.fx.Node"],
) -> KernelEmitRequest:
    return KernelEmitRequest(
        node_index=node_index,
        op_node=op_node,
        op_spec=op_node.spec,
        inputs=inputs,
        output_shape=op_node.output_shape,
        output_strides=graph.strides[op_node.node],
        input_shapes=[graph.shapes[node] for node in inputs],
        input_strides=[graph.strides[node] for node in inputs],
        input_dtypes=[graph.dtypes[node] for node in inputs],
        dtype=graph.dtype,
    )


def _compute_arange_size(
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


def _infer_diagonal_output_shape(
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


def _infer_reduction_output_shape(
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


def _unpack_conv2d_input_shape(
    input_shape: Sequence[int],
) -> Tuple[bool, int, int, int, int]:
    if len(input_shape) == 4:
        batch, in_channels, in_h, in_w = input_shape
        return True, batch, in_channels, in_h, in_w
    if len(input_shape) == 3:
        in_channels, in_h, in_w = input_shape
        return False, 1, in_channels, in_h, in_w
    raise RefBackendError("codegen conv2d requires 3D or 4D input tensors")


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
        raise RefBackendError(
            "codegen conv2d requires input channels to match weight channels * groups"
        )
    if out_channels % groups != 0:
        raise RefBackendError(
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
        raise RefBackendError(
            "codegen conv1d requires input channels to match weight channels * groups"
        )
    if out_channels % groups != 0:
        raise RefBackendError(
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
        raise RefBackendError(
            "codegen conv1d requires output shape (N, C_out, L_out)"
        )
    out_l = numerator // stride + 1
    return batch, out_channels, out_l


def _pool1d_output_shape_from_shapes(
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


def _pool2d_output_shape_from_shapes(
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


def _pool3d_output_shape_from_shapes(
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


class ArangeHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        req = _make_request(node_index, op_node, graph, ())
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        output_size = _compute_arange_size(
            op_node.p("start"), op_node.p("end"), op_node.p("step")
        )
        return (output_size,)


class ElementwiseHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        kernel_inputs = self._ctx.kernel_inputs(op_node)
        req = _make_request(node_index, op_node, graph, kernel_inputs)
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        op_spec = op_node.spec
        if op_spec.kind == "binary":
            if op_spec.name == "copy":
                output_shape = input_shapes[0]
                broadcast_shape = shape_utils.broadcast_output_shape(
                    op_spec.name, *input_shapes
                )
                if broadcast_shape != output_shape:
                    raise RefBackendError(
                        "codegen copy expects source to be broadcastable to the destination"
                    )
                return output_shape
            return shape_utils.broadcast_output_shape(
                op_spec.name, *input_shapes
            )
        if op_spec.kind == "where":
            return shape_utils.broadcast_output_shape(
                op_spec.name, *input_shapes
            )
        if op_spec.kind in {"unary", "fill"}:
            return input_shapes[0]
        raise NotImplementedError(
            f"Shape inference not implemented for kind '{op_spec.kind}'."
        )


class FlipHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        input_node = op_node.inputs[0]
        req = _make_request(node_index, op_node, graph, (input_node,))
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        return input_shapes[0]


class PadHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        input_node = op_node.inputs[0]
        req = _make_request(node_index, op_node, graph, (input_node,))
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        input_shape = input_shapes[0]
        pad_before = op_node.p("pad_before", ())
        pad_after = op_node.p("pad_after", ())
        if not pad_before or not pad_after:
            raise RefBackendError(
                "codegen constant_pad_nd expects constant padding arguments"
            )
        output_shape = []
        for size, before, after in zip(input_shape, pad_before, pad_after):
            new_size = size + before + after
            if new_size < 0:
                raise RefBackendError(
                    "codegen constant_pad_nd expects non-negative output sizes"
                )
            output_shape.append(new_size)
        return tuple(output_shape)


class ViewHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        input_node = op_node.inputs[0]
        req = _make_request(node_index, op_node, graph, (input_node,))
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        if op_node.spec.name == "as_strided":
            size = op_node.p("size", None)
            if size is None:
                raise RefBackendError(
                    "codegen as_strided expects size and stride"
                )
            return tuple(size)
        if op_node.spec.name == "squeeze":
            input_shape = input_shapes[0]
            squeeze_dims = op_node.p("squeeze_dims", ())
            remove_dims = {
                dim for dim in squeeze_dims if input_shape[dim] == 1
            }
            return tuple(
                size
                for dim, size in enumerate(input_shape)
                if dim not in remove_dims
            )
        raise RefBackendError(f"Unsupported view op: {op_node.spec.name}")


class ResizeHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        input_node = op_node.inputs[0]
        req = _make_request(node_index, op_node, graph, (input_node,))
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        size = op_node.p("size", None)
        if size is None:
            raise RefBackendError("codegen resize_ expects a size argument")
        return tuple(size)


class EmptyStridedHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        req = _make_request(node_index, op_node, graph, ())
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        size = op_node.p("size", None)
        if size is None:
            raise RefBackendError("codegen empty_strided expects a size argument")
        return tuple(size)


class DiagonalHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        input_node = op_node.inputs[0]
        req = _make_request(node_index, op_node, graph, (input_node,))
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        return _infer_diagonal_output_shape(
            input_shapes[0],
            int(op_node.p("offset", 0)),
            int(op_node.p("dim1", 0)),
            int(op_node.p("dim2", 1)),
        )


class ReductionHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        input_node = op_node.inputs[0]
        req = _make_request(node_index, op_node, graph, (input_node,))
        req.reduction_dims = op_node.reduction_dims or ()
        req.keepdim = op_node.keepdim
        if op_node.spec.name in {"std", "var"}:
            req.params["unbiased"] = bool(op_node.p("unbiased", True))
        if op_node.spec.name == "norm":
            req.params["p_value"] = float(op_node.p("norm_p", 2.0))
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        return _infer_reduction_output_shape(
            input_shapes[0],
            op_node.reduction_dims or (),
            op_node.keepdim,
            reduce_all=bool(op_node.p("reduce_all", False)),
        )


class ArgReductionHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        input_node = op_node.inputs[0]
        req = _make_request(node_index, op_node, graph, (input_node,))
        req.reduction_dims = op_node.reduction_dims or ()
        req.keepdim = op_node.keepdim
        req.params["reduce_all"] = bool(op_node.p("reduce_all", False))
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        return _infer_reduction_output_shape(
            input_shapes[0],
            op_node.reduction_dims or (),
            op_node.keepdim,
            reduce_all=bool(op_node.p("reduce_all", False)),
        )


class SoftmaxHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        input_node = op_node.inputs[0]
        req = _make_request(node_index, op_node, graph, (input_node,))
        req.params["dim"] = op_node.p("dim")
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        return input_shapes[0]


class CumsumHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        input_node = op_node.inputs[0]
        req = _make_request(node_index, op_node, graph, (input_node,))
        req.params["dim"] = op_node.p("dim")
        req.params["output_dtype"] = graph.dtypes[op_node.node]
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        return input_shapes[0]


class EmbeddingHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        weight_node, indices_node = op_node.inputs
        req = _make_request(
            node_index, op_node, graph, (weight_node, indices_node)
        )
        req.params["padding_idx"] = int(op_node.p("padding_idx", -1))
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        weight_shape, indices_shape = input_shapes
        if len(weight_shape) != 2:
            raise RefBackendError(
                "codegen embedding expects 2D weight tensor"
            )
        return tuple(indices_shape) + (weight_shape[1],)


class EmbeddingBagHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        weight_node, indices_node, offsets_node = op_node.inputs
        req = _make_request(
            node_index,
            op_node,
            graph,
            (weight_node, indices_node, offsets_node),
        )
        req.params["mode"] = int(op_node.p("mode", 0))
        req.params["padding_idx"] = int(op_node.p("padding_idx", -1))
        req.params["include_last_offset"] = bool(
            op_node.p("include_last_offset", False)
        )
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        weight_shape, _indices_shape, offsets_shape = input_shapes
        if len(weight_shape) != 2:
            raise RefBackendError(
                "codegen _embedding_bag expects 2D weight tensor"
            )
        if len(offsets_shape) != 1:
            raise RefBackendError(
                "codegen _embedding_bag expects 1D offsets tensor"
            )
        include_last_offset = bool(op_node.p("include_last_offset", False))
        bag_count = offsets_shape[0] - 1 if include_last_offset else offsets_shape[0]
        return (bag_count, weight_shape[1])


class GatherHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        input_node, index_node = op_node.inputs
        req = _make_request(node_index, op_node, graph, (input_node, index_node))
        req.params["dim"] = int(op_node.p("dim"))
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        return input_shapes[1]


class ConcatHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        req = _make_request(node_index, op_node, graph, op_node.inputs)
        req.params["dim"] = op_node.p("dim", 0)
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        if not input_shapes:
            raise RefBackendError(
                "codegen cat expects a non-empty tensor list input"
            )
        concat_dim = int(op_node.p("dim", 0))
        output_shape = list(input_shapes[0])
        output_shape[concat_dim] = sum(
            shape[concat_dim] for shape in input_shapes
        )
        return tuple(output_shape)


class Pool2dHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        input_node = op_node.inputs[0]
        req = _make_request(node_index, op_node, graph, (input_node,))
        req.params["kernel_size"] = op_node.p("kernel_size", (1, 1))
        req.params["stride"] = op_node.p("stride", (1, 1))
        req.params["padding"] = op_node.p("padding", (0, 0))
        req.params["dilation"] = op_node.p("dilation", (1, 1))
        req.params["ceil_mode"] = bool(op_node.p("ceil_mode", False))
        req.params["count_include_pad"] = bool(
            op_node.p("count_include_pad", False)
        )
        req.params["divisor_override"] = op_node.p("divisor_override")
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        return _pool2d_output_shape_from_shapes(
            input_shapes[0],
            op_node.p("kernel_size", (1, 1)),
            op_node.p("stride", (1, 1)),
            op_node.p("padding", (0, 0)),
            op_node.p("dilation", (1, 1)),
            bool(op_node.p("ceil_mode", False)),
        )


class Pool3dHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        input_node = op_node.inputs[0]
        req = _make_request(node_index, op_node, graph, (input_node,))
        req.params["kernel_size"] = op_node.p("kernel_size", (1, 1, 1))
        req.params["stride"] = op_node.p("stride", (1, 1, 1))
        req.params["padding"] = op_node.p("padding", (0, 0, 0))
        req.params["dilation"] = op_node.p("dilation", (1, 1, 1))
        req.params["ceil_mode"] = bool(op_node.p("ceil_mode", False))
        req.params["count_include_pad"] = bool(
            op_node.p("count_include_pad", False)
        )
        req.params["divisor_override"] = op_node.p("divisor_override")
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        return _pool3d_output_shape_from_shapes(
            input_shapes[0],
            op_node.p("kernel_size", (1, 1, 1)),
            op_node.p("stride", (1, 1, 1)),
            op_node.p("padding", (0, 0, 0)),
            op_node.p("dilation", (1, 1, 1)),
        )


class Pool2dBackwardHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        grad_output_node, input_node = op_node.inputs
        req = _make_request(
            node_index, op_node, graph, (grad_output_node, input_node)
        )
        req.params["kernel_size"] = op_node.p("kernel_size", (1, 1))
        req.params["stride"] = op_node.p("stride", (1, 1))
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        _grad_output_shape, input_shape = input_shapes
        return input_shape


class Pool1dHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        input_node = op_node.inputs[0]
        req = _make_request(node_index, op_node, graph, (input_node,))
        req.params["kernel_size"] = op_node.p("kernel_size", 1)
        req.params["stride"] = op_node.p("stride", 1)
        req.params["padding"] = op_node.p("padding", 0)
        req.params["dilation"] = op_node.p("dilation", 1)
        req.params["ceil_mode"] = bool(op_node.p("ceil_mode", False))
        req.params["count_include_pad"] = bool(
            op_node.p("count_include_pad", False)
        )
        req.params["divisor_override"] = op_node.p("divisor_override")
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        return _pool1d_output_shape_from_shapes(
            input_shapes[0],
            op_node.p("kernel_size", 1),
            op_node.p("stride", 1),
            op_node.p("padding", 0),
            op_node.p("dilation", 1),
            bool(op_node.p("ceil_mode", False)),
        )


class Col2imHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        input_node = op_node.inputs[0]
        req = _make_request(node_index, op_node, graph, (input_node,))
        req.params["output_size"] = op_node.p("output_size", (1, 1))
        req.params["kernel_size"] = op_node.p("kernel_size", (1, 1))
        req.params["dilation"] = op_node.p("dilation", (1, 1))
        req.params["padding"] = op_node.p("padding", (0, 0))
        req.params["stride"] = op_node.p("stride", (1, 1))
        req.params["out_blocks_h"] = op_node.p("out_blocks_h", 1)
        req.params["out_blocks_w"] = op_node.p("out_blocks_w", 1)
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        input_shape = input_shapes[0]
        output_pair = op_node.p("output_size", (1, 1))
        kernel_pair = op_node.p("kernel_size", (1, 1))
        dilation_pair = op_node.p("dilation", (1, 1))
        padding_pair = op_node.p("padding", (0, 0))
        stride_pair = op_node.p("stride", (1, 1))
        k_h, k_w = kernel_pair
        channels_divisor = k_h * k_w
        if len(input_shape) == 3:
            batch, col_channels, col_length = input_shape
            has_batch = True
        else:
            col_channels, col_length = input_shape
            batch = 1
            has_batch = False
        if channels_divisor <= 0 or col_channels % channels_divisor != 0:
            raise RefBackendError(
                "codegen col2im expects input channels divisible by kernel_size"
            )
        out_h, out_w = output_pair
        dil_h, dil_w = dilation_pair
        pad_h, pad_w = padding_pair
        stride_h, stride_w = stride_pair
        effective_kh = dil_h * (k_h - 1) + 1
        effective_kw = dil_w * (k_w - 1) + 1
        numerator_h = out_h + 2 * pad_h - effective_kh
        numerator_w = out_w + 2 * pad_w - effective_kw
        if (
            numerator_h < 0
            or numerator_w < 0
            or numerator_h % stride_h != 0
            or numerator_w % stride_w != 0
        ):
            raise RefBackendError(
                "codegen col2im expects output_size to be compatible with kernel_size, dilation, padding, and stride"
            )
        out_blocks_h = numerator_h // stride_h + 1
        out_blocks_w = numerator_w // stride_w + 1
        expected_length = out_blocks_h * out_blocks_w
        if col_length != expected_length:
            raise RefBackendError(
                "codegen col2im expects input length to match output_size and stride"
            )
        op_node.params["out_blocks_h"] = out_blocks_h
        op_node.params["out_blocks_w"] = out_blocks_w
        channels = col_channels // channels_divisor
        if has_batch:
            return (batch, channels, out_h, out_w)
        return (channels, out_h, out_w)


class BatchNormHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        input_node = op_node.inputs[0]
        req = _make_request(node_index, op_node, graph, (input_node,))
        req.params["eps"] = float(op_node.p("eps", 1e-5))
        req.params["has_weight"] = bool(op_node.p("has_weight", False))
        req.params["has_bias"] = bool(op_node.p("has_bias", False))
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        return input_shapes[0]


class PdistHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        input_node = op_node.inputs[0]
        req = _make_request(node_index, op_node, graph, (input_node,))
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        if len(input_shapes[0]) != 2:
            raise RefBackendError(
                "codegen pdist expects a 2D input tensor"
            )
        n = input_shapes[0][0]
        return (n * (n - 1) // 2,)


class CdistHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        x1_node, x2_node = op_node.inputs
        req = _make_request(node_index, op_node, graph, (x1_node, x2_node))
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        if len(input_shapes) < 2:
            raise RefBackendError(
                "codegen cdist expects two input tensors"
            )
        x1_shape, x2_shape = input_shapes[:2]
        if len(x1_shape) != 2 or len(x2_shape) != 2:
            raise RefBackendError(
                "codegen cdist expects 2D input tensors"
            )
        if x1_shape[1] != x2_shape[1]:
            raise RefBackendError(
                "codegen cdist expects matching feature dimensions"
            )
        return (x1_shape[0], x2_shape[0])


class Conv1dHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        input_node, weight_node, *_ = op_node.inputs
        req = _make_request(node_index, op_node, graph, (input_node, weight_node))
        req.params["stride"] = op_node.p("stride", 1)
        req.params["padding"] = op_node.p("padding", 0)
        req.params["dilation"] = op_node.p("dilation", 1)
        req.params["groups"] = op_node.p("groups", 1)
        req.params["has_bias"] = bool(op_node.p("has_bias", False))
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
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


class Conv2dHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        input_node, weight_node, *_ = op_node.inputs
        req = _make_request(node_index, op_node, graph, (input_node, weight_node))
        req.params["transposed"] = bool(op_node.p("transposed", False))
        req.params["stride"] = op_node.p("stride", (1, 1))
        req.params["padding"] = op_node.p("padding", (0, 0))
        req.params["dilation"] = op_node.p("dilation", (1, 1))
        req.params["groups"] = op_node.p("groups", 1)
        req.params["has_bias"] = bool(op_node.p("has_bias", False))
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
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


class AddmmHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        input_node, mat1_node, mat2_node = op_node.inputs
        req = _make_request(
            node_index, op_node, graph, (input_node, mat1_node, mat2_node)
        )
        req.params["alpha"] = float(op_node.p("alpha", 1.0))
        req.params["beta"] = float(op_node.p("beta", 1.0))
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        input_shape, mat1_shape, mat2_shape = input_shapes
        if len(input_shape) > 2 or len(mat1_shape) != 2 or len(mat2_shape) != 2:
            raise RefBackendError("codegen addmm expects 2D inputs")
        if mat1_shape[1] != mat2_shape[0]:
            raise RefBackendError(
                "codegen addmm requires inner dimensions to match"
            )
        expected_shape = (mat1_shape[0], mat2_shape[1])
        if (
            shape_utils.broadcast_output_shape(
                op_node.spec.name, input_shape, expected_shape
            )
            != expected_shape
        ):
            raise RefBackendError(
                "codegen addmm expects input shape to be broadcastable to matmul output"
            )
        return expected_shape


class AddbmmHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        input_node, batch1_node, batch2_node = op_node.inputs
        req = _make_request(
            node_index, op_node, graph, (input_node, batch1_node, batch2_node)
        )
        req.params["alpha"] = float(op_node.p("alpha", 1.0))
        req.params["beta"] = float(op_node.p("beta", 1.0))
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        input_shape, batch1_shape, batch2_shape = input_shapes
        if (
            len(input_shape) > 2
            or len(batch1_shape) != 3
            or len(batch2_shape) != 3
        ):
            raise RefBackendError(
                "codegen addbmm expects 0-2D input and 3D batches"
            )
        if batch1_shape[0] != batch2_shape[0]:
            raise RefBackendError(
                "codegen addbmm requires batch dimensions to match"
            )
        if batch1_shape[2] != batch2_shape[1]:
            raise RefBackendError(
                "codegen addbmm requires inner dimensions to match"
            )
        expected_shape = (batch1_shape[1], batch2_shape[2])
        if (
            shape_utils.broadcast_output_shape(
                op_node.spec.name, input_shape, expected_shape
            )
            != expected_shape
        ):
            raise RefBackendError(
                "codegen addbmm expects input shape to be broadcastable to bmm output"
            )
        return expected_shape


class AddmvHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        input_node, mat_node, vec_node = op_node.inputs
        req = _make_request(
            node_index, op_node, graph, (input_node, mat_node, vec_node)
        )
        req.params["alpha"] = float(op_node.p("alpha", 1.0))
        req.params["beta"] = float(op_node.p("beta", 1.0))
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        input_shape, mat_shape, vec_shape = input_shapes
        if len(input_shape) not in (0, 1) or len(mat_shape) != 2 or len(vec_shape) != 1:
            raise RefBackendError(
                "codegen addmv expects a scalar or 1D input and 2D matrix/1D vector"
            )
        if mat_shape[1] != vec_shape[0]:
            raise RefBackendError(
                "codegen addmv requires inner dimensions to match"
            )
        expected_shape = (mat_shape[0],)
        if (
            shape_utils.broadcast_output_shape(
                op_node.spec.name, input_shape, expected_shape
            )
            != expected_shape
        ):
            raise RefBackendError(
                "codegen addmv expects input shape to be broadcastable to mat-vec output"
            )
        return expected_shape


class AddrHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        input_node, vec1_node, vec2_node = op_node.inputs
        req = _make_request(
            node_index, op_node, graph, (input_node, vec1_node, vec2_node)
        )
        req.params["alpha"] = float(op_node.p("alpha", 1.0))
        req.params["beta"] = float(op_node.p("beta", 1.0))
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        input_shape, vec1_shape, vec2_shape = input_shapes
        if len(vec1_shape) != 1 or len(vec2_shape) != 1:
            raise RefBackendError(
                "codegen addr expects 1D vectors"
            )
        if len(input_shape) > 2:
            raise RefBackendError(
                "codegen addr expects input with rank <= 2"
            )
        expected_shape = (vec1_shape[0], vec2_shape[0])
        if (
            shape_utils.broadcast_output_shape(
                op_node.spec.name, input_shape, expected_shape
            )
            != expected_shape
        ):
            raise RefBackendError(
                "codegen addr expects input shape to be broadcastable to outer product output"
            )
        return expected_shape


class MatmulHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        lhs, rhs = op_node.inputs
        req = _make_request(node_index, op_node, graph, (lhs, rhs))
        return self._ctx.emit_kernel(op_node.spec.kind, req)

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        op_spec = op_node.spec
        a_shape, b_shape = input_shapes
        if op_spec.name == "matmul":
            if len(a_shape) == 1 and len(b_shape) == 1:
                if a_shape[0] != b_shape[0]:
                    raise RefBackendError(
                        "codegen matmul requires inner dimensions to match"
                    )
                return ()
            if len(a_shape) != 2 or len(b_shape) != 2:
                raise RefBackendError(
                    "codegen matmul requires 1D or 2D inputs"
                )
            if a_shape[1] != b_shape[0]:
                raise RefBackendError(
                    "codegen matmul requires inner dimensions to match"
                )
            return (a_shape[0], b_shape[1])
        if len(a_shape) != 3 or len(b_shape) != 3:
            raise RefBackendError("codegen bmm requires 3D inputs")
        if a_shape[0] != b_shape[0]:
            raise RefBackendError(
                "codegen bmm requires batch dimensions to match"
            )
        if a_shape[2] != b_shape[1]:
            raise RefBackendError(
                "codegen bmm requires inner dimensions to match"
            )
        return (a_shape[0], a_shape[1], b_shape[2])


def build_kind_handlers(context: HandlerContext) -> Dict[str, KindHandler]:
    elementwise = ElementwiseHandler(context)
    return {
        "arange": ArangeHandler(context),
        "binary": elementwise,
        "unary": elementwise,
        "where": elementwise,
        "fill": elementwise,
        "flip": FlipHandler(context),
        "pad": PadHandler(context),
        "view": ViewHandler(context),
        "resize": ResizeHandler(context),
        "empty_strided": EmptyStridedHandler(context),
        "diagonal": DiagonalHandler(context),
        "reduction": ReductionHandler(context),
        "arg_reduction": ArgReductionHandler(context),
        "softmax": SoftmaxHandler(context),
        "cumsum": CumsumHandler(context),
        "embedding": EmbeddingHandler(context),
        "embedding_bag": EmbeddingBagHandler(context),
        "gather": GatherHandler(context),
        "concat": ConcatHandler(context),
        "pool2d": Pool2dHandler(context),
        "pool3d": Pool3dHandler(context),
        "pool2d_backward": Pool2dBackwardHandler(context),
        "pool1d": Pool1dHandler(context),
        "col2im": Col2imHandler(context),
        "batch_norm": BatchNormHandler(context),
        "pdist": PdistHandler(context),
        "cdist": CdistHandler(context),
        "conv1d": Conv1dHandler(context),
        "conv2d": Conv2dHandler(context),
        "addmm": AddmmHandler(context),
        "addbmm": AddbmmHandler(context),
        "addmv": AddmvHandler(context),
        "addr": AddrHandler(context),
        "matmul": MatmulHandler(context),
    }
