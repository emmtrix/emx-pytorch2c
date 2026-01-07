from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Protocol, Sequence, Tuple

from c_ref_backend.cffi_bindings import RefBackendError
from codegen_backend import shape_inference, shape_utils

if TYPE_CHECKING:
    import torch
    from codegen_backend.graph import _GenericGraph, _OpNode
    from codegen_backend.specs import _OpSpec

class HandlerContext(Protocol):
    def kernel_inputs(self, op_node: "_OpNode") -> List["torch.fx.Node"]: ...

    def write_arange_kernel(
        self,
        node_index: int,
        op_node: "_OpNode",
        output_shape: Sequence[int],
        output_strides: Sequence[int],
        dtype: object,
    ) -> List[str]: ...

    def write_elementwise_kernel(
        self,
        node_index: int,
        op_node: "_OpNode",
        output_shape: Sequence[int],
        input_shapes: Sequence[Sequence[int]],
        input_strides: Sequence[Sequence[int]],
        input_dtypes: Sequence[object],
        output_strides: Sequence[int],
        dtype: object,
    ) -> List[str]: ...

    def write_flip_kernel(
        self,
        node_index: int,
        op_node: "_OpNode",
        input_shape: Sequence[int],
        input_strides: Sequence[int],
        input_dtype: object,
        output_shape: Sequence[int],
        output_strides: Sequence[int],
        dtype: object,
    ) -> List[str]: ...

    def write_constant_pad_kernel(
        self,
        node_index: int,
        op_node: "_OpNode",
        input_shape: Sequence[int],
        input_strides: Sequence[int],
        input_dtype: object,
        output_shape: Sequence[int],
        output_strides: Sequence[int],
        dtype: object,
    ) -> List[str]: ...

    def write_view_kernel(
        self,
        node_index: int,
        op_node: "_OpNode",
        input_shape: Sequence[int],
        input_dtype: object,
        output_shape: Sequence[int],
        output_strides: Sequence[int],
        dtype: object,
    ) -> List[str]: ...

    def write_resize_kernel(
        self,
        node_index: int,
        op_node: "_OpNode",
        input_shape: Sequence[int],
        input_strides: Sequence[int],
        input_dtype: object,
        output_shape: Sequence[int],
        output_strides: Sequence[int],
        dtype: object,
    ) -> List[str]: ...

    def write_empty_strided_kernel(
        self,
        node_index: int,
        op_spec: "_OpSpec",
        output_shape: Sequence[int],
        dtype: object,
    ) -> List[str]: ...

    def write_diagonal_kernel(
        self,
        node_index: int,
        op_node: "_OpNode",
        input_shape: Sequence[int],
        input_strides: Sequence[int],
        input_dtype: object,
        output_shape: Sequence[int],
        output_strides: Sequence[int],
        dtype: object,
    ) -> List[str]: ...

    def write_std_kernel(
        self,
        node_index: int,
        op_spec: "_OpSpec",
        input_shape: Sequence[int],
        input_strides: Sequence[int],
        output_shape: Sequence[int],
        output_strides: Sequence[int],
        reduction_dims: Sequence[int],
        keepdim: bool,
        dtype: object,
        *,
        unbiased: bool,
    ) -> List[str]: ...

    def write_var_kernel(
        self,
        node_index: int,
        op_spec: "_OpSpec",
        input_shape: Sequence[int],
        input_strides: Sequence[int],
        output_shape: Sequence[int],
        output_strides: Sequence[int],
        reduction_dims: Sequence[int],
        keepdim: bool,
        dtype: object,
        *,
        unbiased: bool,
    ) -> List[str]: ...

    def write_norm_kernel(
        self,
        node_index: int,
        op_spec: "_OpSpec",
        input_shape: Sequence[int],
        input_strides: Sequence[int],
        output_shape: Sequence[int],
        output_strides: Sequence[int],
        reduction_dims: Sequence[int],
        keepdim: bool,
        dtype: object,
        *,
        p_value: float,
    ) -> List[str]: ...

    def write_reduction_kernel(
        self,
        node_index: int,
        op_spec: "_OpSpec",
        input_shape: Sequence[int],
        input_strides: Sequence[int],
        output_shape: Sequence[int],
        output_strides: Sequence[int],
        reduction_dims: Sequence[int],
        keepdim: bool,
        dtype: object,
    ) -> List[str]: ...

    def write_argminmax_kernel(
        self,
        node_index: int,
        op_spec: "_OpSpec",
        input_shape: Sequence[int],
        input_strides: Sequence[int],
        output_shape: Sequence[int],
        output_strides: Sequence[int],
        reduction_dims: Sequence[int],
        keepdim: bool,
        reduce_all: bool,
        dtype: object,
    ) -> List[str]: ...

    def write_softmax_kernel(
        self,
        node_index: int,
        op_spec: "_OpSpec",
        input_shape: Sequence[int],
        input_strides: Sequence[int],
        output_strides: Sequence[int],
        dim: int,
        dtype: object,
    ) -> List[str]: ...

    def write_cumsum_kernel(
        self,
        node_index: int,
        op_spec: "_OpSpec",
        input_shape: Sequence[int],
        input_strides: Sequence[int],
        output_strides: Sequence[int],
        dim: int,
        dtype: object,
        output_dtype: object,
    ) -> List[str]: ...

    def write_embedding_kernel(
        self,
        node_index: int,
        op_spec: "_OpSpec",
        weight_shape: Sequence[int],
        indices_shape: Sequence[int],
        weight_strides: Sequence[int],
        indices_strides: Sequence[int],
        output_shape: Sequence[int],
        output_strides: Sequence[int],
        indices_dtype: object,
        dtype: object,
        *,
        padding_idx: int,
    ) -> List[str]: ...

    def write_embedding_bag_kernel(
        self,
        node_index: int,
        op_spec: "_OpSpec",
        weight_shape: Sequence[int],
        indices_shape: Sequence[int],
        offsets_shape: Sequence[int],
        weight_strides: Sequence[int],
        indices_strides: Sequence[int],
        offsets_strides: Sequence[int],
        output_shape: Sequence[int],
        output_strides: Sequence[int],
        indices_dtype: object,
        offsets_dtype: object,
        dtype: object,
        *,
        mode: int,
        padding_idx: int,
        include_last_offset: bool,
    ) -> List[str]: ...

    def write_gather_kernel(
        self,
        node_index: int,
        op_spec: "_OpSpec",
        input_shape: Sequence[int],
        index_shape: Sequence[int],
        input_strides: Sequence[int],
        index_strides: Sequence[int],
        output_shape: Sequence[int],
        output_strides: Sequence[int],
        index_dtype: object,
        dim: int,
        dtype: object,
    ) -> List[str]: ...

    def write_concat_kernel(
        self,
        node_index: int,
        op_spec: "_OpSpec",
        input_shapes: Sequence[Sequence[int]],
        input_strides: Sequence[Sequence[int]],
        output_shape: Sequence[int],
        output_strides: Sequence[int],
        dim: int,
        dtype: object,
    ) -> List[str]: ...

    def write_pool2d_kernel(
        self,
        node_index: int,
        op_spec: "_OpSpec",
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        dilation: Tuple[int, int],
        dtype: object,
        ceil_mode: bool,
        count_include_pad: bool,
        divisor_override: object,
    ) -> List[str]: ...

    def write_pool3d_kernel(
        self,
        node_index: int,
        op_spec: "_OpSpec",
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        kernel_size: Tuple[int, int, int],
        stride: Tuple[int, int, int],
        padding: Tuple[int, int, int],
        dilation: Tuple[int, int, int],
        dtype: object,
        ceil_mode: bool,
        count_include_pad: bool,
        divisor_override: object,
    ) -> List[str]: ...

    def write_adaptive_avg_pool2d_backward_kernel(
        self,
        node_index: int,
        op_spec: "_OpSpec",
        grad_output_shape: Sequence[int],
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        dtype: object,
    ) -> List[str]: ...

    def write_pool1d_kernel(
        self,
        node_index: int,
        op_spec: "_OpSpec",
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        dtype: object,
        ceil_mode: bool,
        count_include_pad: bool,
        divisor_override: object,
    ) -> List[str]: ...

    def write_col2im_kernel(
        self,
        node_index: int,
        op_spec: "_OpSpec",
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        output_size: Tuple[int, int],
        kernel_size: Tuple[int, int],
        dilation: Tuple[int, int],
        padding: Tuple[int, int],
        stride: Tuple[int, int],
        dtype: object,
        out_blocks_h: int,
        out_blocks_w: int,
    ) -> List[str]: ...

    def write_batch_norm_kernel(
        self,
        node_index: int,
        op_spec: "_OpSpec",
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        dtype: object,
        eps: float,
        has_weight: bool,
        has_bias: bool,
    ) -> List[str]: ...

    def write_pdist_kernel(
        self,
        node_index: int,
        op_spec: "_OpSpec",
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        dtype: object,
    ) -> List[str]: ...

    def write_cdist_kernel(
        self,
        node_index: int,
        op_spec: "_OpSpec",
        x1_shape: Sequence[int],
        x2_shape: Sequence[int],
        output_shape: Sequence[int],
        dtype: object,
    ) -> List[str]: ...

    def write_conv1d_kernel(
        self,
        node_index: int,
        op_spec: "_OpSpec",
        input_shape: Sequence[int],
        weight_shape: Sequence[int],
        output_shape: Sequence[int],
        stride: int,
        padding: int,
        dilation: int,
        groups: int,
        dtype: object,
        has_bias: bool,
    ) -> List[str]: ...

    def write_conv2d_kernel(
        self,
        node_index: int,
        op_spec: "_OpSpec",
        input_shape: Sequence[int],
        weight_shape: Sequence[int],
        output_shape: Sequence[int],
        transposed: bool,
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        dilation: Tuple[int, int],
        groups: int,
        dtype: object,
        has_bias: bool,
    ) -> List[str]: ...

    def write_addmm_kernel(
        self,
        node_index: int,
        op_spec: "_OpSpec",
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        mat1_shape: Sequence[int],
        mat2_shape: Sequence[int],
        input_strides: Sequence[int],
        mat1_strides: Sequence[int],
        mat2_strides: Sequence[int],
        output_strides: Sequence[int],
        dtype: object,
        *,
        alpha: float,
        beta: float,
    ) -> List[str]: ...

    def write_addbmm_kernel(
        self,
        node_index: int,
        op_spec: "_OpSpec",
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        batch1_shape: Sequence[int],
        batch2_shape: Sequence[int],
        input_strides: Sequence[int],
        batch1_strides: Sequence[int],
        batch2_strides: Sequence[int],
        output_strides: Sequence[int],
        dtype: object,
        *,
        alpha: float,
        beta: float,
    ) -> List[str]: ...

    def write_addmv_kernel(
        self,
        node_index: int,
        op_spec: "_OpSpec",
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        mat_shape: Sequence[int],
        vec_shape: Sequence[int],
        input_strides: Sequence[int],
        mat_strides: Sequence[int],
        vec_strides: Sequence[int],
        output_strides: Sequence[int],
        dtype: object,
        *,
        alpha: float,
        beta: float,
    ) -> List[str]: ...

    def write_addr_kernel(
        self,
        node_index: int,
        op_spec: "_OpSpec",
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        vec1_shape: Sequence[int],
        vec2_shape: Sequence[int],
        input_strides: Sequence[int],
        vec1_strides: Sequence[int],
        vec2_strides: Sequence[int],
        output_strides: Sequence[int],
        dtype: object,
        *,
        alpha: float,
        beta: float,
    ) -> List[str]: ...

    def write_matmul_kernel(
        self,
        node_index: int,
        op_spec: "_OpSpec",
        a_shape: Sequence[int],
        b_shape: Sequence[int],
        a_strides: Sequence[int],
        b_strides: Sequence[int],
        dtype: object,
    ) -> List[str]: ...


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


class ArangeHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        return self._ctx.write_arange_kernel(
            node_index,
            op_node,
            op_node.output_shape,
            graph.strides[op_node.node],
            graph.dtype,
        )

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        output_size = shape_inference.compute_arange_size(
            op_node.p("start"), op_node.p("end"), op_node.p("step")
        )
        return (output_size,)


class ElementwiseHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        kernel_inputs = self._ctx.kernel_inputs(op_node)
        input_shapes = [graph.shapes[arg] for arg in kernel_inputs]
        input_strides = [graph.strides[arg] for arg in kernel_inputs]
        input_dtypes = [graph.dtypes[arg] for arg in kernel_inputs]
        output_strides = graph.strides[op_node.node]
        return self._ctx.write_elementwise_kernel(
            node_index,
            op_node,
            op_node.output_shape,
            input_shapes,
            input_strides,
            input_dtypes,
            output_strides,
            graph.dtype,
        )

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
        return self._ctx.write_flip_kernel(
            node_index,
            op_node,
            graph.shapes[input_node],
            graph.strides[input_node],
            graph.dtypes[input_node],
            op_node.output_shape,
            graph.strides[op_node.node],
            graph.dtype,
        )

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
        return self._ctx.write_constant_pad_kernel(
            node_index,
            op_node,
            graph.shapes[input_node],
            graph.strides[input_node],
            graph.dtypes[input_node],
            op_node.output_shape,
            graph.strides[op_node.node],
            graph.dtype,
        )

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
        return self._ctx.write_view_kernel(
            node_index,
            op_node,
            graph.shapes[input_node],
            graph.dtypes[input_node],
            op_node.output_shape,
            graph.strides[op_node.node],
            graph.dtype,
        )

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
        return self._ctx.write_resize_kernel(
            node_index,
            op_node,
            graph.shapes[input_node],
            graph.strides[input_node],
            graph.dtypes[input_node],
            op_node.output_shape,
            graph.strides[op_node.node],
            graph.dtype,
        )

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
        return self._ctx.write_empty_strided_kernel(
            node_index,
            op_node.spec,
            op_node.output_shape,
            graph.dtype,
        )

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
        return self._ctx.write_diagonal_kernel(
            node_index,
            op_node,
            graph.shapes[input_node],
            graph.strides[input_node],
            graph.dtypes[input_node],
            op_node.output_shape,
            graph.strides[op_node.node],
            graph.dtype,
        )

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        return shape_inference.infer_diagonal_output_shape(
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
        if op_node.spec.name == "std":
            return self._ctx.write_std_kernel(
                node_index,
                op_node.spec,
                graph.shapes[input_node],
                graph.strides[input_node],
                op_node.output_shape,
                graph.strides[op_node.node],
                op_node.reduction_dims or (),
                op_node.keepdim,
                graph.dtype,
                unbiased=bool(op_node.p("unbiased", True)),
            )
        if op_node.spec.name == "var":
            return self._ctx.write_var_kernel(
                node_index,
                op_node.spec,
                graph.shapes[input_node],
                graph.strides[input_node],
                op_node.output_shape,
                graph.strides[op_node.node],
                op_node.reduction_dims or (),
                op_node.keepdim,
                graph.dtype,
                unbiased=bool(op_node.p("unbiased", True)),
            )
        if op_node.spec.name == "norm":
            return self._ctx.write_norm_kernel(
                node_index,
                op_node.spec,
                graph.shapes[input_node],
                graph.strides[input_node],
                op_node.output_shape,
                graph.strides[op_node.node],
                op_node.reduction_dims or (),
                op_node.keepdim,
                graph.dtype,
                p_value=float(op_node.p("norm_p", 2.0)),
            )
        return self._ctx.write_reduction_kernel(
            node_index,
            op_node.spec,
            graph.shapes[input_node],
            graph.strides[input_node],
            op_node.output_shape,
            graph.strides[op_node.node],
            op_node.reduction_dims or (),
            op_node.keepdim,
            graph.dtype,
        )

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        return shape_inference.infer_reduction_output_shape(
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
        return self._ctx.write_argminmax_kernel(
            node_index,
            op_node.spec,
            graph.shapes[input_node],
            graph.strides[input_node],
            op_node.output_shape,
            graph.strides[op_node.node],
            op_node.reduction_dims or (),
            op_node.keepdim,
            bool(op_node.p("reduce_all", False)),
            graph.dtype,
        )

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        return shape_inference.infer_reduction_output_shape(
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
        return self._ctx.write_softmax_kernel(
            node_index,
            op_node.spec,
            graph.shapes[input_node],
            graph.strides[input_node],
            graph.strides[op_node.node],
            op_node.p("dim"),
            graph.dtype,
        )

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
        return self._ctx.write_cumsum_kernel(
            node_index,
            op_node.spec,
            graph.shapes[input_node],
            graph.strides[input_node],
            graph.strides[op_node.node],
            op_node.p("dim"),
            graph.dtype,
            graph.dtypes[op_node.node],
        )

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
        return self._ctx.write_embedding_kernel(
            node_index,
            op_node.spec,
            graph.shapes[weight_node],
            graph.shapes[indices_node],
            graph.strides[weight_node],
            graph.strides[indices_node],
            op_node.output_shape,
            graph.strides[op_node.node],
            graph.dtypes[indices_node],
            graph.dtype,
            padding_idx=int(op_node.p("padding_idx", -1)),
        )

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
        return self._ctx.write_embedding_bag_kernel(
            node_index,
            op_node.spec,
            graph.shapes[weight_node],
            graph.shapes[indices_node],
            graph.shapes[offsets_node],
            graph.strides[weight_node],
            graph.strides[indices_node],
            graph.strides[offsets_node],
            op_node.output_shape,
            graph.strides[op_node.node],
            graph.dtypes[indices_node],
            graph.dtypes[offsets_node],
            graph.dtype,
            mode=int(op_node.p("mode", 0)),
            padding_idx=int(op_node.p("padding_idx", -1)),
            include_last_offset=bool(op_node.p("include_last_offset", False)),
        )

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
        return self._ctx.write_gather_kernel(
            node_index,
            op_node.spec,
            graph.shapes[input_node],
            graph.shapes[index_node],
            graph.strides[input_node],
            graph.strides[index_node],
            op_node.output_shape,
            graph.strides[op_node.node],
            graph.dtypes[index_node],
            int(op_node.p("dim")),
            graph.dtype,
        )

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
        input_shapes = [graph.shapes[arg] for arg in op_node.inputs]
        input_strides = [graph.strides[arg] for arg in op_node.inputs]
        return self._ctx.write_concat_kernel(
            node_index,
            op_node.spec,
            input_shapes,
            input_strides,
            op_node.output_shape,
            graph.strides[op_node.node],
            op_node.p("dim", 0),
            graph.dtype,
        )

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
        return self._ctx.write_pool2d_kernel(
            node_index,
            op_node.spec,
            graph.shapes[input_node],
            op_node.output_shape,
            op_node.p("kernel_size", (1, 1)),
            op_node.p("stride", (1, 1)),
            op_node.p("padding", (0, 0)),
            op_node.p("dilation", (1, 1)),
            graph.dtype,
            bool(op_node.p("ceil_mode", False)),
            bool(op_node.p("count_include_pad", False)),
            op_node.p("divisor_override"),
        )

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        return shape_inference.pool2d_output_shape_from_shapes(
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
        return self._ctx.write_pool3d_kernel(
            node_index,
            op_node.spec,
            graph.shapes[input_node],
            op_node.output_shape,
            op_node.p("kernel_size", (1, 1, 1)),
            op_node.p("stride", (1, 1, 1)),
            op_node.p("padding", (0, 0, 0)),
            op_node.p("dilation", (1, 1, 1)),
            graph.dtype,
            bool(op_node.p("ceil_mode", False)),
            bool(op_node.p("count_include_pad", False)),
            op_node.p("divisor_override"),
        )

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        return shape_inference.pool3d_output_shape_from_shapes(
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
        return self._ctx.write_adaptive_avg_pool2d_backward_kernel(
            node_index,
            op_node.spec,
            graph.shapes[grad_output_node],
            graph.shapes[input_node],
            op_node.output_shape,
            op_node.p("kernel_size", (1, 1)),
            op_node.p("stride", (1, 1)),
            graph.dtype,
        )

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
        return self._ctx.write_pool1d_kernel(
            node_index,
            op_node.spec,
            graph.shapes[input_node],
            op_node.output_shape,
            op_node.p("kernel_size", 1),
            op_node.p("stride", 1),
            op_node.p("padding", 0),
            op_node.p("dilation", 1),
            graph.dtype,
            bool(op_node.p("ceil_mode", False)),
            bool(op_node.p("count_include_pad", False)),
            op_node.p("divisor_override"),
        )

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        return shape_inference.pool1d_output_shape_from_shapes(
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
        return self._ctx.write_col2im_kernel(
            node_index,
            op_node.spec,
            graph.shapes[input_node],
            op_node.output_shape,
            op_node.p("output_size", (1, 1)),
            op_node.p("kernel_size", (1, 1)),
            op_node.p("dilation", (1, 1)),
            op_node.p("padding", (0, 0)),
            op_node.p("stride", (1, 1)),
            graph.dtype,
            op_node.p("out_blocks_h", 1),
            op_node.p("out_blocks_w", 1),
        )

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
        return self._ctx.write_batch_norm_kernel(
            node_index,
            op_node.spec,
            graph.shapes[input_node],
            op_node.output_shape,
            graph.dtype,
            float(op_node.p("eps", 1e-5)),
            bool(op_node.p("has_weight", False)),
            bool(op_node.p("has_bias", False)),
        )

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
        return self._ctx.write_pdist_kernel(
            node_index,
            op_node.spec,
            graph.shapes[input_node],
            op_node.output_shape,
            graph.dtype,
        )

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
        return self._ctx.write_cdist_kernel(
            node_index,
            op_node.spec,
            graph.shapes[x1_node],
            graph.shapes[x2_node],
            op_node.output_shape,
            graph.dtype,
        )

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
        return self._ctx.write_conv1d_kernel(
            node_index,
            op_node.spec,
            graph.shapes[input_node],
            graph.shapes[weight_node],
            op_node.output_shape,
            op_node.p("stride", 1),
            op_node.p("padding", 0),
            op_node.p("dilation", 1),
            op_node.p("groups", 1),
            graph.dtype,
            bool(op_node.p("has_bias", False)),
        )

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
                return shape_inference.conv1d_output_shape_from_shapes(
                    input_shape,
                    weight_shape,
                    stride,
                    padding_value,
                    dilation,
                    groups,
                )
            padding_value, out_l = shape_inference.conv1d_same_padding(
                input_shape,
                weight_shape,
                stride,
                dilation,
            )
            op_node.params["padding"] = padding_value
            batch, out_channels = shape_inference.conv1d_validate_channels(
                input_shape,
                weight_shape,
                groups,
            )
            return batch, out_channels, out_l
        return shape_inference.conv1d_output_shape_from_shapes(
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
        return self._ctx.write_conv2d_kernel(
            node_index,
            op_node.spec,
            graph.shapes[input_node],
            graph.shapes[weight_node],
            op_node.output_shape,
            bool(op_node.p("transposed", False)),
            op_node.p("stride", (1, 1)),
            op_node.p("padding", (0, 0)),
            op_node.p("dilation", (1, 1)),
            op_node.p("groups", 1),
            graph.dtype,
            bool(op_node.p("has_bias", False)),
        )

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
                return shape_inference.conv2d_output_shape_from_shapes(
                    input_shape,
                    weight_shape,
                    stride,
                    padding_value,
                    dilation,
                    groups,
                )
            has_batch, out_channels = shape_inference.conv2d_validate_channels(
                input_shape, weight_shape, groups
            )
            padding_value, (out_h, out_w) = shape_inference.conv2d_same_padding(
                input_shape, weight_shape, stride, dilation
            )
            op_node.params["padding"] = padding_value
            if has_batch:
                return (input_shape[0], out_channels, out_h, out_w)
            return (out_channels, out_h, out_w)
        if transposed:
            return shape_inference.conv2d_transposed_output_shape_from_shapes(
                input_shape,
                weight_shape,
                stride,
                padding,
                dilation,
                output_padding,
                groups,
            )
        return shape_inference.conv2d_output_shape_from_shapes(
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
        return self._ctx.write_addmm_kernel(
            node_index,
            op_node.spec,
            graph.shapes[input_node],
            graph.shapes[op_node.node],
            graph.shapes[mat1_node],
            graph.shapes[mat2_node],
            graph.strides[input_node],
            graph.strides[mat1_node],
            graph.strides[mat2_node],
            graph.strides[op_node.node],
            graph.dtype,
            alpha=float(op_node.p("alpha", 1.0)),
            beta=float(op_node.p("beta", 1.0)),
        )

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
        return self._ctx.write_addbmm_kernel(
            node_index,
            op_node.spec,
            graph.shapes[input_node],
            graph.shapes[op_node.node],
            graph.shapes[batch1_node],
            graph.shapes[batch2_node],
            graph.strides[input_node],
            graph.strides[batch1_node],
            graph.strides[batch2_node],
            graph.strides[op_node.node],
            graph.dtype,
            alpha=float(op_node.p("alpha", 1.0)),
            beta=float(op_node.p("beta", 1.0)),
        )

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
        return self._ctx.write_addmv_kernel(
            node_index,
            op_node.spec,
            graph.shapes[input_node],
            graph.shapes[op_node.node],
            graph.shapes[mat_node],
            graph.shapes[vec_node],
            graph.strides[input_node],
            graph.strides[mat_node],
            graph.strides[vec_node],
            graph.strides[op_node.node],
            graph.dtype,
            alpha=float(op_node.p("alpha", 1.0)),
            beta=float(op_node.p("beta", 1.0)),
        )

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
        return self._ctx.write_addr_kernel(
            node_index,
            op_node.spec,
            graph.shapes[input_node],
            graph.shapes[op_node.node],
            graph.shapes[vec1_node],
            graph.shapes[vec2_node],
            graph.strides[input_node],
            graph.strides[vec1_node],
            graph.strides[vec2_node],
            graph.strides[op_node.node],
            graph.dtype,
            alpha=float(op_node.p("alpha", 1.0)),
            beta=float(op_node.p("beta", 1.0)),
        )

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
        return self._ctx.write_matmul_kernel(
            node_index,
            op_node.spec,
            graph.shapes[lhs],
            graph.shapes[rhs],
            graph.strides[lhs],
            graph.strides[rhs],
            graph.dtype,
        )

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
