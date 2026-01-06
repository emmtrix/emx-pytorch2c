from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple

if TYPE_CHECKING:
    from codegen_backend.graph import _GenericGraph, _OpNode
    from codegen_backend.specs import _OpSpec


def _backend_module():
    from codegen_backend import backend as _backend

    return _backend


class KindHandler(ABC):
    @abstractmethod
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        raise NotImplementedError

    def infer_output_shape(
        self,
        op_spec: _OpSpec,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        raise NotImplementedError(
            f"Shape inference not implemented for kind '{op_spec.kind}'."
        )


class ArangeHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        backend = _backend_module()
        return backend._write_arange_kernel(
            node_index,
            op_node,
            op_node.output_shape,
            graph.strides[op_node.node],
            graph.dtype,
        )


class ElementwiseHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        backend = _backend_module()
        kernel_inputs = backend._kernel_inputs(op_node)
        input_shapes = [graph.shapes[arg] for arg in kernel_inputs]
        input_strides = [graph.strides[arg] for arg in kernel_inputs]
        input_dtypes = [graph.dtypes[arg] for arg in kernel_inputs]
        output_strides = graph.strides[op_node.node]
        return backend._write_elementwise_kernel(
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
        op_spec: _OpSpec,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        backend = _backend_module()
        if op_spec.kind == "binary":
            if op_spec.name == "copy":
                output_shape = input_shapes[0]
                broadcast_shape = backend._broadcast_output_shape(
                    op_spec, *input_shapes
                )
                if broadcast_shape != output_shape:
                    raise backend.RefBackendError(
                        "codegen copy expects source to be broadcastable to the destination"
                    )
                return output_shape
            return backend._broadcast_output_shape(op_spec, *input_shapes)
        if op_spec.kind == "where":
            return backend._broadcast_output_shape(op_spec, *input_shapes)
        if op_spec.kind in {"unary", "fill"}:
            return input_shapes[0]
        raise NotImplementedError(
            f"Shape inference not implemented for kind '{op_spec.kind}'."
        )


class FlipHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        backend = _backend_module()
        input_node = op_node.inputs[0]
        return backend._write_flip_kernel(
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
        op_spec: _OpSpec,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        return input_shapes[0]


class PadHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        backend = _backend_module()
        input_node = op_node.inputs[0]
        return backend._write_constant_pad_kernel(
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
        op_spec: _OpSpec,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        backend = _backend_module()
        raise backend.RefBackendError(
            "codegen constant_pad_nd expects constant padding arguments"
        )


class ViewHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        backend = _backend_module()
        input_node = op_node.inputs[0]
        return backend._write_view_kernel(
            node_index,
            op_node,
            graph.shapes[input_node],
            graph.dtypes[input_node],
            op_node.output_shape,
            graph.strides[op_node.node],
            graph.dtype,
        )


class EmptyStridedHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        backend = _backend_module()
        return backend._write_empty_strided_kernel(
            node_index,
            op_node.spec,
            op_node.output_shape,
            graph.dtype,
        )


class DiagonalHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        backend = _backend_module()
        input_node = op_node.inputs[0]
        return backend._write_diagonal_kernel(
            node_index,
            op_node,
            graph.shapes[input_node],
            graph.strides[input_node],
            graph.dtypes[input_node],
            op_node.output_shape,
            graph.strides[op_node.node],
            graph.dtype,
        )


class ReductionHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        backend = _backend_module()
        input_node = op_node.inputs[0]
        if op_node.spec.name == "std":
            return backend._write_std_kernel(
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
            return backend._write_var_kernel(
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
            return backend._write_norm_kernel(
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
        return backend._write_reduction_kernel(
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
        op_spec: _OpSpec,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        return ()


class ArgReductionHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        backend = _backend_module()
        input_node = op_node.inputs[0]
        return backend._write_argminmax_kernel(
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


class SoftmaxHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        backend = _backend_module()
        input_node = op_node.inputs[0]
        return backend._write_softmax_kernel(
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
        op_spec: _OpSpec,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        return input_shapes[0]


class CumsumHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        backend = _backend_module()
        input_node = op_node.inputs[0]
        return backend._write_cumsum_kernel(
            node_index,
            op_node.spec,
            graph.shapes[input_node],
            graph.strides[input_node],
            graph.strides[op_node.node],
            op_node.p("dim"),
            graph.dtypes[input_node],
            graph.dtypes[op_node.node],
            graph.dtype,
        )


class EmbeddingHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        backend = _backend_module()
        weight_node, indices_node = op_node.inputs
        return backend._write_embedding_kernel(
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
        op_spec: _OpSpec,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        backend = _backend_module()
        weight_shape, indices_shape = input_shapes
        if len(weight_shape) != 2:
            raise backend.RefBackendError(
                "codegen embedding expects 2D weight tensor"
            )
        return tuple(indices_shape) + (weight_shape[1],)


class GatherHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        backend = _backend_module()
        input_node, index_node = op_node.inputs
        return backend._write_gather_kernel(
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


class ConcatHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        backend = _backend_module()
        input_shapes = [graph.shapes[arg] for arg in op_node.inputs]
        input_strides = [graph.strides[arg] for arg in op_node.inputs]
        return backend._write_concat_kernel(
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
        op_spec: _OpSpec,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        backend = _backend_module()
        raise backend.RefBackendError("codegen cat expects a tensor list input")


class Pool2dHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        backend = _backend_module()
        input_node = op_node.inputs[0]
        return backend._write_pool2d_kernel(
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


class Pool1dHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        backend = _backend_module()
        input_node = op_node.inputs[0]
        return backend._write_pool1d_kernel(
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


class Col2imHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        backend = _backend_module()
        input_node = op_node.inputs[0]
        return backend._write_col2im_kernel(
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


class BatchNormHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        backend = _backend_module()
        input_node = op_node.inputs[0]
        return backend._write_batch_norm_kernel(
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
        op_spec: _OpSpec,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        return input_shapes[0]


class PdistHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        backend = _backend_module()
        input_node = op_node.inputs[0]
        return backend._write_pdist_kernel(
            node_index,
            op_node.spec,
            graph.shapes[input_node],
            op_node.output_shape,
            graph.dtype,
        )

    def infer_output_shape(
        self,
        op_spec: _OpSpec,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        backend = _backend_module()
        if len(input_shapes[0]) != 2:
            raise backend.RefBackendError(
                "codegen pdist expects a 2D input tensor"
            )
        n = input_shapes[0][0]
        return (n * (n - 1) // 2,)


class Conv1dHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        backend = _backend_module()
        input_node, weight_node, *_ = op_node.inputs
        return backend._write_conv1d_kernel(
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
        op_spec: _OpSpec,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        backend = _backend_module()
        raise backend.RefBackendError("codegen conv1d expects convolution arguments")


class Conv2dHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        backend = _backend_module()
        input_node, weight_node, *_ = op_node.inputs
        return backend._write_conv2d_kernel(
            node_index,
            op_node.spec,
            graph.shapes[input_node],
            graph.shapes[weight_node],
            op_node.output_shape,
            op_node.p("stride", (1, 1)),
            op_node.p("padding", (0, 0)),
            op_node.p("dilation", (1, 1)),
            op_node.p("groups", 1),
            graph.dtype,
            bool(op_node.p("has_bias", False)),
        )

    def infer_output_shape(
        self,
        op_spec: _OpSpec,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        backend = _backend_module()
        raise backend.RefBackendError("codegen conv2d expects convolution arguments")


class AddmmHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        backend = _backend_module()
        input_node, mat1_node, mat2_node = op_node.inputs
        return backend._write_addmm_kernel(
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
        op_spec: _OpSpec,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        backend = _backend_module()
        input_shape, mat1_shape, mat2_shape = input_shapes
        if len(input_shape) > 2 or len(mat1_shape) != 2 or len(mat2_shape) != 2:
            raise backend.RefBackendError("codegen addmm expects 2D inputs")
        if mat1_shape[1] != mat2_shape[0]:
            raise backend.RefBackendError(
                "codegen addmm requires inner dimensions to match"
            )
        expected_shape = (mat1_shape[0], mat2_shape[1])
        if (
            backend._broadcast_output_shape(op_spec, input_shape, expected_shape)
            != expected_shape
        ):
            raise backend.RefBackendError(
                "codegen addmm expects input shape to be broadcastable to matmul output"
            )
        return expected_shape


class AddbmmHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        backend = _backend_module()
        input_node, batch1_node, batch2_node = op_node.inputs
        return backend._write_addbmm_kernel(
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
        op_spec: _OpSpec,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        backend = _backend_module()
        input_shape, batch1_shape, batch2_shape = input_shapes
        if (
            len(input_shape) > 2
            or len(batch1_shape) != 3
            or len(batch2_shape) != 3
        ):
            raise backend.RefBackendError(
                "codegen addbmm expects 0-2D input and 3D batches"
            )
        if batch1_shape[0] != batch2_shape[0]:
            raise backend.RefBackendError(
                "codegen addbmm requires batch dimensions to match"
            )
        if batch1_shape[2] != batch2_shape[1]:
            raise backend.RefBackendError(
                "codegen addbmm requires inner dimensions to match"
            )
        expected_shape = (batch1_shape[1], batch2_shape[2])
        if (
            backend._broadcast_output_shape(op_spec, input_shape, expected_shape)
            != expected_shape
        ):
            raise backend.RefBackendError(
                "codegen addbmm expects input shape to be broadcastable to bmm output"
            )
        return expected_shape


class AddmvHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        backend = _backend_module()
        input_node, mat_node, vec_node = op_node.inputs
        return backend._write_addmv_kernel(
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
        op_spec: _OpSpec,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        backend = _backend_module()
        input_shape, mat_shape, vec_shape = input_shapes
        if len(input_shape) not in (0, 1) or len(mat_shape) != 2 or len(vec_shape) != 1:
            raise backend.RefBackendError(
                "codegen addmv expects a scalar or 1D input and 2D matrix/1D vector"
            )
        if mat_shape[1] != vec_shape[0]:
            raise backend.RefBackendError(
                "codegen addmv requires inner dimensions to match"
            )
        expected_shape = (mat_shape[0],)
        if (
            backend._broadcast_output_shape(op_spec, input_shape, expected_shape)
            != expected_shape
        ):
            raise backend.RefBackendError(
                "codegen addmv expects input shape to be broadcastable to mat-vec output"
            )
        return expected_shape


class AddrHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        backend = _backend_module()
        input_node, vec1_node, vec2_node = op_node.inputs
        return backend._write_addr_kernel(
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
        op_spec: _OpSpec,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        backend = _backend_module()
        input_shape, vec1_shape, vec2_shape = input_shapes
        if len(vec1_shape) != 1 or len(vec2_shape) != 1:
            raise backend.RefBackendError(
                "codegen addr expects 1D vectors"
            )
        if len(input_shape) > 2:
            raise backend.RefBackendError(
                "codegen addr expects input with rank <= 2"
            )
        expected_shape = (vec1_shape[0], vec2_shape[0])
        if (
            backend._broadcast_output_shape(op_spec, input_shape, expected_shape)
            != expected_shape
        ):
            raise backend.RefBackendError(
                "codegen addr expects input shape to be broadcastable to outer product output"
            )
        return expected_shape


class MatmulHandler(KindHandler):
    def emit_kernel(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        backend = _backend_module()
        lhs, rhs = op_node.inputs
        return backend._write_matmul_kernel(
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
        op_spec: _OpSpec,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        backend = _backend_module()
        a_shape, b_shape = input_shapes
        if op_spec.name == "matmul":
            if len(a_shape) == 1 and len(b_shape) == 1:
                if a_shape[0] != b_shape[0]:
                    raise backend.RefBackendError(
                        "codegen matmul requires inner dimensions to match"
                    )
                return ()
            if len(a_shape) != 2 or len(b_shape) != 2:
                raise backend.RefBackendError(
                    "codegen matmul requires 1D or 2D inputs"
                )
            if a_shape[1] != b_shape[0]:
                raise backend.RefBackendError(
                    "codegen matmul requires inner dimensions to match"
                )
            return (a_shape[0], b_shape[1])
        if len(a_shape) != 3 or len(b_shape) != 3:
            raise backend.RefBackendError("codegen bmm requires 3D inputs")
        if a_shape[0] != b_shape[0]:
            raise backend.RefBackendError(
                "codegen bmm requires batch dimensions to match"
            )
        if a_shape[2] != b_shape[1]:
            raise backend.RefBackendError(
                "codegen bmm requires inner dimensions to match"
            )
        return (a_shape[0], a_shape[1], b_shape[2])


def build_kind_handlers() -> Dict[str, KindHandler]:
    elementwise = ElementwiseHandler()
    return {
        "arange": ArangeHandler(),
        "binary": elementwise,
        "unary": elementwise,
        "where": elementwise,
        "fill": elementwise,
        "flip": FlipHandler(),
        "pad": PadHandler(),
        "view": ViewHandler(),
        "empty_strided": EmptyStridedHandler(),
        "diagonal": DiagonalHandler(),
        "reduction": ReductionHandler(),
        "arg_reduction": ArgReductionHandler(),
        "softmax": SoftmaxHandler(),
        "cumsum": CumsumHandler(),
        "embedding": EmbeddingHandler(),
        "gather": GatherHandler(),
        "concat": ConcatHandler(),
        "pool2d": Pool2dHandler(),
        "pool1d": Pool1dHandler(),
        "col2im": Col2imHandler(),
        "batch_norm": BatchNormHandler(),
        "pdist": PdistHandler(),
        "conv1d": Conv1dHandler(),
        "conv2d": Conv2dHandler(),
        "addmm": AddmmHandler(),
        "addbmm": AddbmmHandler(),
        "addmv": AddmvHandler(),
        "addr": AddrHandler(),
        "matmul": MatmulHandler(),
    }
