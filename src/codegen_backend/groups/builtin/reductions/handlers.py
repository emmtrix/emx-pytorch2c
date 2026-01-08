from __future__ import annotations

from typing import Dict, List, Sequence, TYPE_CHECKING, Tuple

import torch
import torch.fx

from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.argreduction import ArgReductionEmitter
from codegen_backend.emitters.reduction import ReductionEmitter
from codegen_backend.emitters.softmax import SoftmaxEmitter
from codegen_backend.errors import CodegenBackendError
from codegen_backend.graph import _GenericGraph, _OpNode
from codegen_backend.indexing import _contiguous_strides
from codegen_backend.kinds import (
    HandlerContextProvider,
    OpKindHandler,
    OpKindHandlerFactory,
    OpNodeBuildResult,
    ReductionContext,
)
from codegen_backend.specs import OpKind, _OpSpec

if TYPE_CHECKING:
    from codegen_backend.emitters.registry import KindHandlerRegistration


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


class ReductionHandler(OpKindHandler):
    def emit(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        req = self._make_standard_request(node_index, op_node, graph)
        req.reduction_dims = op_node.reduction_dims or ()
        req.keepdim = op_node.keepdim
        if op_node.spec.name in {"std", "var"}:
            req.params["unbiased"] = bool(op_node.p("unbiased", True))
        if op_node.spec.name == "norm":
            req.params["p_value"] = float(op_node.p("norm_p", 2.0))
        return self._emit_request(req)

    def infer_shapes(
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


class ArgReductionHandler(OpKindHandler):
    def emit(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        req = self._make_standard_request(node_index, op_node, graph)
        req.reduction_dims = op_node.reduction_dims or ()
        req.keepdim = op_node.keepdim
        req.params["reduce_all"] = bool(op_node.p("reduce_all", False))
        return self._emit_request(req)

    def infer_shapes(
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


class SoftmaxHandler(OpKindHandler):
    def emit(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        return self._emit_standard(
            node_index,
            op_node,
            graph,
            params={"dim": op_node.p("dim")},
        )

    def infer_shapes(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        return input_shapes[0]


class _BackendReductionHandler(ReductionHandler):
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
            raise CodegenBackendError(
                "codegen backend requires at least one tensor input or a factory op dtype"
            )
        if len(node.args) < 1:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects one input"
            )
        args_to_check = node.args[:1]
        input_nodes: List[torch.fx.Node] = []
        input_shapes: List[tuple[int, ...]] = []
        for arg in args_to_check:
            if not isinstance(arg, torch.fx.Node):
                raise self._ctx.analysis_service.error_expected_tensor(op_spec.name)
            if arg not in shapes:
                raise self._ctx.analysis_service.error_expected_tensor(op_spec.name)
            input_nodes.append(arg)
            input_shapes.append(shapes[arg])
        input_dtypes = [dtypes[arg] for arg in input_nodes]
        if any(dtype is not dtype_info.torch_dtype for dtype in input_dtypes):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects inputs to share the graph dtype"
            )
        param_values: Dict[str, object] = {}
        parser = self._ctx.arg_parser
        if op_spec.name == "norm":
            if dtype_info.torch_dtype not in (torch.float32, torch.float64):
                raise CodegenBackendError(
                    "codegen norm supports only torch.float32 or torch.float64 tensors"
                )
            reduction_dims, keepdim, reduce_all, norm_p = parser.parse_norm_args(
                op_spec.name, node, input_shapes[0]
            )
            param_values["norm_p"] = norm_p
        else:
            reduction_dims, keepdim, reduce_all, unbiased = parser.parse_reduction_args(
                op_spec.name, node, input_shapes[0]
            )
            if unbiased is not None:
                param_values["unbiased"] = unbiased
            if op_spec.name == "var" and dtype_info.torch_dtype not in (
                torch.float32,
                torch.float64,
            ):
                raise CodegenBackendError(
                    "codegen var supports only torch.float32 or torch.float64 tensors"
                )
        param_values["reduce_all"] = reduce_all
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=list(input_nodes),
            output_shape=(),
            inplace_input=inplace_input,
            reduction_dims=reduction_dims,
            keepdim=keepdim,
            params=param_values,
        )
        self.validate(op_node, input_shapes, input_dtypes, dtype_info)
        output_shape = self._ctx.analysis_service.infer_output_shape(
            op_node, input_shapes
        )
        op_node.output_shape = output_shape
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        if inplace_input is not None:
            strides[node] = strides[input_nodes[inplace_input]]
        else:
            strides[node] = _contiguous_strides(output_shape)
        return OpNodeBuildResult(op_node)


class _BackendArgReductionHandler(ArgReductionHandler):
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
            raise CodegenBackendError(
                "codegen backend requires at least one tensor input or a factory op dtype"
            )
        if len(node.args) < 1:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects one input"
            )
        args_to_check = node.args[:1]
        input_nodes: List[torch.fx.Node] = []
        input_shapes: List[tuple[int, ...]] = []
        for arg in args_to_check:
            if not isinstance(arg, torch.fx.Node):
                raise self._ctx.analysis_service.error_expected_tensor(op_spec.name)
            if arg not in shapes:
                raise self._ctx.analysis_service.error_expected_tensor(op_spec.name)
            input_nodes.append(arg)
            input_shapes.append(shapes[arg])
        input_dtypes = [dtypes[arg] for arg in input_nodes]
        if any(dtype is not dtype_info.torch_dtype for dtype in input_dtypes):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects inputs to share the graph dtype"
            )
        reduction_dims, keepdim, reduce_all = self._ctx.arg_parser.parse_argminmax_args(
            op_spec.name, node, input_shapes[0]
        )
        reduction_count = 1
        if reduce_all:
            for size in input_shapes[0]:
                reduction_count *= size
        else:
            for dim in reduction_dims:
                reduction_count *= input_shapes[0][dim]
        if reduction_count == 0:
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects a non-empty reduction dimension"
            )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=list(input_nodes),
            output_shape=(),
            inplace_input=inplace_input,
            reduction_dims=reduction_dims,
            keepdim=keepdim,
            params={"reduce_all": reduce_all},
        )
        self.validate(op_node, input_shapes, input_dtypes, dtype_info)
        output_shape = self._ctx.analysis_service.infer_output_shape(
            op_node, input_shapes
        )
        op_node.output_shape = output_shape
        shapes[node] = output_shape
        dtypes[node] = torch.int64
        if inplace_input is not None:
            strides[node] = strides[input_nodes[inplace_input]]
        else:
            strides[node] = _contiguous_strides(output_shape)
        return OpNodeBuildResult(op_node)


class _BackendSoftmaxHandler(SoftmaxHandler):
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
        if not node.args:
            raise CodegenBackendError(f"codegen {op_spec.name} expects one input")
        input_arg = node.args[0]
        if not isinstance(input_arg, torch.fx.Node) or input_arg not in shapes:
            raise self._ctx.analysis_service.error_expected_tensor(op_spec.name)
        if dtype_info.torch_dtype not in (torch.float32, torch.float64):
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only torch.float32 or torch.float64 tensors"
            )
        if dtypes[input_arg] not in (torch.float32, torch.float64):
            raise CodegenBackendError(
                f"codegen {op_spec.name} supports only torch.float32 or torch.float64 tensors"
            )
        dim, dtype = self._ctx.arg_parser.parse_softmax_args(
            op_spec.name, node, shapes[input_arg]
        )
        if dtype is not None and dtype not in (torch.float32, torch.float64):
            raise CodegenBackendError(
                f"codegen {op_spec.name} expects dtype to be torch.float32, torch.float64, or None"
            )
        output_shape = shapes[input_arg]
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        strides[node] = _contiguous_strides(output_shape)
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[input_arg],
            output_shape=output_shape,
            inplace_input=None,
            params={"dim": dim},
        )
        return OpNodeBuildResult(op_node)


def build_handlers(context: ReductionContext) -> Dict[OpKind, OpKindHandler]:
    return {
        OpKind.REDUCTION: _BackendReductionHandler(context, ReductionEmitter()),
        OpKind.ARG_REDUCTION: _BackendArgReductionHandler(
            context, ArgReductionEmitter()
        ),
        OpKind.SOFTMAX: _BackendSoftmaxHandler(context, SoftmaxEmitter()),
    }


class ReductionsKindHandlerFactory:
    def build_handlers(
        self, context_provider: HandlerContextProvider
    ) -> Dict[OpKind, OpKindHandler]:
        return build_handlers(context_provider.reductions)


def build_kind_handler_registrations() -> Dict[OpKind, "KindHandlerRegistration"]:
    from codegen_backend.emitters.registry import KindHandlerRegistration

    return {
        OpKind.REDUCTION: KindHandlerRegistration(
            _BackendReductionHandler, ReductionEmitter
        ),
        OpKind.ARG_REDUCTION: KindHandlerRegistration(
            _BackendArgReductionHandler, ArgReductionEmitter
        ),
        OpKind.SOFTMAX: KindHandlerRegistration(
            _BackendSoftmaxHandler, SoftmaxEmitter
        ),
    }


__all__ = [
    "ReductionsKindHandlerFactory",
    "build_handlers",
    "build_kind_handler_registrations",
]
