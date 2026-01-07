from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Mapping,
    Protocol,
    Sequence,
    Tuple,
)

from codegen_backend.errors import CodegenBackendError
from codegen_backend.specs import OpKind

if TYPE_CHECKING:
    import torch
    from codegen_backend.dtypes import _CodegenDType
    from codegen_backend.emitters.base import KindEmitter
    from codegen_backend.emitters.registry import KindHandlerRegistration
    from codegen_backend.graph import _GenericGraph, _OpNode
    from codegen_backend.groups.builtin.reductions.args import ReductionsArgParser
    from codegen_backend.specs import _OpSpec
    from codegen_backend.services import GraphAnalysisService

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
    """Base context shared by handler implementations."""

    @property
    def analysis_service(self) -> "GraphAnalysisService": ...

    def kernel_inputs(self, op_node: "_OpNode") -> List["torch.fx.Node"]: ...


class LegacyHandlerContext(HandlerContext, Protocol):
    """Deprecated monolithic handler context (use per-group interfaces)."""

    @property
    def kind_handlers(self) -> Dict[OpKind, "OpKindHandler"]: ...


class KernelInputsContext(Protocol):
    def kernel_inputs(self, op_node: "_OpNode") -> List["torch.fx.Node"]: ...


class AnalysisContext(Protocol):
    @property
    def analysis_service(self) -> "GraphAnalysisService": ...


class ElementwiseContext(KernelInputsContext, AnalysisContext, Protocol):
    pass


class ReductionContext(KernelInputsContext, AnalysisContext, Protocol):
    @property
    def arg_parser(self) -> "ReductionsArgParser": ...


class PoolingContext(KernelInputsContext, AnalysisContext, Protocol):
    pass


class ConvContext(KernelInputsContext, AnalysisContext, Protocol):
    pass


class EmbeddingContext(KernelInputsContext, AnalysisContext, Protocol):
    pass


class TensorContext(KernelInputsContext, AnalysisContext, Protocol):
    pass


class HandlerContextProvider(Protocol):
    @property
    def elementwise(self) -> ElementwiseContext: ...

    @property
    def reductions(self) -> ReductionContext: ...

    @property
    def pooling(self) -> PoolingContext: ...

    @property
    def conv(self) -> ConvContext: ...

    @property
    def embedding(self) -> EmbeddingContext: ...

    @property
    def tensor(self) -> TensorContext: ...


class ContextProviderFactory(Protocol):
    def build_context_provider(
        self,
        base_provider: HandlerContextProvider,
        backend: object,
    ) -> HandlerContextProvider: ...


class OpKindHandlerFactory(Protocol):
    def build_handlers(
        self, context_provider: HandlerContextProvider
    ) -> Dict[OpKind, "OpKindHandler"]: ...


@dataclass
class OpNodeBuildResult:
    op_node: "_OpNode"
    dtype_info: "_CodegenDType | None" = None


class OpKindHandler(ABC):
    def __init__(
        self,
        context: HandlerContext,
        emitter: "KindEmitter | None" = None,
        builder: Callable[
            [
                "torch.fx.Node",
                "_OpSpec",
                "_CodegenDType | None",
                Dict["torch.fx.Node", Tuple[int, ...]],
                Dict["torch.fx.Node", Tuple[int, ...]],
                Dict["torch.fx.Node", "torch.dtype"],
                Dict["torch.fx.Node", object],
                int | None,
            ],
            OpNodeBuildResult | None,
        ]
        | None = None,
    ) -> None:
        self._ctx = context
        self._emitter = emitter
        self._builder = builder

    def _make_standard_request(
        self,
        node_index: int,
        op_node: _OpNode,
        graph: _GenericGraph,
        *,
        inputs: Sequence["torch.fx.Node"] | None = None,
        params: Dict[str, object] | None = None,
    ) -> KernelEmitRequest:
        resolved_inputs = (
            list(inputs) if inputs is not None else self._ctx.kernel_inputs(op_node)
        )
        req = _make_request(node_index, op_node, graph, resolved_inputs)
        if params:
            req.params.update(params)
        return req

    def _emit_standard(
        self,
        node_index: int,
        op_node: _OpNode,
        graph: _GenericGraph,
        *,
        inputs: Sequence["torch.fx.Node"] | None = None,
        params: Dict[str, object] | None = None,
    ) -> List[str]:
        if self._emitter is None:
            raise CodegenBackendError("codegen handler requires an emitter")
        req = self._make_standard_request(
            node_index, op_node, graph, inputs=inputs, params=params
        )
        return self._emitter.emit(req)

    def _emit_request(self, req: KernelEmitRequest) -> List[str]:
        if self._emitter is None:
            raise CodegenBackendError("codegen handler requires an emitter")
        return self._emitter.emit(req)

    def build_op_node(
        self,
        node: "torch.fx.Node",
        op_spec: "_OpSpec",
        dtype_info: "_CodegenDType | None",
        shapes: Dict["torch.fx.Node", Tuple[int, ...]],
        strides: Dict["torch.fx.Node", Tuple[int, ...]],
        dtypes: Dict["torch.fx.Node", "torch.dtype"],
        scalar_values: Dict["torch.fx.Node", object],
        inplace_input: int | None,
    ) -> OpNodeBuildResult | None:
        if self._builder is None:
            return None
        return self._builder(
            node,
            op_spec,
            dtype_info,
            shapes,
            strides,
            dtypes,
            scalar_values,
            inplace_input,
        )

    def infer_graph_dtype(
        self, node: "torch.fx.Node", op_spec: "_OpSpec"
    ) -> "torch.dtype | None":
        return None

    def validate(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
        input_dtypes: Sequence["torch.dtype"],
        dtype_info: "_CodegenDType",
    ) -> None:
        return None

    @abstractmethod
    def emit(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def infer_shapes(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        raise NotImplementedError

    def infer_output_shape(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        return self.infer_shapes(op_node, input_shapes)

    def postprocess(self, op_node: _OpNode, graph: _GenericGraph) -> None:
        return None


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
        params=dict(op_node.params),
    )
