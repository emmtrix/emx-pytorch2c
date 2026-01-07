from __future__ import annotations

from typing import Dict, List, Sequence, TYPE_CHECKING, Tuple

import torch
import torch.fx

from codegen_backend.dtypes import _CodegenDType, _EMBEDDING_INDEX_DTYPES
from codegen_backend.emitters.embedding import EmbeddingEmitter
from codegen_backend.emitters.embedding_bag import EmbeddingBagEmitter
from codegen_backend.errors import CodegenBackendError
from codegen_backend.graph import _GenericGraph, _OpNode
from codegen_backend.indexing import _contiguous_strides
from codegen_backend.kinds import (
    EmbeddingContext,
    HandlerContextProvider,
    OpKindHandler,
    OpKindHandlerFactory,
    OpNodeBuildResult,
)
from codegen_backend.specs import OpKind, _OpSpec
from codegen_backend.groups.builtin.embedding.parsing import (
    parse_embedding_args,
    parse_embedding_bag_args,
)

if TYPE_CHECKING:
    from codegen_backend.emitters.registry import KindHandlerRegistration


class EmbeddingHandler(OpKindHandler):
    def emit(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        weight_node, indices_node = op_node.inputs
        return self._emit_standard(
            node_index,
            op_node,
            graph,
            inputs=(weight_node, indices_node),
            params={"padding_idx": int(op_node.p("padding_idx", -1))},
        )

    def infer_shapes(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        weight_shape, indices_shape = input_shapes
        if len(weight_shape) != 2:
            raise CodegenBackendError(
                "codegen embedding expects 2D weight tensor"
            )
        return tuple(indices_shape) + (weight_shape[1],)


class EmbeddingBagHandler(OpKindHandler):
    def emit(
        self, node_index: int, op_node: _OpNode, graph: _GenericGraph
    ) -> List[str]:
        weight_node, indices_node, offsets_node = op_node.inputs
        return self._emit_standard(
            node_index,
            op_node,
            graph,
            inputs=(weight_node, indices_node, offsets_node),
            params={
                "mode": int(op_node.p("mode", 0)),
                "padding_idx": int(op_node.p("padding_idx", -1)),
                "include_last_offset": bool(
                    op_node.p("include_last_offset", False)
                ),
            },
        )

    def infer_shapes(
        self,
        op_node: _OpNode,
        input_shapes: Sequence[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        weight_shape, _indices_shape, offsets_shape = input_shapes
        if len(weight_shape) != 2:
            raise CodegenBackendError(
                "codegen _embedding_bag expects 2D weight tensor"
            )
        if len(offsets_shape) != 1:
            raise CodegenBackendError(
                "codegen _embedding_bag expects 1D offsets tensor"
            )
        include_last_offset = bool(op_node.p("include_last_offset", False))
        bag_count = offsets_shape[0] - 1 if include_last_offset else offsets_shape[0]
        return (bag_count, weight_shape[1])


class _BackendEmbeddingHandler(EmbeddingHandler):
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
        weight, indices, padding_idx, scale_grad_by_freq, sparse = (
            parse_embedding_args(node)
        )
        if scale_grad_by_freq or sparse:
            raise CodegenBackendError(
                "codegen embedding supports only scale_grad_by_freq=False and sparse=False"
            )
        if not isinstance(weight, torch.fx.Node) or weight not in shapes:
            raise self._ctx.analysis_service.error_expected_tensor(op_spec.name)
        if not isinstance(indices, torch.fx.Node) or indices not in shapes:
            raise self._ctx.analysis_service.error_expected_tensor(op_spec.name)
        if dtypes[weight] is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                "codegen embedding expects weight to match the graph dtype"
            )
        if dtypes[indices] not in _EMBEDDING_INDEX_DTYPES:
            raise CodegenBackendError(
                "codegen embedding expects indices to have dtype torch.int32 or torch.int64"
            )
        weight_shape = shapes[weight]
        if len(weight_shape) != 2:
            raise CodegenBackendError("codegen embedding expects 2D weight tensor")
        if padding_idx != -1:
            if padding_idx < 0 or padding_idx >= weight_shape[0]:
                raise CodegenBackendError(
                    "codegen embedding expects padding_idx to be -1 or within num_embeddings"
                )
        indices_shape = shapes[indices]
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[weight, indices],
            output_shape=(),
            inplace_input=None,
            params={"padding_idx": padding_idx},
        )
        output_shape = self._ctx.analysis_service.infer_output_shape(
            op_node, [weight_shape, indices_shape]
        )
        op_node.output_shape = output_shape
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        strides[node] = _contiguous_strides(output_shape)
        return OpNodeBuildResult(op_node)


class _BackendEmbeddingBagHandler(EmbeddingBagHandler):
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
            weight,
            indices,
            offsets,
            scale_grad_by_freq,
            mode,
            sparse,
            per_sample_weights,
            include_last_offset,
            padding_idx,
        ) = parse_embedding_bag_args(node)
        if scale_grad_by_freq or sparse:
            raise CodegenBackendError(
                "codegen _embedding_bag supports only scale_grad_by_freq=False and sparse=False"
            )
        if per_sample_weights is not None:
            raise CodegenBackendError(
                "codegen _embedding_bag does not support per_sample_weights"
            )
        if not isinstance(weight, torch.fx.Node) or weight not in shapes:
            raise self._ctx.analysis_service.error_expected_tensor(op_spec.name)
        if not isinstance(indices, torch.fx.Node) or indices not in shapes:
            raise self._ctx.analysis_service.error_expected_tensor(op_spec.name)
        if not isinstance(offsets, torch.fx.Node) or offsets not in shapes:
            raise self._ctx.analysis_service.error_expected_tensor(op_spec.name)
        if dtype_info.torch_dtype is not torch.float32:
            raise CodegenBackendError(
                "codegen _embedding_bag supports only torch.float32 tensors"
            )
        if dtypes[weight] is not dtype_info.torch_dtype:
            raise CodegenBackendError(
                "codegen _embedding_bag expects weight to match the graph dtype"
            )
        if dtypes[indices] not in _EMBEDDING_INDEX_DTYPES:
            raise CodegenBackendError(
                "codegen _embedding_bag expects indices to have dtype torch.int32 or torch.int64"
            )
        if dtypes[offsets] not in _EMBEDDING_INDEX_DTYPES:
            raise CodegenBackendError(
                "codegen _embedding_bag expects offsets to have dtype torch.int32 or torch.int64"
            )
        weight_shape = shapes[weight]
        if len(weight_shape) != 2:
            raise CodegenBackendError("codegen _embedding_bag expects 2D weight tensor")
        if padding_idx != -1:
            if padding_idx < 0 or padding_idx >= weight_shape[0]:
                raise CodegenBackendError(
                    "codegen _embedding_bag expects padding_idx to be -1 or within num_embeddings"
                )
        indices_shape = shapes[indices]
        if len(indices_shape) != 1:
            raise CodegenBackendError("codegen _embedding_bag expects 1D indices tensor")
        offsets_shape = shapes[offsets]
        if len(offsets_shape) != 1:
            raise CodegenBackendError("codegen _embedding_bag expects 1D offsets tensor")
        if include_last_offset and offsets_shape[0] == 0:
            raise CodegenBackendError(
                "codegen _embedding_bag expects non-empty offsets when include_last_offset is True"
            )
        if mode not in (0, 1):
            raise CodegenBackendError(
                "codegen _embedding_bag supports only mode=0 (sum) or mode=1 (mean)"
            )
        bag_count = (
            offsets_shape[0] - 1 if include_last_offset else offsets_shape[0]
        )
        if bag_count < 0:
            raise CodegenBackendError(
                "codegen _embedding_bag expects offsets to contain at least one entry"
            )
        op_node = _OpNode(
            node=node,
            spec=op_spec,
            inputs=[weight, indices, offsets],
            output_shape=(),
            inplace_input=None,
            params={
                "mode": mode,
                "padding_idx": padding_idx,
                "include_last_offset": include_last_offset,
            },
        )
        output_shape = self._ctx.analysis_service.infer_output_shape(
            op_node, [weight_shape, indices_shape, offsets_shape]
        )
        op_node.output_shape = output_shape
        shapes[node] = output_shape
        dtypes[node] = dtype_info.torch_dtype
        strides[node] = _contiguous_strides(output_shape)
        return OpNodeBuildResult(op_node)


def build_handlers(context: EmbeddingContext) -> Dict[OpKind, OpKindHandler]:
    return {
        OpKind.EMBEDDING: _BackendEmbeddingHandler(context, EmbeddingEmitter()),
        OpKind.EMBEDDING_BAG: _BackendEmbeddingBagHandler(
            context, EmbeddingBagEmitter()
        ),
    }


class EmbeddingKindHandlerFactory:
    def build_handlers(
        self, context_provider: HandlerContextProvider
    ) -> Dict[OpKind, OpKindHandler]:
        return build_handlers(context_provider.embedding)


def build_kind_handler_registrations() -> Dict[OpKind, "KindHandlerRegistration"]:
    from codegen_backend.emitters.registry import KindHandlerRegistration

    return {
        OpKind.EMBEDDING: KindHandlerRegistration(
            _BackendEmbeddingHandler, EmbeddingEmitter
        ),
        OpKind.EMBEDDING_BAG: KindHandlerRegistration(
            _BackendEmbeddingBagHandler, EmbeddingBagEmitter
        ),
    }


__all__ = [
    "EmbeddingKindHandlerFactory",
    "build_handlers",
    "build_kind_handler_registrations",
]
