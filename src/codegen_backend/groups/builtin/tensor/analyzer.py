from __future__ import annotations

import operator
from typing import Dict, Mapping, Tuple

import torch
import torch.fx

from codegen_backend.dtypes import _CodegenDType
from codegen_backend.analysis_helpers import resolve_scalar_arg
from codegen_backend.errors import CodegenBackendError
from codegen_backend.groups.analysis import GroupAnalysisResult, RegistryGroupAnalyzer
from codegen_backend.groups.builtin.reductions.args import ReductionsArgParser
from codegen_backend.graph import _OpNode
from codegen_backend.indexing import _contiguous_strides
from codegen_backend.kinds import OpKind, OpKindHandler
from codegen_backend.registry import _TargetInfo
from codegen_backend.specs import _OpSpec


_ARGMAX_SPEC = _OpSpec(
    name="argmax",
    kind=OpKind.ARG_REDUCTION,
    symbol=None,
    supported_targets=set(),
)


class TensorAnalyzer(RegistryGroupAnalyzer):
    def __init__(
        self,
        supported_ops: Mapping[str, _OpSpec],
        target_registry: Mapping[object, _TargetInfo],
    ) -> None:
        super().__init__(
            name="tensor",
            supported_ops=supported_ops,
            target_registry=target_registry,
        )

    def match_node(self, node: torch.fx.Node) -> bool:
        if node.op == "call_method" and node.target == "item":
            return True
        if node.op == "call_function" and node.target is operator.getitem:
            return True
        return super().match_node(node)

    def build_op_node(
        self,
        node: torch.fx.Node,
        *,
        dtype_info,
        shapes,
        strides,
        dtypes,
        scalar_values,
        alias_map,
        empty_outputs,
        kind_handlers,
    ) -> GroupAnalysisResult:
        if node.op == "call_method" and node.target == "item":
            return GroupAnalysisResult(op_node=None)
        if node.op == "call_function" and node.target is operator.getitem:
            op_node = self._handle_getitem_node(
                node,
                dtype_info=dtype_info,
                shapes=shapes,
                strides=strides,
                dtypes=dtypes,
                scalar_values=scalar_values,
                alias_map=alias_map,
                empty_outputs=empty_outputs,
                kind_handlers=kind_handlers,
            )
            return GroupAnalysisResult(op_node=op_node)
        return super().build_op_node(
            node,
            dtype_info=dtype_info,
            shapes=shapes,
            strides=strides,
            dtypes=dtypes,
            scalar_values=scalar_values,
            alias_map=alias_map,
            empty_outputs=empty_outputs,
            kind_handlers=kind_handlers,
        )

    def _handle_getitem_node(
        self,
        node: torch.fx.Node,
        *,
        dtype_info: _CodegenDType | None,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
        alias_map: Dict[torch.fx.Node, torch.fx.Node],
        empty_outputs: set[torch.fx.Node],
        kind_handlers: Dict[OpKind, OpKindHandler],
    ) -> _OpNode | None:
        if node.kwargs:
            raise CodegenBackendError(
                "codegen backend expects getitem to use positional args"
            )
        if len(node.args) != 2:
            raise CodegenBackendError(
                "codegen backend expects getitem to have two inputs"
            )
        source, index = node.args
        if not isinstance(source, torch.fx.Node):
            raise CodegenBackendError(
                "codegen backend expects getitem source to be a tensor op"
            )
        if isinstance(index, torch.fx.Node):
            raise CodegenBackendError(
                "codegen backend supports only constant getitem indices"
            )
        if index not in (0, 0.0, 1, 1.0, 2, 2.0):
            raise CodegenBackendError(
                "codegen backend supports only getitem[0], getitem[1], or getitem[2]"
            )
        if source not in shapes:
            raise CodegenBackendError(
                "codegen backend expects getitem source to be analyzed"
            )
        if source.target not in {
            torch.ops.aten._native_batch_norm_legit,
            torch.ops.aten._native_batch_norm_legit.default,
            torch.ops.aten._native_batch_norm_legit_no_training,
            torch.ops.aten._native_batch_norm_legit_no_training.default,
            torch.ops.aten._embedding_bag,
            torch.ops.aten._embedding_bag.default,
            torch.ops.aten.max.dim,
        }:
            raise CodegenBackendError(
                "codegen backend supports getitem only for _native_batch_norm_legit* ops, _embedding_bag, or max.dim"
            )
        if source.target is torch.ops.aten.max.dim:
            if index in (0, 0.0):
                alias_map[node] = source
                shapes[node] = shapes[source]
                strides[node] = strides[source]
                dtypes[node] = dtypes[source]
                return None
            if index not in (1, 1.0):
                raise CodegenBackendError(
                    "codegen backend supports max.dim getitem only for indices 0 or 1"
                )
            if dtype_info is None:
                raise CodegenBackendError(
                    "codegen backend requires at least one tensor input or a factory op dtype"
                )
            input_arg = source.args[0] if source.args else None
            if not isinstance(input_arg, torch.fx.Node) or input_arg not in shapes:
                raise CodegenBackendError(
                    "codegen max.dim expects a tensor input"
                )
            if dtypes[input_arg] is not dtype_info.torch_dtype:
                raise CodegenBackendError(
                    "codegen max.dim expects inputs to share the graph dtype"
                )
            parser = ReductionsArgParser()
            reduction_dims, keepdim, reduce_all = parser.parse_argminmax_args(
                "argmax", source, shapes[input_arg]
            )
            reduction_count = 1
            if reduce_all:
                for size in shapes[input_arg]:
                    reduction_count *= size
            else:
                for dim in reduction_dims:
                    reduction_count *= shapes[input_arg][dim]
            if reduction_count == 0:
                raise CodegenBackendError(
                    "codegen max.dim expects a non-empty reduction dimension"
                )
            op_node = _OpNode(
                node=node,
                spec=_ARGMAX_SPEC,
                inputs=[input_arg],
                output_shape=(),
                inplace_input=None,
                reduction_dims=reduction_dims,
                keepdim=keepdim,
                params={"reduce_all": reduce_all},
            )
            handler = kind_handlers.get(OpKind.ARG_REDUCTION)
            if handler is None:
                raise CodegenBackendError(
                    "codegen backend does not support kind 'arg_reduction'"
                )
            output_shape = handler.infer_shapes(op_node, [shapes[input_arg]])
            op_node.output_shape = output_shape
            shapes[node] = output_shape
            strides[node] = _contiguous_strides(output_shape)
            dtypes[node] = torch.int64
            return op_node
        if index in (0, 0.0):
            alias_map[node] = source
            shapes[node] = shapes[source]
            strides[node] = strides[source]
            dtypes[node] = dtypes[source]
            return None
        output_shape = (0,)
        if source.target in {
            torch.ops.aten._native_batch_norm_legit,
            torch.ops.aten._native_batch_norm_legit.default,
        }:
            if len(source.args) < 6:
                raise CodegenBackendError(
                    "codegen backend expects batch_norm to include training flag"
                )
            training_arg = source.args[5]
            training_value = resolve_scalar_arg(
                "batch_norm", training_arg, scalar_values
            )
            if bool(training_value):
                if len(shapes[source]) < 2:
                    raise CodegenBackendError(
                        "codegen backend expects batch_norm input rank >= 2"
                    )
                output_shape = (shapes[source][1],)
        shapes[node] = output_shape
        strides[node] = _contiguous_strides(shapes[node])
        dtypes[node] = dtypes[source]
        empty_outputs.add(node)
        return None


__all__ = ["TensorAnalyzer"]
