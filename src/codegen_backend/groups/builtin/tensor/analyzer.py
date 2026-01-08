from __future__ import annotations

import operator
from typing import Dict, Mapping, Tuple

import torch
import torch.fx

from codegen_backend.analysis_helpers import resolve_scalar_arg
from codegen_backend.errors import CodegenBackendError
from codegen_backend.groups.analysis import GroupAnalysisResult, RegistryGroupAnalyzer
from codegen_backend.indexing import _contiguous_strides
from codegen_backend.registry import _TargetInfo
from codegen_backend.specs import _OpSpec


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
            self._handle_getitem_node(
                node,
                shapes=shapes,
                strides=strides,
                dtypes=dtypes,
                scalar_values=scalar_values,
                alias_map=alias_map,
                empty_outputs=empty_outputs,
            )
            return GroupAnalysisResult(op_node=None)
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
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
        alias_map: Dict[torch.fx.Node, torch.fx.Node],
        empty_outputs: set[torch.fx.Node],
    ) -> None:
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
        }:
            raise CodegenBackendError(
                "codegen backend supports getitem only for _native_batch_norm_legit* ops"
            )
        if index in (0, 0.0):
            alias_map[node] = source
            shapes[node] = shapes[source]
            strides[node] = strides[source]
            dtypes[node] = dtypes[source]
        else:
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


__all__ = ["TensorAnalyzer"]
