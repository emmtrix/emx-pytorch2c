from __future__ import annotations

import operator
import numbers
from typing import Dict, Mapping, Tuple

import torch
import torch.fx

from codegen_backend.dtypes import _CodegenDType
from codegen_backend.analysis_helpers import resolve_scalar_arg
from codegen_backend.errors import CodegenBackendError
from codegen_backend.groups.analysis import GroupAnalysisResult, RegistryGroupAnalyzer
from codegen_backend.groups.builtin.reductions.args import ReductionsArgParser
from codegen_backend.groups.builtin.tensor.parsing import (
    parse_split_with_sizes_args,
    validate_split_with_sizes,
)
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
_ARGMIN_SPEC = _OpSpec(
    name="argmin",
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
        if node.op == "call_function" and node.target in {
            torch.ops.aten.split_with_sizes,
            torch.ops.aten.split_with_sizes.default,
        }:
            input_arg, split_sizes, dim = parse_split_with_sizes_args(node)
            validate_split_with_sizes(input_arg, split_sizes, dim, shapes)
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
        if source.target in {
            torch.ops.aten.split_with_sizes,
            torch.ops.aten.split_with_sizes.default,
        }:
            input_arg, split_sizes, dim = parse_split_with_sizes_args(source)
            _, input_shape, dim = validate_split_with_sizes(
                input_arg, split_sizes, dim, shapes
            )
            if isinstance(index, float) and index.is_integer():
                index_value = int(index)
            else:
                try:
                    index_value = int(operator.index(index))
                except TypeError as exc:
                    raise CodegenBackendError(
                        "codegen split_with_sizes expects integer getitem indices"
                    ) from exc
            if index_value < 0 or index_value >= len(split_sizes):
                raise CodegenBackendError(
                    "codegen split_with_sizes getitem index is out of range"
                )
            split_offset = sum(split_sizes[:index_value])
            split_size = split_sizes[index_value]
            op_spec = self._supported_ops.get("split_with_sizes")
            if op_spec is None:
                raise CodegenBackendError(
                    "codegen backend does not support split_with_sizes"
                )
            op_node = _OpNode(
                node=node,
                spec=op_spec,
                inputs=[input_arg],
                output_shape=(),
                inplace_input=None,
                params={
                    "dim": dim,
                    "offset": split_offset,
                    "split_size": split_size,
                },
            )
            handler = kind_handlers.get(op_spec.kind)
            if handler is None:
                raise CodegenBackendError(
                    "codegen backend does not support kind 'split_with_sizes'"
                )
            output_shape = handler.infer_shapes(op_node, [input_shape])
            op_node.output_shape = output_shape
            shapes[node] = output_shape
            strides[node] = _contiguous_strides(output_shape)
            dtypes[node] = dtypes[input_arg]
            return op_node
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
            torch.ops.aten.min.dim,
            torch.ops.aten.native_dropout,
            torch.ops.aten.native_dropout.default,
            torch.ops.aten.native_layer_norm,
            torch.ops.aten.native_layer_norm.default,
            torch.ops.aten.native_layer_norm_backward,
            torch.ops.aten.native_layer_norm_backward.default,
            torch.ops.aten.native_group_norm,
            torch.ops.aten.native_group_norm.default,
            torch.ops.aten.native_group_norm_backward,
            torch.ops.aten.native_group_norm_backward.default,
            torch.ops.aten.max_pool3d_with_indices,
            torch.ops.aten.max_pool3d_with_indices.default,
            torch.ops.aten.sort,
            torch.ops.aten.sort.default,
        }:
            raise CodegenBackendError(
                "codegen backend supports getitem only for _native_batch_norm_legit* "
                "ops, _embedding_bag, native_dropout, native_layer_norm, "
                "native_group_norm, max_pool3d_with_indices, sort, max.dim, or min.dim"
            )
        if source.target in {torch.ops.aten.max.dim, torch.ops.aten.min.dim}:
            if index in (0, 0.0):
                alias_map[node] = source
                shapes[node] = shapes[source]
                strides[node] = strides[source]
                dtypes[node] = dtypes[source]
                return None
            if index not in (1, 1.0):
                raise CodegenBackendError(
                    "codegen backend supports max.dim/min.dim getitem only for indices 0 or 1"
                )
            if dtype_info is None:
                raise CodegenBackendError(
                    "codegen backend requires at least one tensor input or a factory op dtype"
                )
            input_arg = source.args[0] if source.args else None
            op_label = (
                "max.dim"
                if source.target is torch.ops.aten.max.dim
                else "min.dim"
            )
            if not isinstance(input_arg, torch.fx.Node) or input_arg not in shapes:
                raise CodegenBackendError(
                    f"codegen {op_label} expects a tensor input"
                )
            if dtypes[input_arg] is not dtype_info.torch_dtype:
                raise CodegenBackendError(
                    f"codegen {op_label} expects inputs to share the graph dtype"
                )
            parser = ReductionsArgParser()
            op_name = (
                "argmax"
                if source.target is torch.ops.aten.max.dim
                else "argmin"
            )
            reduction_dims, keepdim, reduce_all = parser.parse_argminmax_args(
                op_name, source, shapes[input_arg]
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
                    f"codegen {op_label} expects a non-empty reduction dimension"
                )
            op_spec = _ARGMAX_SPEC if op_name == "argmax" else _ARGMIN_SPEC
            op_node = _OpNode(
                node=node,
                spec=op_spec,
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
        if source.target in {
            torch.ops.aten.native_dropout,
            torch.ops.aten.native_dropout.default,
        }:
            if index not in (1, 1.0):
                raise CodegenBackendError(
                    "codegen backend supports native_dropout getitem only for indices 0 or 1"
                )
            shapes[node] = shapes[source]
            strides[node] = _contiguous_strides(shapes[source])
            dtypes[node] = torch.bool
            empty_outputs.add(node)
            return None
        if source.target in {
            torch.ops.aten.max_pool3d_with_indices,
            torch.ops.aten.max_pool3d_with_indices.default,
        }:
            if index not in (1, 1.0):
                raise CodegenBackendError(
                    "codegen backend supports max_pool3d_with_indices getitem only for indices 0 or 1"
                )
            shapes[node] = shapes[source]
            strides[node] = _contiguous_strides(shapes[node])
            dtypes[node] = torch.int64
            empty_outputs.add(node)
            return None
        if source.target in {torch.ops.aten.sort, torch.ops.aten.sort.default}:
            if index not in (1, 1.0):
                raise CodegenBackendError(
                    "codegen backend supports sort getitem only for indices 0 or 1"
                )
            shapes[node] = shapes[source]
            strides[node] = _contiguous_strides(shapes[node])
            dtypes[node] = torch.int64
            empty_outputs.add(node)
            return None
        if source.target in {
            torch.ops.aten.native_layer_norm,
            torch.ops.aten.native_layer_norm.default,
        }:
            if index not in (1, 1.0, 2, 2.0):
                raise CodegenBackendError(
                    "codegen backend supports native_layer_norm getitem only for indices 0, 1, or 2"
                )
            normalized_shape = source.args[1] if len(source.args) > 1 else None
            if isinstance(normalized_shape, torch.fx.Node):
                raise CodegenBackendError(
                    "codegen native_layer_norm expects normalized_shape to be a constant"
                )
            if isinstance(normalized_shape, torch.Size):
                normalized_shape = tuple(normalized_shape)
            if isinstance(normalized_shape, numbers.Integral):
                normalized_shape_tuple = (int(normalized_shape),)
            elif isinstance(normalized_shape, (tuple, list)):
                normalized_shape_tuple = tuple(
                    int(operator.index(item)) for item in normalized_shape
                )
            else:
                raise CodegenBackendError(
                    "codegen native_layer_norm expects normalized_shape to be a tuple of ints"
                )
            input_shape = shapes[source]
            mean_shape = input_shape[: -len(normalized_shape_tuple)] + (
                1,
            ) * len(normalized_shape_tuple)
            shapes[node] = mean_shape
            strides[node] = _contiguous_strides(shapes[node])
            dtypes[node] = dtypes[source]
            empty_outputs.add(node)
            return None
        if source.target in {
            torch.ops.aten.native_group_norm,
            torch.ops.aten.native_group_norm.default,
        }:
            if index not in (1, 1.0, 2, 2.0):
                raise CodegenBackendError(
                    "codegen backend supports native_group_norm getitem only for indices 0, 1, or 2"
                )
            n_value = source.args[3] if len(source.args) > 3 else None
            group_value = source.args[6] if len(source.args) > 6 else None
            if isinstance(n_value, torch.fx.Node) or isinstance(
                group_value, torch.fx.Node
            ):
                raise CodegenBackendError(
                    "codegen native_group_norm expects N and group to be constants"
                )
            n_value = int(operator.index(n_value))
            group_value = int(operator.index(group_value))
            shapes[node] = (n_value, group_value)
            strides[node] = _contiguous_strides(shapes[node])
            dtypes[node] = dtypes[source]
            empty_outputs.add(node)
            return None
        if source.target in {
            torch.ops.aten.native_layer_norm_backward,
            torch.ops.aten.native_layer_norm_backward.default,
        }:
            if index not in (1, 1.0, 2, 2.0):
                raise CodegenBackendError(
                    "codegen backend supports native_layer_norm_backward getitem only for indices 0, 1, or 2"
                )
            normalized_shape = source.args[2] if len(source.args) > 2 else None
            if isinstance(normalized_shape, torch.fx.Node):
                raise CodegenBackendError(
                    "codegen native_layer_norm_backward expects normalized_shape to be a constant"
                )
            if isinstance(normalized_shape, torch.Size):
                normalized_shape = tuple(normalized_shape)
            if isinstance(normalized_shape, numbers.Integral):
                normalized_shape_tuple = (int(normalized_shape),)
            elif isinstance(normalized_shape, (tuple, list)):
                normalized_shape_tuple = tuple(
                    int(operator.index(item)) for item in normalized_shape
                )
            else:
                raise CodegenBackendError(
                    "codegen native_layer_norm_backward expects normalized_shape to be a tuple of ints"
                )
            shapes[node] = normalized_shape_tuple
            strides[node] = _contiguous_strides(shapes[node])
            dtypes[node] = dtypes[source]
            empty_outputs.add(node)
            return None
        if source.target in {
            torch.ops.aten.native_group_norm_backward,
            torch.ops.aten.native_group_norm_backward.default,
        }:
            if index not in (1, 1.0, 2, 2.0):
                raise CodegenBackendError(
                    "codegen backend supports native_group_norm_backward getitem only for indices 0, 1, or 2"
                )
            c_value = source.args[6] if len(source.args) > 6 else None
            if isinstance(c_value, torch.fx.Node):
                raise CodegenBackendError(
                    "codegen native_group_norm_backward expects C to be constant"
                )
            c_value = int(operator.index(c_value))
            shapes[node] = (c_value,)
            strides[node] = _contiguous_strides(shapes[node])
            dtypes[node] = dtypes[source]
            empty_outputs.add(node)
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
