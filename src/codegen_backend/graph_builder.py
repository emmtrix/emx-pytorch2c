from __future__ import annotations

import numbers
import operator
from typing import Callable, Dict, List, Sequence, Tuple

import torch
import torch.fx
from torch.fx.immutable_collections import immutable_list

from codegen_backend.backend_helpers import _resolve_alias
from codegen_backend.dtypes import _CODEGEN_DTYPES, _EMBEDDING_INDEX_DTYPES
from codegen_backend.emitters.base import _is_contiguous
from codegen_backend.errors import CodegenBackendError
from codegen_backend.graph import _GenericGraph, _OpNode
from codegen_backend.groups.analysis import GroupAnalyzer
from codegen_backend.groups.registry import GroupRegistry
from codegen_backend.kinds import OpKind, OpKindHandler
from codegen_backend.parser import Parser


class GraphBuilder:
    def __init__(
        self,
        *,
        group_registry: Callable[[], GroupRegistry],
        kind_handlers: Callable[[], Dict[OpKind, OpKindHandler]],
        parser: Parser,
    ) -> None:
        self._group_registry = group_registry
        self._kind_handlers = kind_handlers
        self._parser = parser

    def build(
        self, gm: torch.fx.GraphModule, example_inputs: Sequence[object]
    ) -> _GenericGraph:
        group_registry = self._group_registry()
        return self._analyze_generic_graph(
            gm,
            example_inputs,
            group_analyzers=group_registry.build_group_analyzers(),
            kind_handlers=self._kind_handlers(),
        )

    def _analyze_generic_graph(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: Sequence[object],
        *,
        group_analyzers: Sequence[GroupAnalyzer],
        kind_handlers: Dict[OpKind, OpKindHandler],
    ) -> _GenericGraph:
        dtype_info = self._parser.resolve_dtype_info(gm, example_inputs)
        output_node = None
        placeholders: List[torch.fx.Node] = []
        tensor_placeholders: List[torch.fx.Node] = []
        op_nodes: List[_OpNode] = []
        shapes: Dict[torch.fx.Node, Tuple[int, ...]] = {}
        strides: Dict[torch.fx.Node, Tuple[int, ...]] = {}
        dtypes: Dict[torch.fx.Node, torch.dtype] = {}
        scalar_values: Dict[torch.fx.Node, object] = {}
        alias_map: Dict[torch.fx.Node, torch.fx.Node] = {}
        empty_outputs: set[torch.fx.Node] = set()
        input_iter = iter(example_inputs)

        for node in gm.graph.nodes:
            if node.op == "placeholder":
                try:
                    example = next(input_iter)
                except StopIteration as exc:
                    raise CodegenBackendError(
                        "codegen backend expects example inputs to match placeholder count"
                    ) from exc
                placeholders.append(node)
                if isinstance(example, torch.Tensor):
                    if example.dtype not in _CODEGEN_DTYPES:
                        if example.dtype in _EMBEDDING_INDEX_DTYPES:
                            shapes[node] = tuple(example.shape)
                            strides[node] = tuple(example.stride())
                            dtypes[node] = example.dtype
                            tensor_placeholders.append(node)
                        elif example.numel() == 1:
                            continue
                        continue
                    shapes[node] = tuple(example.shape)
                    strides[node] = tuple(example.stride())
                    dtypes[node] = example.dtype
                    tensor_placeholders.append(node)
                else:
                    if isinstance(example, numbers.Number):
                        scalar_values[node] = example
                    else:
                        try:
                            scalar_values[node] = operator.index(example)
                        except TypeError:
                            raise CodegenBackendError(
                                "codegen backend only supports Tensor or scalar "
                                f"(number/indexable) inputs for placeholder {node}"
                            )
                continue
            if node.op in {"call_function", "call_method"}:
                handled = False
                for analyzer in group_analyzers:
                    if not analyzer.match_node(node):
                        continue
                    result = analyzer.build_op_node(
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
                    if result.op_node is not None:
                        op_nodes.append(result.op_node)
                    if result.dtype_info is not None:
                        dtype_info = result.dtype_info
                    handled = True
                    break
                if not handled:
                    if node.op == "call_method":
                        raise CodegenBackendError(
                            f"Unsupported call_method: {node.target}"
                        )
                    raise CodegenBackendError(
                        f"Unsupported call_function: {node.target}"
                    )
                continue
            if node.op == "output":
                output_node = node
                continue
            raise CodegenBackendError(f"Unsupported node op: {node.op}")

        try:
            next(input_iter)
        except StopIteration:
            pass
        else:
            raise CodegenBackendError(
                "codegen backend expects example inputs to match placeholder count"
            )

        if not op_nodes:
            raise CodegenBackendError("codegen backend requires at least one operation")
        if output_node is None:
            raise CodegenBackendError("codegen backend requires an output node")
        if not tensor_placeholders and dtype_info is None:
            raise CodegenBackendError(
                "codegen backend requires at least one tensor input or a factory op dtype"
            )
        if dtype_info is None:
            raise CodegenBackendError("codegen backend could not infer a graph dtype")
        output_value, output_structure = self._parser.unwrap_output_node(output_node)
        while output_value in alias_map:
            output_value = alias_map[output_value]
        output_nodes = self._collect_output_nodes(output_structure, alias_map)
        if not output_nodes:
            raise CodegenBackendError("codegen backend expects a single output node")
        output_value = output_nodes[0]
        if output_value not in shapes:
            raise CodegenBackendError("codegen backend expects a single output node")
        op_node_set = {op.node for op in op_nodes}
        if output_value not in op_node_set:
            raise CodegenBackendError("codegen backend output must be an operator result")

        output_op = next(op for op in op_nodes if op.node is output_value)
        output_nodes_set = set(output_nodes)
        for op_node in op_nodes:
            if (
                op_node.spec.kind == OpKind.EMPTY_STRIDED
                and op_node.node not in output_nodes_set
                and not _is_contiguous(op_node.output_shape, strides[op_node.node])
            ):
                raise CodegenBackendError(
                    "codegen empty_strided supports non-contiguous strides only for outputs"
                )

        output_inplace_input = None
        if len(output_nodes) == 1:
            for op_node in op_nodes:
                if op_node.node is output_value and op_node.inplace_input is not None:
                    candidate = op_node.inputs[op_node.inplace_input]
                    if candidate in tensor_placeholders:
                        output_inplace_input = candidate
                    break

        return _GenericGraph(
            placeholders=placeholders,
            tensor_placeholders=tensor_placeholders,
            op_nodes=op_nodes,
            output_node=output_node,
            output_value=output_value,
            output_nodes=output_nodes,
            output_op=output_op,
            output_inplace_input=output_inplace_input,
            output_structure=output_structure,
            shapes=shapes,
            strides=strides,
            dtypes=dtypes,
            dtype=dtype_info,
            alias_map=alias_map,
            empty_outputs=empty_outputs,
        )

    @staticmethod
    def _collect_output_nodes(
        output_structure: object,
        alias_map: Dict[torch.fx.Node, torch.fx.Node],
    ) -> List[torch.fx.Node]:
        output_nodes: List[torch.fx.Node] = []

        def visit(value: object) -> None:
            if isinstance(value, torch.fx.Node):
                resolved = _resolve_alias(value, alias_map)
                if resolved not in output_nodes:
                    output_nodes.append(resolved)
                return
            if isinstance(value, (list, tuple, immutable_list)):
                for item in value:
                    visit(item)

        visit(output_structure)
        return output_nodes
