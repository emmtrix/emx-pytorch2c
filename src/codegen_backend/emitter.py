from __future__ import annotations

from math import prod
from typing import Callable, Dict, List, Sequence

import torch
import torch.fx

from codegen_backend.backend_helpers import _kernel_inputs, _resolve_alias
from codegen_backend.c_types import _dtype_to_c_type, _input_c_type
from codegen_backend.emitters.base import _format_array_suffix, _format_dim_args
from codegen_backend.emitters.registry import KindHandlerRegistration
from codegen_backend.errors import CodegenBackendError
from codegen_backend.graph import _GenericGraph
from codegen_backend.kinds import OpKind, OpKindHandler


def _format_c_indentation(source: str, *, indent: str = "    ") -> str:
    formatted_lines: List[str] = []
    indent_level = 0
    for line in source.splitlines():
        stripped = line.lstrip()
        if not stripped:
            formatted_lines.append("")
            continue
        if stripped.startswith("}"):
            indent_level = max(indent_level - 1, 0)
        formatted_lines.append(f"{indent * indent_level}{stripped}")
        open_count = stripped.count("{")
        close_count = stripped.count("}")
        if stripped.startswith("}"):
            close_count = max(close_count - 1, 0)
        indent_level += open_count - close_count
        indent_level = max(indent_level, 0)
    return "\n".join(formatted_lines)


class Emitter:
    def __init__(
        self,
        *,
        templates_env: Callable[[], object],
        kind_handlers: Callable[[], Dict[OpKind, OpKindHandler]],
        kind_handler_registrations: Callable[
            [], Dict[OpKind, KindHandlerRegistration]
        ]
        | None = None,
        temp_allocation_threshold: int = 1024,
    ) -> None:
        self._templates_env = templates_env
        self._kind_handlers = kind_handlers
        self._kind_handler_registrations = (
            kind_handler_registrations
            if kind_handler_registrations is not None
            else lambda: {}
        )
        if temp_allocation_threshold < 0:
            raise ValueError("temp_allocation_threshold must be >= 0")
        self._temp_allocation_threshold = temp_allocation_threshold

    def _temp_pointer_suffix(self, shape: Sequence[int]) -> str:
        if len(shape) <= 1:
            return ""
        return _format_array_suffix(shape[1:])

    def _format_numel_expr(self, shape: Sequence[int]) -> str:
        if not shape:
            return "1"
        return " * ".join(str(dim) for dim in shape)

    def emit(self, graph: _GenericGraph) -> str:
        return self._write_generic_source(graph)

    def _write_generic_source(self, graph: _GenericGraph) -> str:
        placeholders = graph.tensor_placeholders
        op_nodes = graph.op_nodes
        headers = [
            "#include <stdint.h>",
            "#include <sys/types.h>",
            "#include <stdbool.h>",
            f"#include \"{graph.dtype.scalar_header}\"",
        ]
        kernels: List[str] = []
        kind_handlers = self._kind_handlers()
        for index, op_node in enumerate(op_nodes, start=1):
            handler = kind_handlers.get(op_node.spec.kind)
            if handler is None:
                registrations = self._kind_handler_registrations()
                if op_node.spec.kind in registrations:
                    raise CodegenBackendError(
                        "codegen backend did not build handler for kind "
                        f"'{op_node.spec.kind.value}'"
                    )
                raise CodegenBackendError(
                    "codegen backend does not support kind "
                    f"'{op_node.spec.kind.value}'"
                )
            handler.postprocess(op_node, graph)
            kernel_lines = handler.emit(index, op_node, graph)
            kernels.append(_format_c_indentation("\n".join(kernel_lines)))
        signature_parts: List[str] = []
        dim_args = _format_dim_args(graph.variable_dim_order)
        if dim_args:
            signature_parts.append(dim_args)
        input_args = [
            (
                f"const {_input_c_type(graph.dtypes[node], graph.dtype)} "
                f"input_{idx}{_format_array_suffix(graph.shapes[node], graph.variable_dim_names.get(node))}"
            )
            for idx, node in enumerate(placeholders)
        ]
        signature_parts.extend(input_args)
        output_nodes = graph.output_nodes
        if len(output_nodes) == 1:
            output_names = ["out"]
        else:
            output_names = [f"out_{idx}" for idx in range(len(output_nodes))]
        output_args = [
            (
                f"{_dtype_to_c_type(graph.dtypes[node], graph.dtype)} "
                f"{name}{_format_array_suffix(graph.shapes[node], graph.variable_dim_names.get(node))}"
            )
            for name, node in zip(output_names, output_nodes)
        ]
        signature_parts.extend(output_args)
        signature_args = ", ".join(signature_parts)
        signature = (
            f"void ref_codegen_main_{graph.dtype.suffix}("
            f"{signature_args}) {{"
        )
        name_map: Dict[torch.fx.Node, str] = {}
        for idx, placeholder in enumerate(placeholders):
            name_map[placeholder] = f"input_{idx}"
        for output_name, output_node in zip(output_names, output_nodes):
            name_map[output_node] = output_name
        temp_index = 0
        temp_decls: List[str] = []
        temp_allocs: List[str] = []
        temp_frees: List[str] = []
        needs_malloc = False
        for op_node in op_nodes:
            if op_node.node in output_nodes:
                if op_node.inplace_input is not None:
                    if (
                        len(output_nodes) == 1
                        and op_node.node is graph.output_value
                    ):
                        name_map[op_node.node] = name_map[
                            op_node.inputs[op_node.inplace_input]
                        ]
                else:
                    name_map[op_node.node] = name_map[op_node.node]
                continue
            if op_node.inplace_input is not None:
                name_map[op_node.node] = name_map[
                    op_node.inputs[op_node.inplace_input]
                ]
                continue
            temp_name = f"tmp_{temp_index}"
            temp_index += 1
            name_map[op_node.node] = temp_name
            temp_dtype = graph.dtypes[op_node.node]
            temp_c_type = _dtype_to_c_type(temp_dtype, graph.dtype)
            temp_numel = prod(op_node.output_shape) if op_node.output_shape else 1
            temp_bytes = (
                torch.tensor([], dtype=temp_dtype).element_size() * temp_numel
            )
            if self._temp_allocation_threshold > 0 and (
                temp_bytes > self._temp_allocation_threshold
            ):
                needs_malloc = True
                pointer_suffix = self._temp_pointer_suffix(op_node.output_shape)
                numel_expr = self._format_numel_expr(op_node.output_shape)
                if pointer_suffix:
                    temp_allocs.append(
                        f"{temp_c_type} (*{temp_name}){pointer_suffix} = "
                        f"malloc(sizeof({temp_c_type}) * {numel_expr});"
                    )
                else:
                    temp_allocs.append(
                        f"{temp_c_type} *{temp_name} = "
                        f"malloc(sizeof({temp_c_type}) * {numel_expr});"
                    )
                temp_frees.append(f"free({temp_name});")
            else:
                temp_decls.append(
                    f"{temp_c_type} {temp_name}"
                    f"{_format_array_suffix(op_node.output_shape)};"
                )
        call_lines: List[str] = []
        for index, op_node in enumerate(op_nodes, start=1):
            input_names = [
                name_map[_resolve_alias(arg, graph.alias_map)]
                for arg in _kernel_inputs(op_node)
            ]
            output_name = name_map[op_node.node]
            args_parts = []
            if graph.variable_dim_order:
                args_parts.append(", ".join(graph.variable_dim_order))
            args_parts.extend([*input_names, output_name])
            args = ", ".join(args_parts)
            call_lines.append(
                f"node{index}_{op_node.spec.name}_{graph.dtype.suffix}({args});"
            )
        if needs_malloc:
            headers.append("#include <stdlib.h>")
        template = self._templates_env().get_template("generic_source.c.j2")
        return (
            template.render(
                headers=headers,
                kernels=kernels,
                signature=signature,
                temp_decls=temp_decls,
                temp_allocs=temp_allocs,
                temp_frees=temp_frees,
                call_lines=call_lines,
            )
            + "\n"
        )
