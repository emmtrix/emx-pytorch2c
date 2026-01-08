from __future__ import annotations

from typing import Callable, Dict, List

import torch
import torch.fx

from codegen_backend.backend_helpers import _kernel_inputs, _resolve_alias
from codegen_backend.c_types import _dtype_to_c_type, _input_c_type
from codegen_backend.emitters.base import _format_array_suffix
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
    ) -> None:
        self._templates_env = templates_env
        self._kind_handlers = kind_handlers
        self._kind_handler_registrations = (
            kind_handler_registrations
            if kind_handler_registrations is not None
            else lambda: {}
        )

    def emit(self, graph: _GenericGraph) -> str:
        return self._write_generic_source(graph)

    def _write_generic_source(self, graph: _GenericGraph) -> str:
        placeholders = graph.tensor_placeholders
        op_nodes = graph.op_nodes
        headers = [
            "#include <stdint.h>",
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
        input_args = ", ".join(
            [
                (
                    f"const {_input_c_type(graph.dtypes[node], graph.dtype)} "
                    f"input_{idx}{_format_array_suffix(graph.shapes[node])}"
                )
                for idx, node in enumerate(placeholders)
            ]
        )
        input_args = f"{input_args}, " if input_args else ""
        output_nodes = graph.output_nodes
        if len(output_nodes) == 1:
            output_names = ["out"]
        else:
            output_names = [f"out_{idx}" for idx in range(len(output_nodes))]
        output_args = ", ".join(
            [
                (
                    f"{_dtype_to_c_type(graph.dtypes[node], graph.dtype)} "
                    f"{name}{_format_array_suffix(graph.shapes[node])}"
                )
                for name, node in zip(output_names, output_nodes)
            ]
        )
        signature = (
            f"void ref_codegen_main_{graph.dtype.suffix}("
            f"{input_args}{output_args}) {{"
        )
        name_map: Dict[torch.fx.Node, str] = {}
        for idx, placeholder in enumerate(placeholders):
            name_map[placeholder] = f"input_{idx}"
        for output_name, output_node in zip(output_names, output_nodes):
            name_map[output_node] = output_name
        temp_index = 0
        temp_decls: List[str] = []
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
            temp_decls.append(
                f"{temp_c_type} {temp_name}{_format_array_suffix(op_node.output_shape)};"
            )
        call_lines: List[str] = []
        for index, op_node in enumerate(op_nodes, start=1):
            input_names = [
                name_map[_resolve_alias(arg, graph.alias_map)]
                for arg in _kernel_inputs(op_node)
            ]
            output_name = name_map[op_node.node]
            args = ", ".join([*input_names, output_name])
            call_lines.append(
                f"node{index}_{op_node.spec.name}_{graph.dtype.suffix}({args});"
            )
        template = self._templates_env().get_template("generic_source.c.j2")
        return (
            template.render(
                headers=headers,
                kernels=kernels,
                signature=signature,
                temp_decls=temp_decls,
                call_lines=call_lines,
            )
            + "\n"
        )
