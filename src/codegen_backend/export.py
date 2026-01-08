import re
import struct
from math import prod
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.fx

from codegen_backend.errors import CodegenBackendError
from codegen_backend.backend import CodegenBackend
from codegen_backend.backend_helpers import _resolve_alias
from codegen_backend.c_types import _dtype_to_c_type, _input_c_type
from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.base import _format_array_suffix, _format_dim_args
from codegen_backend.graph import _GenericGraph


def _sanitize_weight_name(target: str) -> str:
    sanitized = re.sub(r"[^0-9a-zA-Z_]+", "_", target).strip("_")
    if not sanitized or sanitized[0].isdigit():
        sanitized = f"w_{sanitized}"
    return sanitized


def _resolve_attr(obj: object, target: str) -> object:
    current = obj
    for name in target.split("."):
        current = getattr(current, name)
    return current


def _lift_get_attr_to_placeholders(
    gm: torch.fx.GraphModule, example_inputs: Sequence[object]
) -> Tuple[
    torch.fx.GraphModule, List[object], Dict[torch.fx.Node, str], List[torch.Tensor]
]:
    graph = torch.fx.Graph()
    env: Dict[torch.fx.Node, torch.fx.Node] = {}
    input_iter = iter(example_inputs)
    updated_inputs: List[object] = []
    weight_placeholders: Dict[torch.fx.Node, str] = {}
    weight_tensors: List[torch.Tensor] = []
    used_weight_names: Dict[str, int] = {}

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            try:
                example = next(input_iter)
            except StopIteration as exc:
                raise CodegenBackendError(
                    "codegen backend expects example inputs to match placeholder count"
                ) from exc
            placeholder = graph.placeholder(node.target)
            env[node] = placeholder
            updated_inputs.append(example)
            continue
        if node.op == "get_attr":
            weight = _resolve_attr(gm, node.target)
            if not isinstance(weight, torch.Tensor):
                raise CodegenBackendError(
                    "codegen backend export expects get_attr tensors only"
                )
            base_name = f"weight_{_sanitize_weight_name(node.target)}"
            count = used_weight_names.get(base_name, 0)
            used_weight_names[base_name] = count + 1
            weight_name = base_name if count == 0 else f"{base_name}_{count}"
            placeholder = graph.placeholder(weight_name)
            env[node] = placeholder
            updated_inputs.append(weight)
            weight_placeholders[placeholder] = weight_name
            weight_tensors.append(weight)
            continue
        env[node] = graph.node_copy(node, lambda n: env[n])

    graph.lint()
    return (
        torch.fx.GraphModule(gm, graph),
        updated_inputs,
        weight_placeholders,
        weight_tensors,
    )


def _format_float32_hex(value: float) -> str:
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    sign = "-" if (bits >> 31) else ""
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    if exponent == 0 and mantissa == 0:
        return f"{sign}0x0.0p+0"
    if exponent == 0xFF:
        if mantissa == 0:
            return f"{sign}INFINITY"
        return "NAN"
    if exponent == 0:
        shift = mantissa.bit_length() - 1
        exponent_val = shift - 149
        fraction = (mantissa - (1 << shift)) << (23 - shift)
    else:
        exponent_val = exponent - 127
        fraction = mantissa
    return f"{sign}0x1.{fraction:06x}p{exponent_val:+d}"


def _format_float64_hex(value: float) -> str:
    bits = struct.unpack("<Q", struct.pack("<d", float(value)))[0]
    sign = "-" if (bits >> 63) else ""
    exponent = (bits >> 52) & 0x7FF
    mantissa = bits & 0xFFFFFFFFFFFFF
    if exponent == 0 and mantissa == 0:
        return f"{sign}0x0.0p+0"
    if exponent == 0x7FF:
        if mantissa == 0:
            return f"{sign}INFINITY"
        return "NAN"
    if exponent == 0:
        shift = mantissa.bit_length() - 1
        exponent_val = shift - 1074
        fraction = (mantissa - (1 << shift)) << (52 - shift)
    else:
        exponent_val = exponent - 1023
        fraction = mantissa
    return f"{sign}0x1.{fraction:013x}p{exponent_val:+d}"


def _format_weight_value(value: object, dtype: torch.dtype) -> str:
    if dtype is torch.bool:
        return "1" if bool(value) else "0"
    if dtype is torch.float32:
        return f"{_format_float32_hex(float(value))}f"
    if dtype is torch.float64:
        return _format_float64_hex(float(value))
    return str(int(value))


def _emit_initializer_lines(
    values: List[str], shape: List[int], indent: str = "    ", per_line: int = 8
) -> List[str]:
    if len(shape) == 1:
        lines: List[str] = []
        for index in range(0, len(values), per_line):
            chunk = ", ".join(values[index : index + per_line])
            lines.append(f"{indent}{chunk},")
        if lines:
            lines[-1] = lines[-1].rstrip(",")
        return lines
    sub_shape = shape[1:]
    sub_size = prod(sub_shape)
    lines = []
    for index in range(shape[0]):
        start = index * sub_size
        end = start + sub_size
        lines.append(f"{indent}{{")
        lines.extend(
            _emit_initializer_lines(values[start:end], sub_shape, indent + "    ", per_line)
        )
        lines.append(f"{indent}}},")
    if lines:
        lines[-1] = lines[-1].rstrip(",")
    return lines


def _emit_initializer_lines_truncated(
    values: List[str],
    shape: List[int],
    truncate_after: int,
    indent: str = "    ",
    per_line: int = 8,
    start_index: int = 0,
    emitted: int = 0,
) -> tuple[List[str], int, int, bool]:
    if len(shape) == 1:
        items: List[str] = []
        truncated = False
        index = start_index
        for _ in range(shape[0]):
            if emitted >= truncate_after:
                items.append("...")
                truncated = True
                break
            items.append(values[index])
            index += 1
            emitted += 1
        lines: List[str] = []
        for item_index in range(0, len(items), per_line):
            chunk = ", ".join(items[item_index : item_index + per_line])
            lines.append(f"{indent}{chunk},")
        if lines:
            lines[-1] = lines[-1].rstrip(",")
        return lines, index, emitted, truncated
    sub_shape = shape[1:]
    sub_size = prod(sub_shape)
    lines: List[str] = []
    index = start_index
    truncated = False
    for _ in range(shape[0]):
        lines.append(f"{indent}{{")
        sub_lines, index, emitted, sub_truncated = _emit_initializer_lines_truncated(
            values,
            sub_shape,
            truncate_after,
            indent + "    ",
            per_line,
            index,
            emitted,
        )
        lines.extend(sub_lines)
        lines.append(f"{indent}}},")
        if sub_truncated:
            truncated = True
            break
    if lines:
        lines[-1] = lines[-1].rstrip(",")
    return lines, index, emitted, truncated


def _emit_inline_weights(
    weights: Dict[str, torch.Tensor],
    graph_dtype: _CodegenDType,
    truncate_weights_after: int | None = None,
) -> List[str]:
    lines: List[str] = []
    for name, tensor in weights.items():
        flat = tensor.detach().cpu().contiguous().view(-1)
        c_type = _dtype_to_c_type(flat.dtype, graph_dtype)
        shape = list(tensor.shape) if tensor.dim() > 0 else [1]
        values = [_format_weight_value(value, flat.dtype) for value in flat.tolist()]
        dims = "".join(f"[{dim}]" for dim in shape)
        lines.append(f"static const {c_type} {name}{dims} = {{")
        if truncate_weights_after is not None and len(values) > truncate_weights_after:
            truncated_lines, _, _, _ = _emit_initializer_lines_truncated(
                values, shape, truncate_weights_after
            )
            lines.extend(truncated_lines)
        else:
            lines.extend(_emit_initializer_lines(values, shape))
        lines.append("};")
        lines.append("")
    return lines


def _insert_inline_weights(source: str, weight_lines: List[str]) -> str:
    if not weight_lines:
        return source
    lines = source.splitlines()
    insert_at = 0
    while insert_at < len(lines) and lines[insert_at].startswith("#include"):
        insert_at += 1
    while insert_at < len(lines) and lines[insert_at].strip() == "":
        insert_at += 1
    return "\n".join(lines[:insert_at] + weight_lines + lines[insert_at:]) + "\n"


def _build_variable_dim_names(
    graph: _GenericGraph,
    variable_dim_inputs: Dict[int, Sequence[int]],
    variable_dim_outputs: Dict[int, Sequence[int]],
) -> tuple[
    List[str], Dict[int, Dict[int, str]], Dict[int, Dict[int, str]], Dict[torch.fx.Node, Dict[int, str]]
]:
    dim_order: List[str] = []
    dim_names_inputs: Dict[int, Dict[int, str]] = {}
    dim_names_outputs: Dict[int, Dict[int, str]] = {}
    dim_vars: Dict[tuple[str, int, int], str] = {}

    def _register_dim(kind: str, tensor_index: int, dim_index: int) -> str:
        key = (kind, tensor_index, dim_index)
        if key not in dim_vars:
            dim_name = f"dim{len(dim_order) + 1}"
            dim_vars[key] = dim_name
            dim_order.append(dim_name)
        return dim_vars[key]

    def _build_dim_names(
        kind: str,
        tensor_index: int,
        shape: Sequence[int],
        variable_dims: Dict[int, Sequence[int]],
    ) -> Dict[int, str]:
        dim_names: Dict[int, str] = {}
        for dim_index in variable_dims.get(tensor_index, ()):
            if dim_index < 0 or dim_index >= len(shape):
                raise ValueError(
                    f"variable {kind} dim {dim_index} is out of range for shape {shape}"
                )
            dim_names[dim_index] = _register_dim(kind, tensor_index, dim_index)
        return dim_names

    for idx, node in enumerate(graph.tensor_placeholders):
        dim_names = _build_dim_names(
            "input", idx, graph.shapes[node], variable_dim_inputs
        )
        if dim_names:
            dim_names_inputs[idx] = dim_names

    for idx, node in enumerate(graph.output_nodes):
        dim_names = _build_dim_names(
            "output", idx, graph.shapes[node], variable_dim_outputs
        )
        if dim_names:
            dim_names_outputs[idx] = dim_names

    variable_dim_names: Dict[torch.fx.Node, Dict[int, str]] = {}
    for idx, node in enumerate(graph.tensor_placeholders):
        if idx in dim_names_inputs:
            variable_dim_names[node] = dim_names_inputs[idx]
    for idx, node in enumerate(graph.output_nodes):
        if idx in dim_names_outputs:
            variable_dim_names[node] = dim_names_outputs[idx]

    for op_node in graph.op_nodes:
        if op_node.node in variable_dim_names:
            continue
        for input_node in op_node.inputs:
            resolved = _resolve_alias(input_node, graph.alias_map)
            if resolved in variable_dim_names and graph.shapes[resolved] == tuple(
                op_node.output_shape
            ):
                variable_dim_names[op_node.node] = variable_dim_names[resolved]
                break

    for idx, node in enumerate(graph.output_nodes):
        if idx not in dim_names_outputs and node in variable_dim_names:
            dim_names_outputs[idx] = variable_dim_names[node]

    return dim_order, dim_names_inputs, dim_names_outputs, variable_dim_names


def _emit_model_wrapper(
    graph: _GenericGraph,
    weight_placeholders: Dict[torch.fx.Node, str],
    function_name: str,
    dim_order: Sequence[str],
    input_dim_names: Dict[int, Dict[int, str]],
    output_dim_names: Dict[int, Dict[int, str]],
) -> str:
    input_args: List[str] = []
    call_args: List[str] = []
    array_lines: List[str] = []

    use_array_signature = function_name in {"entry", "model_run"}
    input_index = 0
    for placeholder in graph.tensor_placeholders:
        weight_name = weight_placeholders.get(placeholder)
        if weight_name is not None:
            call_args.append(weight_name)
            continue
        c_type = _input_c_type(graph.dtypes[placeholder], graph.dtype)
        arg_name = f"in{input_index}"
        dim_names = input_dim_names.get(input_index, {})
        input_index += 1
        input_suffix = _format_array_suffix(graph.shapes[placeholder], dim_names)
        if use_array_signature:
            input_args.append(f"const {c_type} {arg_name}{input_suffix}")
            call_args.append(arg_name)
        else:
            input_args.append(f"const {c_type}* {arg_name}")
            array_name = f"{arg_name}_array"
            array_lines.append(
                f"    const {c_type} (*{array_name}){input_suffix} = "
                f"(const {c_type} (*){input_suffix}){arg_name};"
            )
            call_args.append(f"*{array_name}")
    output_args: List[str] = []
    for idx, output_node in enumerate(graph.output_nodes):
        output_c_type = _dtype_to_c_type(graph.dtypes[output_node], graph.dtype)
        output_name = f"out{idx}"
        dim_names = output_dim_names.get(idx, {})
        output_suffix = _format_array_suffix(graph.shapes[output_node], dim_names)
        if use_array_signature:
            output_args.append(f"{output_c_type} {output_name}{output_suffix}")
            call_args.append(output_name)
        else:
            output_args.append(f"{output_c_type}* {output_name}")
            array_name = f"{output_name}_array"
            array_lines.append(
                f"    {output_c_type} (*{array_name}){output_suffix} = "
                f"({output_c_type} (*){output_suffix}){output_name};"
            )
            call_args.append(f"*{array_name}")
    signature_parts = []
    dim_args = _format_dim_args(dim_order)
    if dim_args:
        signature_parts.append(dim_args)
    signature_parts.extend(input_args)
    signature_parts.extend(output_args)
    signature_args = ", ".join(signature_parts)
    call_parts = []
    if dim_order:
        call_parts.append(", ".join(dim_order))
    call_parts.extend(call_args)
    call = ", ".join(call_parts)
    wrapper_lines = [
        f"void {function_name}({signature_args}) {{",
        *array_lines,
        f"    ref_codegen_main_{graph.dtype.suffix}({call});",
        "}",
    ]
    return "\n".join(wrapper_lines) + "\n"


def export_generic_c(
    gm: torch.fx.GraphModule,
    example_inputs: Sequence[object],
    out_path: str,
    function_name: str = "model_run",
    truncate_weights_after: int | None = None,
    temp_allocation_threshold: int = 1024,
    variable_dim_inputs: Dict[int, Sequence[int]] | None = None,
    variable_dim_outputs: Dict[int, Sequence[int]] | None = None,
) -> str:
    if truncate_weights_after is not None and truncate_weights_after < 1:
        raise ValueError("truncate_weights_after must be >= 1")
    if temp_allocation_threshold < 0:
        raise ValueError("temp_allocation_threshold must be >= 0")
    (
        lifted_gm,
        lifted_inputs,
        weight_placeholders,
        weight_tensors,
    ) = _lift_get_attr_to_placeholders(gm, example_inputs)
    backend = CodegenBackend(temp_allocation_threshold=temp_allocation_threshold)
    graph = backend._analyze_generic_graph(lifted_gm, lifted_inputs)
    variable_dim_inputs = variable_dim_inputs or {}
    variable_dim_outputs = variable_dim_outputs or {}
    (
        dim_order,
        input_dim_names,
        output_dim_names,
        variable_dim_names,
    ) = _build_variable_dim_names(
        graph,
        variable_dim_inputs,
        variable_dim_outputs,
    )
    graph.variable_dim_names = variable_dim_names
    graph.variable_dim_order = dim_order
    source = backend._write_generic_source(graph)
    weights = {
        weight_placeholders[node]: tensor
        for node, tensor in zip(weight_placeholders.keys(), weight_tensors)
    }
    weight_lines = _emit_inline_weights(
        weights, graph.dtype, truncate_weights_after=truncate_weights_after
    )
    source = _insert_inline_weights(source, weight_lines)
    wrapper = _emit_model_wrapper(
        graph,
        weight_placeholders,
        function_name,
        dim_order=dim_order,
        input_dim_names=input_dim_names,
        output_dim_names=output_dim_names,
    )
    final_source = source.rstrip() + "\n\n" + wrapper
    Path(out_path).write_text(final_source, encoding="utf-8")
    return final_source
