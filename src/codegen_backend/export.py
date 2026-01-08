import re
import struct
from math import prod
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.fx

from codegen_backend.errors import CodegenBackendError
from codegen_backend.backend import CodegenBackend
from codegen_backend.c_types import _dtype_to_c_type, _input_c_type
from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.base import _format_array_suffix
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


def _emit_model_wrapper(
    graph: _GenericGraph,
    weight_placeholders: Dict[torch.fx.Node, str],
    function_name: str,
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
        input_index += 1
        input_suffix = _format_array_suffix(graph.shapes[placeholder])
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
        output_suffix = _format_array_suffix(graph.shapes[output_node])
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
    signature_args = ", ".join([*input_args, *output_args])
    call = ", ".join(call_args)
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
    source = backend._write_generic_source(graph)
    weights = {
        weight_placeholders[node]: tensor
        for node, tensor in zip(weight_placeholders.keys(), weight_tensors)
    }
    weight_lines = _emit_inline_weights(
        weights, graph.dtype, truncate_weights_after=truncate_weights_after
    )
    source = _insert_inline_weights(source, weight_lines)
    wrapper = _emit_model_wrapper(graph, weight_placeholders, function_name)
    final_source = source.rstrip() + "\n\n" + wrapper
    Path(out_path).write_text(final_source, encoding="utf-8")
    return final_source
