import hashlib
import operator
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.fx
from torch.fx.immutable_collections import immutable_list

from c_ref_backend.cffi_bindings import RefBackendError


@dataclass(frozen=True)
class _OpSpec:
    name: str
    kind: str
    symbol: str | None
    supported_targets: set


SUPPORTED_OPS = {
    "add": _OpSpec(
        name="add",
        kind="binary",
        symbol="+",
        supported_targets={
            operator.add,
            torch.add,
            torch.ops.prims.add,
            torch.ops.prims.add.default,
            torch.ops.aten.add.Tensor,
        },
    ),
    "sub": _OpSpec(
        name="sub",
        kind="binary",
        symbol="-",
        supported_targets={
            operator.sub,
            torch.sub,
            torch.ops.prims.sub,
            torch.ops.prims.sub.default,
            torch.ops.aten.sub.Tensor,
        },
    ),
    "matmul": _OpSpec(
        name="matmul",
        kind="matmul",
        symbol=None,
        supported_targets={
            operator.matmul,
            torch.matmul,
            torch.ops.aten.mm,
            torch.ops.aten.mm.default,
            torch.ops.aten.matmul,
            torch.ops.aten.matmul.default,
        },
    ),
    "bmm": _OpSpec(
        name="bmm",
        kind="matmul",
        symbol=None,
        supported_targets={
            torch.bmm,
            torch.ops.aten.bmm,
            torch.ops.aten.bmm.default,
        },
    ),
    "relu": _OpSpec(
        name="relu",
        kind="unary",
        symbol=None,
        supported_targets={
            torch.relu,
            torch.ops.aten.relu,
            torch.ops.aten.relu.default,
        },
    ),
}


TARGET_TO_OP: Dict[object, _OpSpec] = {
    target: spec
    for spec in SUPPORTED_OPS.values()
    for target in spec.supported_targets
}


@dataclass
class _OpNode:
    node: torch.fx.Node
    spec: _OpSpec
    inputs: Tuple[torch.fx.Node, ...]
    output_shape: Tuple[int, ...]


@dataclass
class _GenericGraph:
    placeholders: List[torch.fx.Node]
    tensor_placeholders: List[torch.fx.Node]
    op_nodes: List[_OpNode]
    output_node: torch.fx.Node
    output_value: torch.fx.Node
    output_structure: object
    shapes: Dict[torch.fx.Node, Tuple[int, ...]]


@dataclass
class _GenericLibrary:
    so_path: Path
    lib: object
    input_shapes: Tuple[Tuple[int, ...], ...]
    output_shape: Tuple[int, ...]

    def run(self, inputs: Sequence[torch.Tensor], out: torch.Tensor) -> None:
        fn = getattr(self.lib, "ref_codegen_main_f32")
        args = [tensor.data_ptr() for tensor in inputs]
        args.append(out.data_ptr())
        fn(*args)


_LIBRARY_CACHE: Dict[str, object] = {}


def _format_array_suffix(shape: Sequence[int]) -> str:
    return "".join(f"[{dim}]" for dim in shape) or "[1]"


def _write_elementwise_kernel(
    node_index: int, op_spec: _OpSpec, shape: Sequence[int]
) -> List[str]:
    array_suffix = _format_array_suffix(shape)
    if op_spec.kind == "binary":
        signature = (
            f"void node{node_index}_{op_spec.name}_f32(const float a{array_suffix}, "
            f"const float b{array_suffix}, float out{array_suffix}) {{"
        )
    else:
        signature = (
            f"void node{node_index}_{op_spec.name}_f32(const float a{array_suffix}, "
            f"float out{array_suffix}) {{"
        )
    lines = [signature]
    indent = "    "
    if shape:
        for dim, size in enumerate(shape):
            lines.append(
                f"{indent}for (int64_t i{dim} = 0; i{dim} < {size}; ++i{dim}) {{"
            )
            indent += "    "
    index_expr = "".join(f"[i{dim}]" for dim in range(len(shape))) or "[0]"
    if op_spec.kind == "binary":
        lines.append(
            f"{indent}out{index_expr} = a{index_expr} {op_spec.symbol} b{index_expr};"
        )
    else:
        lines.append(
            f"{indent}out{index_expr} = a{index_expr} > 0.0f ? a{index_expr} : 0.0f;"
        )
    if shape:
        for _ in range(len(shape)):
            indent = indent[:-4]
            lines.append(f"{indent}}}")
    lines.append("}")
    return lines


def _write_matmul_kernel(
    node_index: int,
    op_spec: _OpSpec,
    a_shape: Sequence[int],
    b_shape: Sequence[int],
) -> List[str]:
    if op_spec.name == "matmul":
        m, k = a_shape
        _, n = b_shape
        a_suffix = _format_array_suffix((m, k))
        b_suffix = _format_array_suffix((k, n))
        out_suffix = _format_array_suffix((m, n))
        lines = [
            f"void node{node_index}_{op_spec.name}_f32(const float a{a_suffix}, const float b{b_suffix}, float out{out_suffix}) {{",
            f"    for (int64_t i = 0; i < {m}; ++i) {{",
            f"        for (int64_t j = 0; j < {n}; ++j) {{",
            "            float acc = 0.0f;",
            f"            for (int64_t t = 0; t < {k}; ++t) {{",
            "                acc += a[i][t] * b[t][j];",
            "            }",
            "            out[i][j] = acc;",
            "        }",
            "    }",
            "}",
        ]
        return lines
    batch, m, k = a_shape
    _, _, n = b_shape
    a_suffix = _format_array_suffix((batch, m, k))
    b_suffix = _format_array_suffix((batch, k, n))
    out_suffix = _format_array_suffix((batch, m, n))
    lines = [
        f"void node{node_index}_{op_spec.name}_f32(const float a{a_suffix}, const float b{b_suffix}, float out{out_suffix}) {{",
        f"    for (int64_t b_idx = 0; b_idx < {batch}; ++b_idx) {{",
        f"        for (int64_t i = 0; i < {m}; ++i) {{",
        f"            for (int64_t j = 0; j < {n}; ++j) {{",
        "                float acc = 0.0f;",
        f"                for (int64_t t = 0; t < {k}; ++t) {{",
        "                    acc += a[b_idx][i][t] * b[b_idx][t][j];",
        "                }",
        "                out[b_idx][i][j] = acc;",
        "            }",
        "        }",
        "    }",
        "}",
    ]
    return lines


def _write_generic_source(graph: _GenericGraph) -> str:
    placeholders = graph.tensor_placeholders
    op_nodes = graph.op_nodes
    lines = [
        "#include <stdint.h>",
        "",
    ]
    for index, op_node in enumerate(op_nodes, start=1):
        if op_node.spec.kind in {"binary", "unary"}:
            lines.extend(_write_elementwise_kernel(index, op_node.spec, op_node.output_shape))
        else:
            lhs, rhs = op_node.inputs
            lhs_shape = graph.shapes[lhs]
            rhs_shape = graph.shapes[rhs]
            lines.extend(
                _write_matmul_kernel(index, op_node.spec, lhs_shape, rhs_shape)
            )
        lines.append("")
    input_args = ", ".join(
        [
            f"const float input_{idx}{_format_array_suffix(graph.shapes[node])}"
            for idx, node in enumerate(placeholders)
        ]
    )
    input_args = f"{input_args}, " if input_args else ""
    lines.append(f"void ref_codegen_main_f32({input_args}float out{_format_array_suffix(graph.shapes[graph.output_value])}) {{")
    name_map: Dict[torch.fx.Node, str] = {}
    for idx, placeholder in enumerate(placeholders):
        name_map[placeholder] = f"input_{idx}"
    temp_index = 0
    for op_node in op_nodes:
        if op_node.node is graph.output_value:
            name_map[op_node.node] = "out"
        else:
            temp_name = f"tmp_{temp_index}"
            temp_index += 1
            name_map[op_node.node] = temp_name
            lines.append(
                f"    float {temp_name}{_format_array_suffix(op_node.output_shape)};"
            )
    for index, op_node in enumerate(op_nodes, start=1):
        input_names = [name_map[arg] for arg in op_node.inputs]
        output_name = name_map[op_node.node]
        args = ", ".join([*input_names, output_name])
        lines.append(f"    node{index}_{op_node.spec.name}_f32({args});")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _validate_example_inputs(example_inputs: Sequence[torch.Tensor]) -> None:
    tensor_examples = [example for example in example_inputs if isinstance(example, torch.Tensor)]
    if not tensor_examples:
        raise RefBackendError("codegen backend requires at least one example tensor input")
    for example in tensor_examples:
        if example.dtype is not torch.float32:
            raise RefBackendError("codegen backend supports only torch.float32 tensors")
        if example.device.type != "cpu":
            raise RefBackendError("codegen backend supports only CPU tensors")


def _unwrap_output_node(output_node: torch.fx.Node) -> Tuple[torch.fx.Node, object]:
    output_value = output_node.args[0]
    output_structure = output_value
    if isinstance(output_value, (tuple, list, immutable_list)):
        if len(output_value) != 1 or not isinstance(output_value[0], torch.fx.Node):
            raise RefBackendError("codegen backend expects a single output node")
        output_value = output_value[0]
    if not isinstance(output_value, torch.fx.Node):
        raise RefBackendError("codegen backend expects a single output node")
    return output_value, output_structure


def _infer_output_shape(
    op_spec: _OpSpec, input_shapes: Sequence[Tuple[int, ...]]
) -> Tuple[int, ...]:
    if op_spec.kind == "binary":
        a_shape, b_shape = input_shapes
        if a_shape != b_shape:
            raise RefBackendError(
                f"codegen {op_spec.name} requires inputs to have identical shapes"
            )
        return a_shape
    if op_spec.kind == "unary":
        return input_shapes[0]
    a_shape, b_shape = input_shapes
    if op_spec.name == "matmul":
        if len(a_shape) != 2 or len(b_shape) != 2:
            raise RefBackendError("codegen matmul requires 2D inputs")
        if a_shape[1] != b_shape[0]:
            raise RefBackendError("codegen matmul requires inner dimensions to match")
        return (a_shape[0], b_shape[1])
    if len(a_shape) != 3 or len(b_shape) != 3:
        raise RefBackendError("codegen bmm requires 3D inputs")
    if a_shape[0] != b_shape[0]:
        raise RefBackendError("codegen bmm requires batch dimensions to match")
    if a_shape[2] != b_shape[1]:
        raise RefBackendError("codegen bmm requires inner dimensions to match")
    return (a_shape[0], a_shape[1], b_shape[2])


def _analyze_generic_graph(
    gm: torch.fx.GraphModule, example_inputs: Sequence[torch.Tensor]
) -> _GenericGraph:
    _validate_example_inputs(example_inputs)
    output_node = None
    placeholders: List[torch.fx.Node] = []
    tensor_placeholders: List[torch.fx.Node] = []
    op_nodes: List[_OpNode] = []
    shapes: Dict[torch.fx.Node, Tuple[int, ...]] = {}
    input_iter = iter(example_inputs)

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            try:
                example = next(input_iter)
            except StopIteration as exc:
                raise RefBackendError(
                    "codegen backend expects example inputs to match placeholder count"
                ) from exc
            placeholders.append(node)
            if isinstance(example, torch.Tensor):
                shapes[node] = tuple(example.shape)
                tensor_placeholders.append(node)
            continue
        if node.op == "call_function":
            if node.kwargs:
                raise RefBackendError("codegen backend expects positional args only")
            op_spec = TARGET_TO_OP.get(node.target)
            if op_spec is None:
                raise RefBackendError(f"Unsupported call_function: {node.target}")
            expected_arity = 1 if op_spec.kind == "unary" else 2
            if len(node.args) != expected_arity:
                if expected_arity == 1:
                    raise RefBackendError(
                        f"codegen {op_spec.name} expects one input"
                    )
                raise RefBackendError(
                    f"codegen {op_spec.name} expects exactly two inputs"
                )
            input_nodes: List[torch.fx.Node] = []
            input_shapes: List[Tuple[int, ...]] = []
            for arg in node.args:
                if not isinstance(arg, torch.fx.Node):
                    raise RefBackendError(
                        f"codegen {op_spec.name} expects tensor inputs only"
                    )
                if arg not in shapes:
                    raise RefBackendError(
                        f"codegen {op_spec.name} expects tensor inputs only"
                    )
                input_nodes.append(arg)
                input_shapes.append(shapes[arg])
            output_shape = _infer_output_shape(op_spec, input_shapes)
            shapes[node] = output_shape
            op_nodes.append(
                _OpNode(
                    node=node,
                    spec=op_spec,
                    inputs=tuple(input_nodes),
                    output_shape=output_shape,
                )
            )
            continue
        if node.op == "output":
            output_node = node
            continue
        raise RefBackendError(f"Unsupported node op: {node.op}")

    try:
        next(input_iter)
    except StopIteration:
        pass
    else:
        raise RefBackendError(
            "codegen backend expects example inputs to match placeholder count"
        )

    if not op_nodes:
        raise RefBackendError("codegen backend requires at least one operation")
    if output_node is None:
        raise RefBackendError("codegen backend requires an output node")
    if not tensor_placeholders:
        raise RefBackendError("codegen backend requires at least one tensor input")
    output_value, output_structure = _unwrap_output_node(output_node)
    if output_value not in shapes:
        raise RefBackendError("codegen backend expects a single output node")
    if output_value not in {op.node for op in op_nodes}:
        raise RefBackendError("codegen backend output must be an operator result")

    return _GenericGraph(
        placeholders=placeholders,
        tensor_placeholders=tensor_placeholders,
        op_nodes=op_nodes,
        output_node=output_node,
        output_value=output_value,
        output_structure=output_structure,
        shapes=shapes,
    )


def _compile_generic_library(graph: _GenericGraph) -> _GenericLibrary:
    source = _write_generic_source(graph)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    cached = _LIBRARY_CACHE.get(digest)
    if cached is not None:
        return cached

    build_dir = Path(tempfile.mkdtemp(prefix="codegen_generic_"))
    c_path = build_dir / "ref_codegen_generic.c"
    so_path = build_dir / "ref_codegen_generic.so"
    c_path.write_text(source, encoding="utf-8")

    cmd = [
        "cc",
        "-shared",
        "-O3",
        "-fPIC",
        str(c_path),
        "-o",
        str(so_path),
    ]
    subprocess.check_call(cmd)

    import ctypes

    lib = ctypes.CDLL(str(so_path))
    argtypes = [ctypes.c_void_p for _ in graph.tensor_placeholders]
    argtypes.append(ctypes.c_void_p)
    lib.ref_codegen_main_f32.argtypes = argtypes
    lib.ref_codegen_main_f32.restype = None

    input_shapes = tuple(graph.shapes[node] for node in graph.tensor_placeholders)
    compiled = _GenericLibrary(
        so_path=so_path,
        lib=lib,
        input_shapes=input_shapes,
        output_shape=graph.shapes[graph.output_value],
    )
    _LIBRARY_CACHE[digest] = compiled
    return compiled


def _validate_runtime_inputs(inputs: Iterable[torch.Tensor]) -> None:
    for tensor in inputs:
        if tensor.dtype is not torch.float32:
            raise RefBackendError("codegen backend supports only torch.float32 tensors")
        if tensor.device.type != "cpu":
            raise RefBackendError("codegen backend supports only CPU tensors")


def _compile_graph(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
) -> Callable[..., torch.Tensor]:
    graph = _analyze_generic_graph(gm, example_inputs)
    lib = _compile_generic_library(graph)
    output_structure = graph.output_structure
    output_value = graph.output_value

    def resolve_output(value: object, env: Dict[torch.fx.Node, object]) -> object:
        if isinstance(value, torch.fx.Node):
            return env[value]
        if isinstance(value, (list, tuple, immutable_list)):
            resolved = [resolve_output(item, env) for item in value]
            return type(value)(resolved)
        return value

    def compiled(*args: object) -> object:
        if len(args) != len(graph.placeholders):
            raise RefBackendError(
                f"codegen backend expects {len(graph.placeholders)} inputs, got {len(args)}"
            )
        env: Dict[torch.fx.Node, object] = {}
        input_tensors = []
        for node, value in zip(graph.placeholders, args):
            env[node] = value
            if node in graph.tensor_placeholders:
                if not isinstance(value, torch.Tensor):
                    raise RefBackendError("codegen backend expects tensor inputs only")
                input_tensors.append(value)
        _validate_runtime_inputs(input_tensors)
        expected_shapes = lib.input_shapes
        for tensor, expected in zip(input_tensors, expected_shapes):
            if tuple(tensor.shape) != expected:
                raise RefBackendError(
                    f"codegen backend requires inputs to have shapes {expected_shapes}"
                )
        contiguous_inputs = [tensor.contiguous() for tensor in input_tensors]
        out = torch.empty(
            lib.output_shape,
            dtype=contiguous_inputs[0].dtype,
            device=contiguous_inputs[0].device,
        )
        lib.run(contiguous_inputs, out)
        env[output_value] = out
        return resolve_output(output_structure, env)

    return compiled


def get_generic_source(
    gm: torch.fx.GraphModule, example_inputs: Sequence[torch.Tensor]
) -> str:
    graph = _analyze_generic_graph(gm, example_inputs)
    return _write_generic_source(graph)


def codegen_generic_backend(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
) -> Callable[..., torch.Tensor]:
    return _compile_graph(gm, example_inputs)
