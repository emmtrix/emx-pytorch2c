import hashlib
import operator
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import torch
import torch.fx
from torch.fx.immutable_collections import immutable_list

from c_ref_backend.cffi_bindings import RefBackendError


@dataclass(frozen=True)
class _BinaryOpSpec:
    name: str
    symbol: str
    supported_targets: set


@dataclass(frozen=True)
class _MatmulOpSpec:
    name: str
    supported_targets: set


SUPPORTED_BINARY_OPS = {
    "add": _BinaryOpSpec(
        name="add",
        symbol="+",
        supported_targets={
            operator.add,
            torch.add,
            torch.ops.prims.add,
            torch.ops.prims.add.default,
            torch.ops.aten.add.Tensor,
        },
    ),
    "sub": _BinaryOpSpec(
        name="sub",
        symbol="-",
        supported_targets={
            operator.sub,
            torch.sub,
            torch.ops.prims.sub,
            torch.ops.prims.sub.default,
            torch.ops.aten.sub.Tensor,
        },
    ),
}

SUPPORTED_MATMUL_OPS = {
    "matmul": _MatmulOpSpec(
        name="matmul",
        supported_targets={
            operator.matmul,
            torch.matmul,
            torch.ops.aten.mm,
            torch.ops.aten.mm.default,
            torch.ops.aten.matmul,
            torch.ops.aten.matmul.default,
        },
    ),
    "bmm": _MatmulOpSpec(
        name="bmm",
        supported_targets={
            torch.bmm,
            torch.ops.aten.bmm,
            torch.ops.aten.bmm.default,
        },
    ),
}


@dataclass
class _BinaryLibrary:
    so_path: Path
    lib: object
    input_count: int
    op_name: str
    shape: Tuple[int, ...]

    def run(self, inputs: Sequence[torch.Tensor], out: torch.Tensor) -> None:
        fn = getattr(self.lib, f"ref_codegen_main_f32")
        args = [tensor.data_ptr() for tensor in inputs]
        args.append(out.data_ptr())
        fn(*args)


_LIBRARY_CACHE: Dict[str, object] = {}


@dataclass
class _MatmulLibrary:
    so_path: Path
    lib: object
    input_count: int
    op_name: str
    input_shapes: Tuple[Tuple[int, ...], ...]
    output_shape: Tuple[int, ...]

    def run(self, inputs: Sequence[torch.Tensor], out: torch.Tensor) -> None:
        fn = getattr(self.lib, "ref_codegen_main_f32")
        args = [tensor.data_ptr() for tensor in inputs]
        args.append(out.data_ptr())
        fn(*args)


@dataclass
class _BinaryGraph:
    placeholders: List[torch.fx.Node]
    op_nodes: List[torch.fx.Node]
    output_node: torch.fx.Node
    output_value: torch.fx.Node
    tensor_placeholders: List[torch.fx.Node]


@dataclass
class _SingleOpGraph:
    placeholders: List[torch.fx.Node]
    op_node: torch.fx.Node
    output_node: torch.fx.Node
    output_value: torch.fx.Node
    tensor_placeholders: List[torch.fx.Node]


def _format_array_suffix(shape: Sequence[int]) -> str:
    return "".join(f"[{dim}]" for dim in shape) or "[1]"


def _write_binary_source(
    op_spec: _BinaryOpSpec, op_graph: _BinaryGraph, shape: Sequence[int]
) -> str:
    inputs = op_graph.tensor_placeholders
    op_nodes = op_graph.op_nodes
    array_suffix = _format_array_suffix(shape)
    input_args = ", ".join(
        [f"const float input_{idx}{array_suffix}" for idx in range(len(inputs))]
    )
    input_args = f"{input_args}, " if input_args else ""
    lines = [
        "#include <stdint.h>",
        "",
    ]
    for idx, _ in enumerate(op_nodes, start=1):
        index_expr = "".join(f"[i{dim}]" for dim in range(len(shape))) or "[0]"
        lines.extend(
            [
                f"void node{idx}_{op_spec.name}_f32(const float a{array_suffix}, const float b{array_suffix}, float out{array_suffix}) {{",
            ]
        )
        indent = "    "
        if shape:
            for dim, size in enumerate(shape):
                lines.append(
                    f"{indent}for (int64_t i{dim} = 0; i{dim} < {size}; ++i{dim}) {{"
                )
                indent += "    "
        lines.append(
            f"{indent}out{index_expr} = a{index_expr} {op_spec.symbol} b{index_expr};"
        )
        if shape:
            for _ in range(len(shape)):
                indent = indent[:-4]
                lines.append(f"{indent}}}")
        lines.extend(
            [
                "}",
                "",
            ]
        )
    lines.append(f"void ref_codegen_main_f32({input_args}float out{array_suffix}) {{")
    name_map: Dict[torch.fx.Node, str] = {}
    for idx, placeholder in enumerate(inputs):
        name_map[placeholder] = f"input_{idx}"
    output_op = op_graph.output_value
    for idx, op_node in enumerate(op_nodes):
        if op_node is output_op:
            out_name = "out"
        else:
            out_name = f"tmp_{idx}"
            lines.append(f"    float {out_name}{array_suffix};")
        name_map[op_node] = out_name
    for idx, op_node in enumerate(op_nodes, start=1):
        lhs, rhs = op_node.args
        lhs_name = name_map[lhs]
        rhs_name = name_map[rhs]
        out_name = name_map[op_node]
        lines.append(
            f"    node{idx}_{op_spec.name}_f32({lhs_name}, {rhs_name}, {out_name});"
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def _write_matmul_source(
    op_spec: _MatmulOpSpec, a_shape: Sequence[int], b_shape: Sequence[int]
) -> str:
    if op_spec.name == "matmul":
        m, k = a_shape
        _, n = b_shape
        a_suffix = _format_array_suffix((m, k))
        b_suffix = _format_array_suffix((k, n))
        out_suffix = _format_array_suffix((m, n))
        loop_prefix = []
        loop_suffix = []
        indent = "    "
        for label, size in (("i", m), ("j", n)):
            loop_prefix.append(
                f"{indent}for (int64_t {label} = 0; {label} < {size}; ++{label}) {{"
            )
            indent += "    "
        loop_prefix.append(f"{indent}float acc = 0.0f;")
        loop_prefix.append(
            f"{indent}for (int64_t t = 0; t < {k}; ++t) {{"
        )
        loop_prefix.append(
            f"{indent}    acc += input_0[i][t] * input_1[t][j];"
        )
        loop_prefix.append(f"{indent}}}")
        loop_prefix.append(f"{indent}out[i][j] = acc;")
        for _ in range(2):
            indent = indent[:-4]
            loop_suffix.append(f"{indent}}}")
        lines = [
            "#include <stdint.h>",
            "",
            f"void ref_codegen_main_f32(const float input_0{a_suffix}, const float input_1{b_suffix}, float out{out_suffix}) {{",
            *loop_prefix,
            *loop_suffix,
            "}",
        ]
        return "\n".join(lines) + "\n"
    batch, m, k = a_shape
    _, _, n = b_shape
    a_suffix = _format_array_suffix((batch, m, k))
    b_suffix = _format_array_suffix((batch, k, n))
    out_suffix = _format_array_suffix((batch, m, n))
    lines = [
        "#include <stdint.h>",
        "",
        f"void ref_codegen_main_f32(const float input_0{a_suffix}, const float input_1{b_suffix}, float out{out_suffix}) {{",
        f"    for (int64_t b = 0; b < {batch}; ++b) {{",
        f"        for (int64_t i = 0; i < {m}; ++i) {{",
        f"            for (int64_t j = 0; j < {n}; ++j) {{",
        f"                float acc = 0.0f;",
        f"                for (int64_t t = 0; t < {k}; ++t) {{",
        f"                    acc += input_0[b][i][t] * input_1[b][t][j];",
        f"                }}",
        f"                out[b][i][j] = acc;",
        f"            }}",
        f"        }}",
        f"    }}",
        "}",
    ]
    return "\n".join(lines) + "\n"


def _extract_shape(example_inputs: Sequence[torch.Tensor]) -> Tuple[int, ...]:
    if not example_inputs:
        raise RefBackendError(
            "codegen backend requires at least one example tensor input"
        )
    shape = tuple(example_inputs[0].shape)
    for tensor in example_inputs[1:]:
        if tuple(tensor.shape) != shape:
            raise RefBackendError(
                "codegen backend requires example inputs to share shapes"
            )
    return shape


def get_add_source(
    gm: torch.fx.GraphModule, example_inputs: Sequence[torch.Tensor]
) -> str:
    op_graph = _analyze_graph(SUPPORTED_BINARY_OPS["add"], gm)
    shape = _extract_shape(example_inputs)
    return _write_binary_source(SUPPORTED_BINARY_OPS["add"], op_graph, shape)


def get_sub_source(
    gm: torch.fx.GraphModule, example_inputs: Sequence[torch.Tensor]
) -> str:
    op_graph = _analyze_graph(SUPPORTED_BINARY_OPS["sub"], gm)
    shape = _extract_shape(example_inputs)
    return _write_binary_source(SUPPORTED_BINARY_OPS["sub"], op_graph, shape)


def get_matmul_source(
    gm: torch.fx.GraphModule, example_inputs: Sequence[torch.Tensor]
) -> str:
    return _get_matmul_source(SUPPORTED_MATMUL_OPS["matmul"], gm, example_inputs)


def get_bmm_source(
    gm: torch.fx.GraphModule, example_inputs: Sequence[torch.Tensor]
) -> str:
    return _get_matmul_source(SUPPORTED_MATMUL_OPS["bmm"], gm, example_inputs)


def _compile_binary_library(
    op_spec: _BinaryOpSpec, op_graph: _BinaryGraph, shape: Sequence[int]
) -> _BinaryLibrary:
    source = _write_binary_source(op_spec, op_graph, shape)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    cached = _LIBRARY_CACHE.get(digest)
    if cached is not None:
        return cached

    build_dir = Path(tempfile.mkdtemp(prefix=f"codegen_{op_spec.name}_"))
    c_path = build_dir / f"ref_codegen_{op_spec.name}.c"
    so_path = build_dir / f"ref_codegen_{op_spec.name}.so"
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
    argtypes = [ctypes.c_void_p for _ in op_graph.tensor_placeholders]
    argtypes.append(ctypes.c_void_p)
    lib.ref_codegen_main_f32.argtypes = argtypes
    lib.ref_codegen_main_f32.restype = None

    compiled = _BinaryLibrary(
        so_path=so_path,
        lib=lib,
        input_count=len(op_graph.tensor_placeholders),
        op_name=op_spec.name,
        shape=tuple(shape),
    )
    _LIBRARY_CACHE[digest] = compiled
    return compiled


def _compile_matmul_library(
    op_spec: _MatmulOpSpec, a_shape: Sequence[int], b_shape: Sequence[int]
) -> _MatmulLibrary:
    source = _write_matmul_source(op_spec, a_shape, b_shape)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    cached = _LIBRARY_CACHE.get(digest)
    if cached is not None:
        return cached

    build_dir = Path(tempfile.mkdtemp(prefix=f"codegen_{op_spec.name}_"))
    c_path = build_dir / f"ref_codegen_{op_spec.name}.c"
    so_path = build_dir / f"ref_codegen_{op_spec.name}.so"
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
    argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.ref_codegen_main_f32.argtypes = argtypes
    lib.ref_codegen_main_f32.restype = None

    if op_spec.name == "matmul":
        output_shape = (a_shape[0], b_shape[1])
    else:
        output_shape = (a_shape[0], a_shape[1], b_shape[2])

    compiled = _MatmulLibrary(
        so_path=so_path,
        lib=lib,
        input_count=2,
        op_name=op_spec.name,
        input_shapes=(tuple(a_shape), tuple(b_shape)),
        output_shape=tuple(output_shape),
    )
    _LIBRARY_CACHE[digest] = compiled
    return compiled


def _validate_binary_inputs(op_spec: _BinaryOpSpec, inputs: Sequence[torch.Tensor]) -> None:
    if not inputs:
        raise RefBackendError(f"codegen {op_spec.name} expects tensor inputs only")
    reference = inputs[0]
    if reference.dtype is not torch.float32:
        raise RefBackendError(
            f"codegen {op_spec.name} supports only torch.float32 tensors"
        )
    if reference.device.type != "cpu":
        raise RefBackendError(f"codegen {op_spec.name} supports only CPU tensors")
    reference_shape = reference.shape
    for tensor in inputs[1:]:
        if tensor.dtype is not torch.float32:
            raise RefBackendError(
                f"codegen {op_spec.name} supports only torch.float32 tensors"
            )
        if tensor.device.type != "cpu":
            raise RefBackendError(f"codegen {op_spec.name} supports only CPU tensors")
        if tensor.shape != reference_shape:
            raise RefBackendError(
                f"codegen {op_spec.name} requires inputs to have identical shapes"
            )


def _validate_matmul_inputs(op_name: str, inputs: Sequence[torch.Tensor]) -> None:
    if not inputs:
        raise RefBackendError(f"codegen {op_name} expects tensor inputs only")
    for tensor in inputs:
        if tensor.dtype is not torch.float32:
            raise RefBackendError(
                f"codegen {op_name} supports only torch.float32 tensors"
            )
        if tensor.device.type != "cpu":
            raise RefBackendError(f"codegen {op_name} supports only CPU tensors")


def _extract_matmul_shapes(
    op_name: str, example_inputs: Sequence[torch.Tensor]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    tensor_inputs = [
        example for example in example_inputs if isinstance(example, torch.Tensor)
    ]
    if len(tensor_inputs) != 2:
        raise RefBackendError(f"codegen {op_name} expects tensor inputs only")
    _validate_matmul_inputs(op_name, tensor_inputs)
    a_shape = tuple(tensor_inputs[0].shape)
    b_shape = tuple(tensor_inputs[1].shape)
    if op_name == "matmul":
        if len(a_shape) != 2 or len(b_shape) != 2:
            raise RefBackendError("codegen matmul requires 2D inputs")
        if a_shape[1] != b_shape[0]:
            raise RefBackendError("codegen matmul requires inner dimensions to match")
    else:
        if len(a_shape) != 3 or len(b_shape) != 3:
            raise RefBackendError("codegen bmm requires 3D inputs")
        if a_shape[0] != b_shape[0]:
            raise RefBackendError("codegen bmm requires batch dimensions to match")
        if a_shape[2] != b_shape[1]:
            raise RefBackendError("codegen bmm requires inner dimensions to match")
    return a_shape, b_shape


def _get_matmul_source(
    op_spec: _MatmulOpSpec,
    gm: torch.fx.GraphModule,
    example_inputs: Sequence[torch.Tensor],
) -> str:
    _analyze_single_op_graph(op_spec, gm)
    a_shape, b_shape = _extract_matmul_shapes(op_spec.name, example_inputs)
    return _write_matmul_source(op_spec, a_shape, b_shape)


def _analyze_graph(
    op_spec: _BinaryOpSpec, gm: torch.fx.GraphModule
) -> _BinaryGraph:
    output_node = None
    placeholders: List[torch.fx.Node] = []
    op_nodes: List[torch.fx.Node] = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            placeholders.append(node)
            continue
        if node.op == "call_function":
            if node.target not in op_spec.supported_targets:
                raise RefBackendError(f"Unsupported call_function: {node.target}")
            if node.kwargs:
                raise RefBackendError(
                    f"codegen {op_spec.name} backend expects positional {op_spec.name} args"
                )
            if len(node.args) != 2:
                raise RefBackendError(
                    f"codegen {op_spec.name} expects exactly two inputs"
                )
            lhs, rhs = node.args
            if not isinstance(lhs, torch.fx.Node) or not isinstance(
                rhs, torch.fx.Node
            ):
                raise RefBackendError(
                    f"codegen {op_spec.name} expects tensor inputs only"
                )
            op_nodes.append(node)
            continue
        if node.op == "output":
            output_node = node
            continue
        raise RefBackendError(f"Unsupported node op: {node.op}")

    if not op_nodes:
        raise RefBackendError(
            f"codegen {op_spec.name} backend requires a {op_spec.name} operation"
        )
    if output_node is None:
        raise RefBackendError(
            f"codegen {op_spec.name} backend requires an output node"
        )
    output_value = output_node.args[0]
    if isinstance(output_value, (tuple, list, immutable_list)):
        if len(output_value) != 1 or not isinstance(output_value[0], torch.fx.Node):
            raise RefBackendError(
                f"codegen {op_spec.name} backend expects a single output node"
            )
        output_value = output_value[0]
    if not isinstance(output_value, torch.fx.Node):
        raise RefBackendError(
            f"codegen {op_spec.name} backend expects a single output node"
        )
    if output_value not in op_nodes:
        raise RefBackendError(
            f"codegen {op_spec.name} backend output must be a {op_spec.name} node"
        )
    if output_value is not op_nodes[-1]:
        raise RefBackendError(
            f"codegen {op_spec.name} backend output must be the final {op_spec.name}"
        )
    return _BinaryGraph(
        placeholders=placeholders,
        op_nodes=op_nodes,
        output_node=output_node,
        output_value=output_value,
        tensor_placeholders=placeholders,
    )


def _analyze_single_op_graph(
    op_spec: _MatmulOpSpec, gm: torch.fx.GraphModule
) -> _SingleOpGraph:
    output_node = None
    placeholders: List[torch.fx.Node] = []
    op_nodes: List[torch.fx.Node] = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            placeholders.append(node)
            continue
        if node.op == "call_function":
            if node.target not in op_spec.supported_targets:
                raise RefBackendError(f"Unsupported call_function: {node.target}")
            if node.kwargs:
                raise RefBackendError(
                    f"codegen {op_spec.name} backend expects positional {op_spec.name} args"
                )
            if len(node.args) != 2:
                raise RefBackendError(
                    f"codegen {op_spec.name} expects exactly two inputs"
                )
            lhs, rhs = node.args
            if not isinstance(lhs, torch.fx.Node) or not isinstance(
                rhs, torch.fx.Node
            ):
                raise RefBackendError(
                    f"codegen {op_spec.name} expects tensor inputs only"
                )
            op_nodes.append(node)
            continue
        if node.op == "output":
            output_node = node
            continue
        raise RefBackendError(f"Unsupported node op: {node.op}")

    if not op_nodes:
        raise RefBackendError(
            f"codegen {op_spec.name} backend requires a {op_spec.name} operation"
        )
    if len(op_nodes) != 1:
        raise RefBackendError(
            f"codegen {op_spec.name} backend supports a single {op_spec.name} operation"
        )
    if output_node is None:
        raise RefBackendError(
            f"codegen {op_spec.name} backend requires an output node"
        )
    output_value = output_node.args[0]
    if isinstance(output_value, (tuple, list, immutable_list)):
        if len(output_value) != 1 or not isinstance(output_value[0], torch.fx.Node):
            raise RefBackendError(
                f"codegen {op_spec.name} backend expects a single output node"
            )
        output_value = output_value[0]
    if not isinstance(output_value, torch.fx.Node):
        raise RefBackendError(
            f"codegen {op_spec.name} backend expects a single output node"
        )
    if output_value is not op_nodes[0]:
        raise RefBackendError(
            f"codegen {op_spec.name} backend output must be a {op_spec.name} node"
        )
    return _SingleOpGraph(
        placeholders=placeholders,
        op_node=op_nodes[0],
        output_node=output_node,
        output_value=output_value,
        tensor_placeholders=placeholders,
    )


def _compile_graph(
    op_spec: _BinaryOpSpec,
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
) -> Callable[..., torch.Tensor]:
    op_graph = _analyze_graph(op_spec, gm)
    placeholders = op_graph.placeholders
    if len(example_inputs) != len(placeholders):
        raise RefBackendError(
            f"codegen {op_spec.name} backend expects example inputs to match placeholder count"
        )
    tensor_placeholders = [
        placeholder
        for placeholder, example in zip(placeholders, example_inputs)
        if isinstance(example, torch.Tensor)
    ]
    tensor_example_inputs = [
        example
        for placeholder, example in zip(placeholders, example_inputs)
        if isinstance(example, torch.Tensor)
    ]
    if not tensor_example_inputs:
        raise RefBackendError(
            f"codegen {op_spec.name} backend expects tensor inputs only"
        )
    shape = tuple(tensor_example_inputs[0].shape)
    for tensor in tensor_example_inputs[1:]:
        if tuple(tensor.shape) != shape:
            raise RefBackendError(
                f"codegen {op_spec.name} requires inputs to have identical shapes"
            )
    op_graph = _BinaryGraph(
        placeholders=op_graph.placeholders,
        op_nodes=op_graph.op_nodes,
        output_node=op_graph.output_node,
        output_value=op_graph.output_value,
        tensor_placeholders=tensor_placeholders,
    )
    output_node = op_graph.output_node
    output_value = op_graph.output_value
    output_structure = output_node.args[0]
    lib = _compile_binary_library(op_spec, op_graph, shape)

    def resolve_output(value: object, env: Dict[str, object]) -> object:
        if isinstance(value, torch.fx.Node):
            return env[value.name]
        if isinstance(value, (list, tuple, immutable_list)):
            resolved = [resolve_output(item, env) for item in value]
            return type(value)(resolved)
        return value

    def compiled(*args: object) -> object:
        if len(args) != len(placeholders):
            raise RefBackendError(
                f"codegen {op_spec.name} expects {len(placeholders)} inputs, got {len(args)}"
            )
        env: Dict[str, object] = {}
        for node, value in zip(placeholders, args):
            env[node.name] = value
        input_tensors = []
        for node in op_graph.tensor_placeholders:
            value = env[node.name]
            if not isinstance(value, torch.Tensor):
                raise RefBackendError(
                    f"codegen {op_spec.name} expects tensor inputs only"
                )
            input_tensors.append(value)
        _validate_binary_inputs(op_spec, input_tensors)
        if input_tensors[0].shape != lib.shape:
            raise RefBackendError(
                f"codegen {op_spec.name} requires inputs to have shape {lib.shape}"
            )
        contiguous_inputs = [tensor.contiguous() for tensor in input_tensors]
        out = torch.empty_like(
            contiguous_inputs[0], memory_format=torch.contiguous_format
        )
        lib.run(contiguous_inputs, out)
        env[output_value.name] = out
        return resolve_output(output_structure, env)

    return compiled


def _compile_matmul_graph(
    op_spec: _MatmulOpSpec,
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
) -> Callable[..., torch.Tensor]:
    op_graph = _analyze_single_op_graph(op_spec, gm)
    a_shape, b_shape = _extract_matmul_shapes(op_spec.name, example_inputs)
    lib = _compile_matmul_library(op_spec, a_shape, b_shape)
    output_node = op_graph.output_node
    output_value = op_graph.output_value
    output_structure = output_node.args[0]

    def resolve_output(value: object, env: Dict[str, object]) -> object:
        if isinstance(value, torch.fx.Node):
            return env[value.name]
        if isinstance(value, (list, tuple, immutable_list)):
            resolved = [resolve_output(item, env) for item in value]
            return type(value)(resolved)
        return value

    def compiled(*args: object) -> object:
        if len(args) != len(op_graph.placeholders):
            raise RefBackendError(
                f"codegen {op_spec.name} expects {len(op_graph.placeholders)} inputs, got {len(args)}"
            )
        env: Dict[str, object] = {}
        for node, value in zip(op_graph.placeholders, args):
            env[node.name] = value
        input_tensors = [value for value in args if isinstance(value, torch.Tensor)]
        if len(input_tensors) != 2:
            raise RefBackendError(
                f"codegen {op_spec.name} expects tensor inputs only"
            )
        _validate_matmul_inputs(op_spec.name, input_tensors)
        a_expected, b_expected = lib.input_shapes
        if tuple(input_tensors[0].shape) != a_expected or tuple(
            input_tensors[1].shape
        ) != b_expected:
            raise RefBackendError(
                f"codegen {op_spec.name} requires inputs to have shapes {a_expected} and {b_expected}"
            )
        contiguous_inputs = [tensor.contiguous() for tensor in input_tensors]
        out = torch.empty(
            lib.output_shape,
            dtype=contiguous_inputs[0].dtype,
            device=contiguous_inputs[0].device,
        )
        lib.run(contiguous_inputs, out)
        env[output_value.name] = out
        return resolve_output(output_structure, env)

    return compiled


def codegen_add_backend(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
) -> Callable[..., torch.Tensor]:
    return _compile_graph(SUPPORTED_BINARY_OPS["add"], gm, example_inputs)


def codegen_sub_backend(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
) -> Callable[..., torch.Tensor]:
    return _compile_graph(SUPPORTED_BINARY_OPS["sub"], gm, example_inputs)


def codegen_matmul_backend(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
) -> Callable[..., torch.Tensor]:
    return _compile_matmul_graph(SUPPORTED_MATMUL_OPS["matmul"], gm, example_inputs)


def codegen_bmm_backend(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
) -> Callable[..., torch.Tensor]:
    return _compile_matmul_graph(SUPPORTED_MATMUL_OPS["bmm"], gm, example_inputs)
