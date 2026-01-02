import hashlib
import operator
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import torch
import torch.fx
from torch.fx.immutable_collections import immutable_list

from c_ref_backend.cffi_bindings import RefBackendError


@dataclass(frozen=True)
class _BinaryOpSpec:
    name: str
    symbol: str
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


@dataclass
class _BinaryLibrary:
    so_path: Path
    lib: object
    input_count: int
    op_name: str

    def run(self, inputs: Sequence[torch.Tensor], out: torch.Tensor) -> None:
        fn = getattr(self.lib, f"ref_codegen_main_f32")
        args = [tensor.data_ptr() for tensor in inputs]
        args.extend([out.data_ptr(), out.numel()])
        fn(*args)


_LIBRARY_CACHE: Dict[str, _BinaryLibrary] = {}


@dataclass
class _BinaryGraph:
    placeholders: List[torch.fx.Node]
    op_nodes: List[torch.fx.Node]
    output_node: torch.fx.Node
    output_value: torch.fx.Node
    tensor_placeholders: List[torch.fx.Node]


def _write_binary_source(op_spec: _BinaryOpSpec, op_graph: _BinaryGraph) -> str:
    inputs = op_graph.tensor_placeholders
    op_nodes = op_graph.op_nodes
    input_args = ", ".join(
        [f"const float* input_{idx}" for idx in range(len(inputs))]
    )
    input_args = f"{input_args}, " if input_args else ""
    lines = [
        "#include <stdint.h>",
        "#include <stdlib.h>",
        "",
    ]
    for idx, _ in enumerate(op_nodes, start=1):
        lines.extend(
            [
                f"void node{idx}_{op_spec.name}_f32(const float* a, const float* b, float* out, int64_t numel) {{",
                "    for (int64_t i = 0; i < numel; ++i) {",
                f"        out[i] = a[i] {op_spec.symbol} b[i];",
                "    }",
                "}",
                "",
            ]
        )
    lines.append(f"void ref_codegen_main_f32({input_args}float* out, int64_t numel) {{")
    name_map: Dict[torch.fx.Node, str] = {}
    for idx, placeholder in enumerate(inputs):
        name_map[placeholder] = f"input_{idx}"
    output_op = op_graph.output_value
    for idx, op_node in enumerate(op_nodes):
        if op_node is output_op:
            out_name = "out"
        else:
            out_name = f"tmp_{idx}"
            lines.append(f"    float* {out_name} = (float*)malloc(numel * sizeof(float));")
        name_map[op_node] = out_name
    for idx, op_node in enumerate(op_nodes, start=1):
        lhs, rhs = op_node.args
        lhs_name = name_map[lhs]
        rhs_name = name_map[rhs]
        out_name = name_map[op_node]
        lines.append(
            f"    node{idx}_{op_spec.name}_f32({lhs_name}, {rhs_name}, {out_name}, numel);"
        )
    for idx, op_node in reversed(list(enumerate(op_nodes))):
        if op_node is output_op:
            continue
        lines.append(f"    free(tmp_{idx});")
    lines.append("}")
    return "\n".join(lines) + "\n"


def get_add_source(gm: torch.fx.GraphModule) -> str:
    op_graph = _analyze_graph(SUPPORTED_BINARY_OPS["add"], gm)
    return _write_binary_source(SUPPORTED_BINARY_OPS["add"], op_graph)


def get_sub_source(gm: torch.fx.GraphModule) -> str:
    op_graph = _analyze_graph(SUPPORTED_BINARY_OPS["sub"], gm)
    return _write_binary_source(SUPPORTED_BINARY_OPS["sub"], op_graph)


def _compile_binary_library(
    op_spec: _BinaryOpSpec, op_graph: _BinaryGraph
) -> _BinaryLibrary:
    source = _write_binary_source(op_spec, op_graph)
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
    argtypes.extend([ctypes.c_void_p, ctypes.c_int64])
    lib.ref_codegen_main_f32.argtypes = argtypes
    lib.ref_codegen_main_f32.restype = None

    compiled = _BinaryLibrary(
        so_path=so_path,
        lib=lib,
        input_count=len(op_graph.tensor_placeholders),
        op_name=op_spec.name,
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
    lib = _compile_binary_library(op_spec, op_graph)

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
        contiguous_inputs = [tensor.contiguous() for tensor in input_tensors]
        out = torch.empty_like(
            contiguous_inputs[0], memory_format=torch.contiguous_format
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
