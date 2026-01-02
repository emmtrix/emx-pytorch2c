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


SUPPORTED_ADD_TARGETS = {
    operator.add,
    torch.add,
    torch.ops.prims.add,
    torch.ops.prims.add.default,
    torch.ops.aten.add.Tensor,
}


@dataclass
class _AddLibrary:
    so_path: Path
    lib: object
    input_count: int

    def run(self, inputs: Sequence[torch.Tensor], out: torch.Tensor) -> None:
        fn = self.lib.ref_codegen_main_f32
        args = [tensor.data_ptr() for tensor in inputs]
        args.extend([out.data_ptr(), out.numel()])
        fn(*args)


_LIBRARY_CACHE: Dict[str, _AddLibrary] = {}


@dataclass
class _AddGraph:
    placeholders: List[torch.fx.Node]
    add_nodes: List[torch.fx.Node]
    output_node: torch.fx.Node
    output_value: torch.fx.Node
    tensor_placeholders: List[torch.fx.Node]


def _write_add_source(add_graph: _AddGraph) -> str:
    inputs = add_graph.tensor_placeholders
    add_nodes = add_graph.add_nodes
    input_args = ", ".join(
        [f"const float* input_{idx}" for idx in range(len(inputs))]
    )
    input_args = f"{input_args}, " if input_args else ""
    lines = [
        "#include <stdint.h>",
        "#include <stdlib.h>",
        "",
        "void ref_codegen_add_f32(const float* a, const float* b, float* out, int64_t numel) {",
        "    for (int64_t i = 0; i < numel; ++i) {",
        "        out[i] = a[i] + b[i];",
        "    }",
        "}",
        "",
        f"void ref_codegen_main_f32({input_args}float* out, int64_t numel) {{",
    ]
    name_map: Dict[torch.fx.Node, str] = {}
    for idx, placeholder in enumerate(inputs):
        name_map[placeholder] = f"input_{idx}"
    output_add = add_graph.output_value
    for idx, add_node in enumerate(add_nodes):
        if add_node is output_add:
            out_name = "out"
        else:
            out_name = f"tmp_{idx}"
            lines.append(f"    float* {out_name} = (float*)malloc(numel * sizeof(float));")
        name_map[add_node] = out_name
    for add_node in add_nodes:
        lhs, rhs = add_node.args
        lhs_name = name_map[lhs]
        rhs_name = name_map[rhs]
        out_name = name_map[add_node]
        lines.append(
            f"    ref_codegen_add_f32({lhs_name}, {rhs_name}, {out_name}, numel);"
        )
    for idx, add_node in reversed(list(enumerate(add_nodes))):
        if add_node is output_add:
            continue
        lines.append(f"    free(tmp_{idx});")
    lines.append("}")
    return "\n".join(lines) + "\n"


def get_add_source(gm: torch.fx.GraphModule) -> str:
    add_graph = _analyze_graph(gm)
    return _write_add_source(add_graph)


def _compile_add_library(add_graph: _AddGraph) -> _AddLibrary:
    source = _write_add_source(add_graph)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    cached = _LIBRARY_CACHE.get(digest)
    if cached is not None:
        return cached

    build_dir = Path(tempfile.mkdtemp(prefix="codegen_add_"))
    c_path = build_dir / "ref_codegen_add.c"
    so_path = build_dir / "ref_codegen_add.so"
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
    argtypes = [ctypes.c_void_p for _ in add_graph.tensor_placeholders]
    argtypes.extend([ctypes.c_void_p, ctypes.c_int64])
    lib.ref_codegen_main_f32.argtypes = argtypes
    lib.ref_codegen_main_f32.restype = None

    compiled = _AddLibrary(
        so_path=so_path, lib=lib, input_count=len(add_graph.tensor_placeholders)
    )
    _LIBRARY_CACHE[digest] = compiled
    return compiled


def _validate_add_inputs(inputs: Sequence[torch.Tensor]) -> None:
    if not inputs:
        raise RefBackendError("codegen add expects tensor inputs only")
    reference = inputs[0]
    if reference.dtype is not torch.float32:
        raise RefBackendError("codegen add supports only torch.float32 tensors")
    if reference.device.type != "cpu":
        raise RefBackendError("codegen add supports only CPU tensors")
    reference_shape = reference.shape
    for tensor in inputs[1:]:
        if tensor.dtype is not torch.float32:
            raise RefBackendError("codegen add supports only torch.float32 tensors")
        if tensor.device.type != "cpu":
            raise RefBackendError("codegen add supports only CPU tensors")
        if tensor.shape != reference_shape:
            raise RefBackendError("codegen add requires inputs to have identical shapes")


def _analyze_graph(gm: torch.fx.GraphModule) -> _AddGraph:
    output_node = None
    placeholders: List[torch.fx.Node] = []
    add_nodes: List[torch.fx.Node] = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            placeholders.append(node)
            continue
        if node.op == "call_function":
            if node.target not in SUPPORTED_ADD_TARGETS:
                raise RefBackendError(f"Unsupported call_function: {node.target}")
            if node.kwargs:
                raise RefBackendError("codegen add backend expects positional add args")
            if len(node.args) != 2:
                raise RefBackendError("codegen add expects exactly two inputs")
            lhs, rhs = node.args
            if not isinstance(lhs, torch.fx.Node) or not isinstance(
                rhs, torch.fx.Node
            ):
                raise RefBackendError("codegen add expects tensor inputs only")
            add_nodes.append(node)
            continue
        if node.op == "output":
            output_node = node
            continue
        raise RefBackendError(f"Unsupported node op: {node.op}")

    if not add_nodes:
        raise RefBackendError("codegen add backend requires an add operation")
    if output_node is None:
        raise RefBackendError("codegen add backend requires an output node")
    output_value = output_node.args[0]
    if isinstance(output_value, (tuple, list, immutable_list)):
        if len(output_value) != 1 or not isinstance(output_value[0], torch.fx.Node):
            raise RefBackendError("codegen add backend expects a single output node")
        output_value = output_value[0]
    if not isinstance(output_value, torch.fx.Node):
        raise RefBackendError("codegen add backend expects a single output node")
    if output_value not in add_nodes:
        raise RefBackendError("codegen add backend output must be an add node")
    if output_value is not add_nodes[-1]:
        raise RefBackendError("codegen add backend output must be the final add")
    return _AddGraph(
        placeholders=placeholders,
        add_nodes=add_nodes,
        output_node=output_node,
        output_value=output_value,
        tensor_placeholders=placeholders,
    )


def _compile_graph(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
) -> Callable[..., torch.Tensor]:
    add_graph = _analyze_graph(gm)
    placeholders = add_graph.placeholders
    if len(example_inputs) != len(placeholders):
        raise RefBackendError(
            "codegen add backend expects example inputs to match placeholder count"
        )
    tensor_placeholders = [
        placeholder
        for placeholder, example in zip(placeholders, example_inputs)
        if isinstance(example, torch.Tensor)
    ]
    add_graph = _AddGraph(
        placeholders=add_graph.placeholders,
        add_nodes=add_graph.add_nodes,
        output_node=add_graph.output_node,
        output_value=add_graph.output_value,
        tensor_placeholders=tensor_placeholders,
    )
    output_node = add_graph.output_node
    output_value = add_graph.output_value
    output_structure = output_node.args[0]
    lib = _compile_add_library(add_graph)

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
                f"codegen add expects {len(placeholders)} inputs, got {len(args)}"
            )
        env: Dict[str, object] = {}
        for node, value in zip(placeholders, args):
            env[node.name] = value
        input_tensors = []
        for node in add_graph.tensor_placeholders:
            value = env[node.name]
            if not isinstance(value, torch.Tensor):
                raise RefBackendError("codegen add expects tensor inputs only")
            input_tensors.append(value)
        _validate_add_inputs(input_tensors)
        contiguous_inputs = [
            tensor.contiguous() for tensor in input_tensors
        ]
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
    return _compile_graph(gm, example_inputs)
