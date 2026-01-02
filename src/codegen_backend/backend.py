import hashlib
import operator
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

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

    def run(self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
        fn = self.lib.ref_codegen_add_f32
        fn(
            a.data_ptr(),
            b.data_ptr(),
            out.data_ptr(),
            out.numel(),
        )


_LIBRARY_CACHE: Dict[str, _AddLibrary] = {}


def _write_add_source() -> str:
    return """
#include <stdint.h>

void ref_codegen_add_f32(const float* a, const float* b, float* out, int64_t numel) {
    for (int64_t i = 0; i < numel; ++i) {
        out[i] = a[i] + b[i];
    }
}
"""


def get_add_source() -> str:
    return _write_add_source()


def _compile_add_library() -> _AddLibrary:
    source = _write_add_source()
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
    lib.ref_codegen_add_f32.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
    ]
    lib.ref_codegen_add_f32.restype = None

    compiled = _AddLibrary(so_path=so_path, lib=lib)
    _LIBRARY_CACHE[digest] = compiled
    return compiled


def _validate_add_inputs(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.dtype is not torch.float32 or b.dtype is not torch.float32:
        raise RefBackendError("codegen add supports only torch.float32 tensors")
    if a.device.type != "cpu" or b.device.type != "cpu":
        raise RefBackendError("codegen add supports only CPU tensors")
    if a.shape != b.shape:
        raise RefBackendError("codegen add requires inputs to have identical shapes")


def _compile_graph(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
) -> Callable[..., torch.Tensor]:
    add_node = None
    output_node = None
    placeholders = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            placeholders.append(node)
            continue
        if node.op == "call_function":
            if node.target not in SUPPORTED_ADD_TARGETS:
                raise RefBackendError(f"Unsupported call_function: {node.target}")
            if add_node is not None:
                raise RefBackendError("codegen add backend supports a single add")
            add_node = node
            continue
        if node.op == "output":
            output_node = node
            continue
        raise RefBackendError(f"Unsupported node op: {node.op}")

    if add_node is None:
        raise RefBackendError("codegen add backend requires an add operation")
    if output_node is None:
        raise RefBackendError("codegen add backend requires an output node")
    if add_node.kwargs:
        raise RefBackendError("codegen add backend expects positional add args")

    lib = _compile_add_library()

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

        if len(add_node.args) != 2:
            raise RefBackendError("codegen add expects exactly two inputs")
        lhs, rhs = add_node.args
        if not isinstance(lhs, torch.fx.Node) or not isinstance(rhs, torch.fx.Node):
            raise RefBackendError("codegen add expects tensor inputs only")
        a = env[lhs.name]
        b = env[rhs.name]
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise RefBackendError("codegen add expects tensor inputs only")
        _validate_add_inputs(a, b)
        a_contig = a.contiguous()
        b_contig = b.contiguous()
        out = torch.empty_like(a_contig, memory_format=torch.contiguous_format)
        lib.run(a_contig, b_contig, out)
        env[add_node.name] = out
        output_val = output_node.args[0]
        return resolve_output(output_val, env)

    return compiled


def codegen_add_backend(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
) -> Callable[..., torch.Tensor]:
    return _compile_graph(gm, example_inputs)
