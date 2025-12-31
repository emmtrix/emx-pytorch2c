import operator
from typing import Callable, Dict, List

import torch
import torch.fx
from torch.fx.immutable_collections import immutable_list
from torch._decomp import get_decompositions
from torch._functorch.aot_autograd import aot_module_simplified

from .cffi_bindings import RefBackendError, run_add, run_bmm, run_broadcast_in_dim, run_matmul


def _run_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_add(a, b, out)
    return out


def _run_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim == 3 and b.ndim == 3:
        out = torch.empty(
            (a.shape[0], a.shape[1], b.shape[2]),
            dtype=a.dtype,
            device=a.device,
            memory_format=torch.contiguous_format,
        )
        run_bmm(a, b, out)
        return out

    out = torch.empty(
        (a.shape[0], b.shape[1]),
        dtype=a.dtype,
        device=a.device,
        memory_format=torch.contiguous_format,
    )
    run_matmul(a, b, out)
    return out


def _run_broadcast_in_dim(
    a: torch.Tensor, shape: List[int], broadcast_dimensions: List[int]
) -> torch.Tensor:
    out = torch.empty(
        tuple(int(dim) for dim in shape),
        dtype=a.dtype,
        device=a.device,
        memory_format=torch.contiguous_format,
    )
    run_broadcast_in_dim(a, out, tuple(broadcast_dimensions))
    return out


def _run_expand(a: torch.Tensor, shape: List[int]) -> torch.Tensor:
    out_rank = len(shape)
    in_rank = a.ndim
    if out_rank < in_rank:
        raise RefBackendError("expand requires output rank >= input rank")
    broadcast_dimensions = list(range(out_rank - in_rank, out_rank))
    resolved_shape = []
    leading = out_rank - in_rank
    for idx, dim in enumerate(shape):
        if dim == -1:
            if idx < leading:
                raise RefBackendError("expand cannot infer leading broadcast dimension")
            dim = a.shape[idx - leading]
        resolved_shape.append(int(dim))
    return _run_broadcast_in_dim(a, resolved_shape, broadcast_dimensions)


def _compile_graph(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
) -> Callable[..., torch.Tensor]:
    supported_targets = {
        operator.add: ("add", _run_add),
        torch.add: ("add", _run_add),
        torch.ops.prims.add: ("add", _run_add),
        torch.ops.prims.add.default: ("add", _run_add),
        torch.ops.aten.add.Tensor: ("add", _run_add),
        operator.matmul: ("matmul", _run_matmul),
        torch.matmul: ("matmul", _run_matmul),
        torch.ops.aten.mm.default: ("matmul", _run_matmul),
        torch.ops.aten.mm: ("matmul", _run_matmul),
        torch.bmm: ("matmul", _run_matmul),
        torch.ops.aten.bmm.default: ("matmul", _run_matmul),
        torch.ops.aten.bmm: ("matmul", _run_matmul),
        torch.ops.aten.expand.default: ("expand", _run_expand),
        torch.ops.prims.broadcast_in_dim: ("broadcast_in_dim", _run_broadcast_in_dim),
        torch.ops.prims.broadcast_in_dim.default: (
            "broadcast_in_dim",
            _run_broadcast_in_dim,
        ),
    }

    def compiled(*args: torch.Tensor) -> torch.Tensor:
        env: Dict[str, torch.Tensor] = {}
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        if len(args) != len(placeholders):
            raise RefBackendError(
                f"Expected {len(placeholders)} inputs, got {len(args)}"
            )
        for node, value in zip(placeholders, args):
            env[node.name] = value

        def resolve_output(value: object) -> object:
            if isinstance(value, torch.fx.Node):
                return env[value.name]
            if isinstance(value, (list, tuple, immutable_list)):
                resolved = [resolve_output(item) for item in value]
                return type(value)(resolved)
            raise RefBackendError("Unsupported output format")

        for node in gm.graph.nodes:
            if node.op == "placeholder":
                continue
            if node.op == "call_function":
                if node.target not in supported_targets:
                    raise RefBackendError(f"Unsupported call_function: {node.target}")
                op_name, op_fn = supported_targets[node.target]
                if op_name == "broadcast_in_dim":
                    if node.kwargs:
                        raise RefBackendError(
                            "broadcast_in_dim expects positional arguments only"
                        )
                    if len(node.args) != 3:
                        raise RefBackendError(
                            "broadcast_in_dim expects tensor, shape, and dimensions"
                        )
                    input_arg, shape, broadcast_dimensions = node.args
                    if not isinstance(input_arg, torch.fx.Node):
                        raise RefBackendError(
                            "broadcast_in_dim expects tensor input only"
                        )
                    if isinstance(shape, torch.fx.Node) or isinstance(
                        broadcast_dimensions, torch.fx.Node
                    ):
                        raise RefBackendError(
                            "broadcast_in_dim expects constant shape and dimensions"
                        )
                    result = op_fn(
                        env[input_arg.name], list(shape), list(broadcast_dimensions)
                    )
                elif op_name == "expand":
                    if node.kwargs:
                        raise RefBackendError("expand expects positional arguments only")
                    if len(node.args) != 2:
                        raise RefBackendError("expand expects tensor and shape")
                    input_arg, shape = node.args
                    if not isinstance(input_arg, torch.fx.Node):
                        raise RefBackendError("expand expects tensor input only")
                    if isinstance(shape, torch.fx.Node):
                        raise RefBackendError("expand expects constant shape")
                    result = op_fn(env[input_arg.name], list(shape))
                else:
                    args_values = []
                    for arg in node.args:
                        if not isinstance(arg, torch.fx.Node):
                            raise RefBackendError(f"{op_name} expects tensor inputs only")
                        args_values.append(env[arg.name])
                    if len(args_values) != 2:
                        raise RefBackendError(f"{op_name} expects exactly two inputs")
                    result = op_fn(*args_values)
                env[node.name] = result
                continue
            if node.op == "output":
                output_val = node.args[0]
                return resolve_output(output_val)
            raise RefBackendError(f"Unsupported node op: {node.op}")
        raise RefBackendError("Graph has no output node")

    return compiled


def ref_backend_backend(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
) -> Callable[..., torch.Tensor]:
    if any(
        node.op == "call_function"
        and node.target
        in (
            torch.ops.prims.broadcast_in_dim,
            torch.ops.prims.broadcast_in_dim.default,
        )
        for node in gm.graph.nodes
    ):
        return _compile_graph(gm, example_inputs)

    decompositions = get_decompositions([torch.ops.aten.add.Tensor])

    def fw_compiler(
        fx_gm: torch.fx.GraphModule, fx_example_inputs: List[torch.Tensor]
    ) -> Callable[..., torch.Tensor]:
        return _compile_graph(fx_gm, fx_example_inputs)

    return aot_module_simplified(
        gm,
        example_inputs,
        fw_compiler=fw_compiler,
        decompositions=decompositions,
    )
