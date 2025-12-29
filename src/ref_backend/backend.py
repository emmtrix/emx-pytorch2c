import operator
from typing import Callable, Dict, List

import torch
import torch.fx

from .cffi_bindings import RefBackendError, run_add


def _run_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    run_add(a, b, out)
    return out


def ref_backend_backend(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
) -> Callable[..., torch.Tensor]:
    supported_targets = {operator.add, torch.add}

    def compiled(*args: torch.Tensor) -> torch.Tensor:
        env: Dict[str, torch.Tensor] = {}
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        if len(args) != len(placeholders):
            raise RefBackendError(
                f"Expected {len(placeholders)} inputs, got {len(args)}"
            )
        for node, value in zip(placeholders, args):
            env[node.name] = value

        for node in gm.graph.nodes:
            if node.op == "placeholder":
                continue
            if node.op == "call_function":
                if node.target not in supported_targets:
                    raise RefBackendError(f"Unsupported call_function: {node.target}")
                args_values = []
                for arg in node.args:
                    if not isinstance(arg, torch.fx.Node):
                        raise RefBackendError("add expects tensor inputs only")
                    args_values.append(env[arg.name])
                if len(args_values) != 2:
                    raise RefBackendError("add expects exactly two inputs")
                result = _run_add(*args_values)
                env[node.name] = result
                continue
            if node.op == "output":
                output_val = node.args[0]
                if isinstance(output_val, torch.fx.Node):
                    return env[output_val.name]
                if isinstance(output_val, (list, tuple)):
                    if len(output_val) != 1 or not isinstance(output_val[0], torch.fx.Node):
                        raise RefBackendError("Only single-tensor outputs are supported")
                    return env[output_val[0].name]
                raise RefBackendError("Unsupported output format")
            raise RefBackendError(f"Unsupported node op: {node.op}")
        raise RefBackendError("Graph has no output node")

    return compiled
