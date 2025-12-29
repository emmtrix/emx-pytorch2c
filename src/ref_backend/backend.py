import operator
from typing import Any, Callable, Dict, List

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
    supported_targets = {
        operator.add,
        torch.add,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add.default,
    }

    def resolve_arg(arg: Any, env: Dict[torch.fx.Node, torch.Tensor]) -> Any:
        return torch.fx.node.map_arg(arg, lambda n: env[n])

    def compiled(*args: torch.Tensor) -> torch.Tensor:
        env: Dict[torch.fx.Node, torch.Tensor] = {}
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        if len(args) != len(placeholders):
            raise RefBackendError(
                f"Expected {len(placeholders)} inputs, got {len(args)}"
            )
        for node, value in zip(placeholders, args):
            env[node] = value

        for node in gm.graph.nodes:
            if node.op == "placeholder":
                continue
            if node.op == "call_function":
                if node.target not in supported_targets:
                    raise RefBackendError(f"Unsupported call_function: {node.target}")
                resolved_args = resolve_arg(node.args, env)
                resolved_kwargs = resolve_arg(node.kwargs, env)
                args_values = list(resolved_args)
                alpha = resolved_kwargs.pop("alpha", None)
                out_arg = resolved_kwargs.pop("out", None)
                if resolved_kwargs:
                    raise RefBackendError(
                        f"Unsupported add kwargs: {sorted(resolved_kwargs.keys())}"
                    )
                if out_arg is not None:
                    raise RefBackendError("add does not support out= in this backend")
                if len(args_values) < 2:
                    raise RefBackendError("add expects at least two inputs")
                if len(args_values) > 2:
                    if alpha is not None:
                        raise RefBackendError("add received multiple alpha values")
                    alpha = args_values[2]
                if alpha is None:
                    alpha = 1
                if alpha != 1:
                    raise RefBackendError("add supports only alpha=1")
                a, b = args_values[:2]
                if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                    raise RefBackendError("add expects tensor inputs only")
                result = _run_add(a, b)
                env[node] = result
                continue
            if node.op == "output":
                output_val = node.args[0]
                if isinstance(output_val, torch.fx.Node):
                    return env[output_val]
                if isinstance(output_val, (list, tuple)):
                    if len(output_val) != 1 or not isinstance(output_val[0], torch.fx.Node):
                        raise RefBackendError("Only single-tensor outputs are supported")
                    return env[output_val[0]]
                raise RefBackendError("Unsupported output format")
            raise RefBackendError(f"Unsupported node op: {node.op}")
        raise RefBackendError("Graph has no output node")

    return compiled
