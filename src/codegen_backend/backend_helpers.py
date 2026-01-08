from __future__ import annotations

from typing import Dict, List

import torch
import torch.fx

from codegen_backend.analysis_helpers import is_out_overload
from codegen_backend.errors import CodegenBackendError
from codegen_backend.graph import _OpNode


def _resolve_alias(
    node: torch.fx.Node, alias_map: Dict[torch.fx.Node, torch.fx.Node]
) -> torch.fx.Node:
    visited: List[torch.fx.Node] = []
    visited_set = set()
    while node in alias_map:
        if node in visited_set:
            cycle_start_index = visited.index(node)
            cycle_nodes = visited[cycle_start_index:] + [node]
            cycle_names = " -> ".join(cycle_node.name for cycle_node in cycle_nodes)
            raise CodegenBackendError(
                "codegen backend alias resolution cycle detected: "
                f"{cycle_names}"
            )
        visited.append(node)
        visited_set.add(node)
        node = alias_map[node]
    return node


def _kernel_inputs(op_node: _OpNode) -> List[torch.fx.Node]:
    if is_out_overload(op_node.node.target) and op_node.inplace_input is not None:
        return [
            arg
            for index, arg in enumerate(op_node.inputs)
            if index != op_node.inplace_input
        ]
    return list(op_node.inputs)
