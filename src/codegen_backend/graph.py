from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.fx

from codegen_backend.dtypes import _CodegenDType
from codegen_backend.specs import _OpSpec


@dataclass
class _OpNode:
    node: torch.fx.Node
    spec: _OpSpec
    inputs: List[torch.fx.Node]
    output_shape: Tuple[int, ...] | List[int]
    inplace_input: int | None = None
    reduction_dims: Tuple[int, ...] | None = None
    keepdim: bool = False
    params: Dict[str, Any] = field(default_factory=dict)

    def p(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)


@dataclass
class _GenericGraph:
    placeholders: List[torch.fx.Node]
    tensor_placeholders: List[torch.fx.Node]
    op_nodes: List[_OpNode]
    output_node: torch.fx.Node
    output_value: torch.fx.Node
    output_op: _OpNode
    output_inplace_input: torch.fx.Node | None
    output_structure: object
    shapes: Dict[torch.fx.Node, Tuple[int, ...]]
    strides: Dict[torch.fx.Node, Tuple[int, ...]]
    dtypes: Dict[torch.fx.Node, torch.dtype]
    dtype: _CodegenDType
    alias_map: Dict[torch.fx.Node, torch.fx.Node]
    empty_outputs: set[torch.fx.Node]


@dataclass
class _GenericLibrary:
    so_path: Path
    lib: object
    input_shapes: Tuple[Tuple[int, ...], ...]
    input_strides: Tuple[Tuple[int, ...], ...]
    output_shape: Tuple[int, ...]
    dtype: _CodegenDType

    def run(self, inputs: Sequence[torch.Tensor], out: torch.Tensor) -> None:
        fn = getattr(self.lib, f"ref_codegen_main_{self.dtype.suffix}")
        args = [tensor.data_ptr() for tensor in inputs]
        args.append(out.data_ptr())
        fn(*args)
