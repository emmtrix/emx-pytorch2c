from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import torch
import torch.fx

from codegen_backend.dtypes import _CodegenDType
from codegen_backend.errors import CodegenBackendError
from codegen_backend.graph import _OpNode
from codegen_backend.kinds import OpKind, OpKindHandler, OpNodeBuildResult
from codegen_backend.registry import _TargetInfo
from codegen_backend.specs import _OpSpec


@dataclass(frozen=True)
class GroupAnalysisResult:
    op_node: _OpNode | None
    dtype_info: _CodegenDType | None = None


class GroupAnalyzer(ABC):
    name: str

    @abstractmethod
    def match_node(self, node: torch.fx.Node) -> bool:
        raise NotImplementedError

    @abstractmethod
    def build_op_node(
        self,
        node: torch.fx.Node,
        *,
        dtype_info: _CodegenDType | None,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
        alias_map: Dict[torch.fx.Node, torch.fx.Node],
        empty_outputs: set[torch.fx.Node],
        kind_handlers: Dict[OpKind, OpKindHandler],
    ) -> GroupAnalysisResult:
        raise NotImplementedError


class RegistryGroupAnalyzer(GroupAnalyzer):
    def __init__(
        self,
        *,
        name: str,
        supported_ops: Mapping[str, _OpSpec],
        target_registry: Mapping[object, _TargetInfo],
    ) -> None:
        self.name = name
        self._supported_ops = dict(supported_ops)
        self._target_registry = dict(target_registry)

    def match_node(self, node: torch.fx.Node) -> bool:
        if node.op == "call_method":
            return node.target in self._supported_ops
        if node.op == "call_function":
            return node.target in self._target_registry
        return False

    def build_op_node(
        self,
        node: torch.fx.Node,
        *,
        dtype_info: _CodegenDType | None,
        shapes: Dict[torch.fx.Node, Tuple[int, ...]],
        strides: Dict[torch.fx.Node, Tuple[int, ...]],
        dtypes: Dict[torch.fx.Node, torch.dtype],
        scalar_values: Dict[torch.fx.Node, object],
        alias_map: Dict[torch.fx.Node, torch.fx.Node],
        empty_outputs: set[torch.fx.Node],
        kind_handlers: Dict[OpKind, OpKindHandler],
    ) -> GroupAnalysisResult:
        op_spec, inplace_input = self._resolve_op_spec(node)
        handler = kind_handlers.get(op_spec.kind)
        if handler is None:
            raise CodegenBackendError(
                "codegen backend does not support kind "
                f"'{op_spec.kind.value}'"
            )
        build_result = handler.build_op_node(
            node,
            op_spec,
            dtype_info,
            shapes,
            strides,
            dtypes,
            scalar_values,
            inplace_input,
        )
        if build_result is None:
            if dtype_info is None:
                raise CodegenBackendError(
                    "codegen backend requires at least one tensor input or a factory op dtype"
                )
            raise CodegenBackendError(
                "codegen backend does not support building kind "
                f"'{op_spec.kind.value}'"
            )
        return GroupAnalysisResult(build_result.op_node, build_result.dtype_info)

    def _resolve_op_spec(self, node: torch.fx.Node) -> tuple[_OpSpec, int | None]:
        if node.op == "call_method":
            return self._supported_ops[node.target], None
        if node.op == "call_function":
            target_info = self._target_registry[node.target]
            return target_info.op_spec, target_info.inplace_arg_index
        raise CodegenBackendError(f"Unsupported node op: {node.op}")


__all__ = ["GroupAnalysisResult", "GroupAnalyzer", "RegistryGroupAnalyzer"]
