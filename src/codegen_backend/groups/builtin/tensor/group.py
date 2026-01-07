from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping, Sequence

from codegen_backend.groups.analysis import GroupAnalyzer
from codegen_backend.groups.base import OperatorGroupDefinition
from codegen_backend.groups.builtin.tensor.analyzer import TensorAnalyzer
from codegen_backend.groups.builtin.tensor import handlers
from codegen_backend.groups.builtin.tensor.registry import (
    build_supported_ops,
    build_target_registry,
)
from codegen_backend.kinds import OpKindHandlerFactory
from codegen_backend.registry import _TargetInfo
from codegen_backend.specs import _OpSpec


@dataclass(frozen=True)
class TensorGroup:
    name: str = "tensor"
    definition: OperatorGroupDefinition = field(
        default_factory=lambda: OperatorGroupDefinition(
            build_supported_ops,
            build_target_registry,
        )
    )

    def kind_handler_factories(self) -> List[OpKindHandlerFactory]:
        return [handlers.TensorKindHandlerFactory()]

    def supported_ops(self) -> Mapping[str, _OpSpec]:
        return self.definition.supported_ops

    def target_registry(self) -> Mapping[object, _TargetInfo]:
        return self.definition.target_registry

    def analyzers(
        self,
        supported_ops: Mapping[object, _OpSpec],
        target_registry: Mapping[object, _TargetInfo],
    ) -> Sequence[GroupAnalyzer]:
        return [TensorAnalyzer(supported_ops, target_registry)]


__all__ = ["TensorGroup"]
