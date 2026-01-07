from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Sequence

from codegen_backend.emitters.registry import KindHandlerRegistration
from codegen_backend.groups.base import OperatorGroupDefinition
from codegen_backend.groups.builtin.tensor.analyzer import TensorAnalyzer
from codegen_backend.groups.builtin.tensor import handlers
from codegen_backend.groups.builtin.tensor.registry import (
    build_supported_ops,
    build_target_registry,
)
from codegen_backend.kinds import OpKindHandlerFactory
from codegen_backend.registry import _TargetInfo
from codegen_backend.specs import OpKind, _OpSpec


@dataclass(frozen=True)
class TensorGroup(OperatorGroupDefinition):
    name: str = "tensor"

    def kind_handler_factories(self) -> List[OpKindHandlerFactory]:
        return [handlers.TensorKindHandlerFactory()]

    def build_kind_handler_registrations(
        self,
    ) -> Mapping[OpKind, KindHandlerRegistration]:
        return handlers.build_kind_handler_registrations()

    def build_supported_ops(self) -> Mapping[str, _OpSpec]:
        return build_supported_ops()

    def build_target_registry(
        self, supported_ops: Mapping[str, _OpSpec]
    ) -> Mapping[object, _TargetInfo]:
        return build_target_registry(supported_ops)

    def build_analyzers(
        self,
        supported_ops: Mapping[str, _OpSpec],
        target_registry: Mapping[object, _TargetInfo],
    ) -> Sequence[TensorAnalyzer]:
        return [TensorAnalyzer(supported_ops, target_registry)]


__all__ = ["TensorGroup"]
