from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Sequence

from codegen_backend.groups.base import OperatorGroupDefinition
from codegen_backend.groups.builtin.reductions.analyzer import ReductionsAnalyzer
from codegen_backend.groups.builtin.reductions import handlers
from codegen_backend.groups.builtin.reductions.registry import (
    build_supported_ops,
    build_target_registry,
)
from codegen_backend.kinds import OpKindHandlerFactory
from codegen_backend.registry import _TargetInfo
from codegen_backend.specs import _OpSpec


@dataclass(frozen=True)
class ReductionsGroup(OperatorGroupDefinition):
    name: str = "reductions"

    def kind_handler_factories(self) -> List[OpKindHandlerFactory]:
        return [handlers.ReductionsKindHandlerFactory()]

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
    ) -> Sequence[ReductionsAnalyzer]:
        return [ReductionsAnalyzer(supported_ops, target_registry)]


__all__ = ["ReductionsGroup"]
