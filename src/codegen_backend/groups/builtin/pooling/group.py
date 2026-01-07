from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Sequence

from codegen_backend.groups.analysis import GroupAnalyzer
from codegen_backend.groups.builtin.pooling.analyzer import PoolingAnalyzer
from codegen_backend.groups.builtin.pooling import handlers
from codegen_backend.groups.builtin.pooling.registry import (
    build_supported_ops,
    build_target_registry,
)
from codegen_backend.kinds import OpKindHandlerFactory
from codegen_backend.registry import _TargetInfo
from codegen_backend.specs import _OpSpec


@dataclass(frozen=True)
class PoolingGroup:
    name: str = "pooling"

    def kind_handler_factories(self) -> List[OpKindHandlerFactory]:
        return [handlers.PoolingKindHandlerFactory()]

    def supported_ops(self) -> Mapping[str, _OpSpec]:
        return build_supported_ops()

    def target_registry(self) -> Mapping[object, _TargetInfo]:
        return build_target_registry(self.supported_ops())

    def analyzers(self) -> Sequence[GroupAnalyzer]:
        supported_ops = self.supported_ops()
        target_registry = build_target_registry(supported_ops)
        return [PoolingAnalyzer(supported_ops, target_registry)]


__all__ = ["PoolingGroup"]
