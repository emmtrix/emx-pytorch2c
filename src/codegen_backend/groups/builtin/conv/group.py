from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Sequence

from codegen_backend.groups.analysis import GroupAnalyzer
from codegen_backend.groups.builtin.conv.analyzer import ConvAnalyzer
from codegen_backend.groups.builtin.conv import handlers
from codegen_backend.kinds import OpKindHandlerFactory
from codegen_backend.ops_registry_conv import build_supported_ops
from codegen_backend.registry import _TargetInfo, build_target_registry
from codegen_backend.specs import _OpSpec


@dataclass(frozen=True)
class ConvGroup:
    name: str = "conv"

    def kind_handler_factories(self) -> List[OpKindHandlerFactory]:
        return [handlers.ConvKindHandlerFactory()]

    def supported_ops(self) -> Mapping[str, _OpSpec]:
        return build_supported_ops()

    def target_registry(self) -> Mapping[object, _TargetInfo]:
        return build_target_registry(self.supported_ops())

    def analyzers(self) -> Sequence[GroupAnalyzer]:
        supported_ops = self.supported_ops()
        target_registry = build_target_registry(supported_ops)
        return [ConvAnalyzer(supported_ops, target_registry)]


__all__ = ["ConvGroup"]
