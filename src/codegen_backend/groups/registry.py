from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from codegen_backend.groups.analysis import GroupAnalyzer
from codegen_backend.groups.base import OperatorGroupDefinition
from codegen_backend.kinds import HandlerContextProvider, OpKindHandler
from codegen_backend.registry import _TargetInfo
from codegen_backend.specs import OpKind, _OpSpec


@dataclass(frozen=True)
class GroupRegistry:
    groups: List[OperatorGroupDefinition]

    def register(self, group: OperatorGroupDefinition) -> "GroupRegistry":
        return GroupRegistry(groups=[*self.groups, group])

    def build_kind_handlers(
        self, context_provider: HandlerContextProvider
    ) -> Dict[OpKind, OpKindHandler]:
        merged: Dict[OpKind, OpKindHandler] = {}
        for group in self.groups:
            for factory in group.kind_handler_factories():
                merged.update(factory.build_handlers(context_provider))
        return merged

    def build_group_analyzers(self) -> List[GroupAnalyzer]:
        analyzers: List[GroupAnalyzer] = []
        for group in self.groups:
            supported_ops = group.supported_ops()
            target_registry = group.target_registry()
            analyzers.extend(group.build_analyzers(supported_ops, target_registry))
        return analyzers

    def merged_supported_ops(self) -> Dict[object, _OpSpec]:
        merged: Dict[object, _OpSpec] = {}
        for group in self.groups:
            merged.update(group.supported_ops())
        return merged

    def merged_target_registry(self) -> Dict[object, _TargetInfo]:
        merged: Dict[object, _TargetInfo] = {}
        for group in self.groups:
            merged.update(group.target_registry())
        return merged


_GROUP_REGISTRY: GroupRegistry | None = None


def get_group_registry() -> GroupRegistry:
    global _GROUP_REGISTRY
    if _GROUP_REGISTRY is None:
        from codegen_backend.groups.builtin.conv.group import ConvGroup
        from codegen_backend.groups.builtin.elementwise.group import ElementwiseGroup
        from codegen_backend.groups.builtin.embedding.group import EmbeddingGroup
        from codegen_backend.groups.builtin.pooling.group import PoolingGroup
        from codegen_backend.groups.builtin.reductions.group import ReductionsGroup
        from codegen_backend.groups.builtin.tensor.group import TensorGroup

        _GROUP_REGISTRY = GroupRegistry(
            groups=[
                ElementwiseGroup(),
                ReductionsGroup(),
                PoolingGroup(),
                ConvGroup(),
                EmbeddingGroup(),
                TensorGroup(),
            ]
        )
    return _GROUP_REGISTRY


__all__ = ["GroupRegistry", "get_group_registry"]
