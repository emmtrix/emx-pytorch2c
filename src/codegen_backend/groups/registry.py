from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from codegen_backend.groups.base import OperatorGroup
from codegen_backend.kinds import HandlerContext, OpKindHandler
from codegen_backend.registry import _TargetInfo
from codegen_backend.specs import OpKind, _OpSpec


@dataclass(frozen=True)
class GroupRegistry:
    groups: List[OperatorGroup]

    def register(self, group: OperatorGroup) -> "GroupRegistry":
        return GroupRegistry(groups=[*self.groups, group])

    def build_kind_handlers(
        self, context: HandlerContext
    ) -> Dict[OpKind, OpKindHandler]:
        merged: Dict[OpKind, OpKindHandler] = {}
        for group in self.groups:
            merged.update(group.kind_handlers(context))
        return merged

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
        from codegen_backend.groups.builtin.legacy_backend import LegacyBackendGroup

        _GROUP_REGISTRY = GroupRegistry(groups=[LegacyBackendGroup()])
    return _GROUP_REGISTRY


__all__ = ["GroupRegistry", "get_group_registry"]
