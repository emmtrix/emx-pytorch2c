from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Dict, Iterable, List, Sequence

from codegen_backend.emitters.registry import KindHandlerRegistration
from codegen_backend.groups.analysis import GroupAnalyzer
from codegen_backend.groups.base import OperatorGroupDefinition
from codegen_backend.groups.context import BackendContextProvider
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

    def build_context_provider(self, backend: object) -> HandlerContextProvider:
        provider: HandlerContextProvider = BackendContextProvider(backend)
        for group in self.groups:
            for factory in group.context_provider_factories():
                provider = factory.build_context_provider(provider, backend)
        return provider

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
        sources: Dict[object, tuple[str, str, _OpSpec]] = {}
        for group in self.groups:
            for target, target_info in group.target_registry().items():
                if target in merged:
                    existing_info = merged[target]
                    if (
                        existing_info != target_info
                        or existing_info.op_spec is not target_info.op_spec
                    ):
                        existing_group, existing_op, _ = sources[target]
                        raise ValueError(
                            "Target collision for "
                            f"{target!r}: "
                            f"{existing_group}/{existing_op} vs "
                            f"{group.name}/{target_info.op_spec.name}"
                        )
                merged[target] = target_info
                sources[target] = (
                    group.name,
                    target_info.op_spec.name,
                    target_info.op_spec,
                )
        return merged

    def merged_kind_handler_registrations(
        self,
    ) -> Dict[OpKind, KindHandlerRegistration]:
        merged: Dict[OpKind, KindHandlerRegistration] = {}
        for group in self.groups:
            merged.update(group.kind_handler_registrations())
        return merged


_GROUP_REGISTRY: GroupRegistry | None = None
_REGISTERED_GROUPS: Dict[str, OperatorGroupDefinition] = {}
_DEFAULT_GROUPS_LOADED = False
_ENTRY_POINTS_LOADED = False


def register_group(group: OperatorGroupDefinition) -> None:
    global _GROUP_REGISTRY
    _REGISTERED_GROUPS[group.name] = group
    _GROUP_REGISTRY = None


def _load_builtin_groups() -> None:
    global _DEFAULT_GROUPS_LOADED
    if _DEFAULT_GROUPS_LOADED:
        return
    _DEFAULT_GROUPS_LOADED = True
    from codegen_backend.groups.builtin.registration import register_builtin_groups

    register_builtin_groups()


def _iter_entry_points() -> Iterable[metadata.EntryPoint]:
    entry_points = metadata.entry_points()
    if hasattr(entry_points, "select"):
        return entry_points.select(group="codegen_backend.groups")
    return entry_points.get("codegen_backend.groups", [])


def _register_from_entry_point(entry_point: metadata.EntryPoint) -> None:
    loaded = entry_point.load()
    result = loaded() if callable(loaded) else loaded
    if result is None:
        return
    if isinstance(result, OperatorGroupDefinition):
        register_group(result)
        return
    if isinstance(result, Sequence):
        for item in result:
            if not isinstance(item, OperatorGroupDefinition):
                raise TypeError(
                    "entry point returned a sequence containing non-group items"
                )
            register_group(item)
        return
    raise TypeError("entry point did not return a group definition")


def _load_entry_point_groups() -> None:
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    for entry_point in _iter_entry_points():
        _register_from_entry_point(entry_point)


def get_group_registry() -> GroupRegistry:
    global _GROUP_REGISTRY
    if _GROUP_REGISTRY is None:
        _load_builtin_groups()
        _load_entry_point_groups()
        _GROUP_REGISTRY = GroupRegistry(groups=list(_REGISTERED_GROUPS.values()))
    return _GROUP_REGISTRY


__all__ = ["GroupRegistry", "get_group_registry", "register_group"]
