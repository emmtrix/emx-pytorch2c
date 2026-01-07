from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol, Sequence

from codegen_backend.groups.analysis import GroupAnalyzer
from codegen_backend.kinds import HandlerContextProvider, OpKindHandlerFactory
from codegen_backend.registry import _TargetInfo
from codegen_backend.specs import _OpSpec


BuildSupportedOps = Callable[[], Mapping[object, _OpSpec]]
BuildTargetRegistry = Callable[[Mapping[object, _OpSpec]], Mapping[object, _TargetInfo]]


@dataclass(frozen=True)
class OperatorGroupDefinition:
    supported_ops: Mapping[object, _OpSpec] = field(init=False)
    target_registry: Mapping[object, _TargetInfo] = field(init=False)

    def __init__(
        self,
        build_supported_ops: BuildSupportedOps,
        build_target_registry: BuildTargetRegistry,
    ) -> None:
        supported_ops = build_supported_ops()
        target_registry = build_target_registry(supported_ops)
        object.__setattr__(self, "supported_ops", supported_ops)
        object.__setattr__(self, "target_registry", target_registry)


class OperatorGroup(Protocol):
    name: str

    def kind_handler_factories(
        self,
    ) -> Sequence[OpKindHandlerFactory]: ...

    def supported_ops(self) -> Mapping[object, _OpSpec]: ...

    def target_registry(self) -> Mapping[object, _TargetInfo]: ...

    def analyzers(
        self,
        supported_ops: Mapping[object, _OpSpec],
        target_registry: Mapping[object, _TargetInfo],
    ) -> Sequence[GroupAnalyzer]: ...


__all__ = ["OperatorGroup", "OperatorGroupDefinition"]
