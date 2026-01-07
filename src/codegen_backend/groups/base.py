from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from codegen_backend.groups.analysis import GroupAnalyzer
from codegen_backend.kinds import HandlerContextProvider, OpKindHandlerFactory
from codegen_backend.registry import _TargetInfo
from codegen_backend.specs import _OpSpec


class OperatorGroup(Protocol):
    name: str

    def kind_handler_factories(
        self,
    ) -> Sequence[OpKindHandlerFactory]: ...

    def supported_ops(self) -> Mapping[object, _OpSpec]: ...

    def target_registry(self) -> Mapping[object, _TargetInfo]: ...

    def analyzers(self) -> Sequence[GroupAnalyzer]: ...


__all__ = ["OperatorGroup"]
