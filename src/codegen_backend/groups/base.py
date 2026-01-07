from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from codegen_backend.emitters.registry import KindHandlerRegistration
from codegen_backend.groups.analysis import GroupAnalyzer
from codegen_backend.registry import _TargetInfo
from codegen_backend.specs import OpKind, _OpSpec


@dataclass(frozen=True)
class OperatorGroupDefinition(ABC):
    name: str
    _supported_ops: Mapping[str, _OpSpec] = field(init=False, repr=False)
    _target_registry: Mapping[object, _TargetInfo] = field(init=False, repr=False)
    _kind_handler_registrations: Mapping[OpKind, KindHandlerRegistration] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        supported_ops = self.build_supported_ops()
        target_registry = self.build_target_registry(supported_ops)
        kind_handler_registrations = self.build_kind_handler_registrations()
        object.__setattr__(self, "_supported_ops", supported_ops)
        object.__setattr__(self, "_target_registry", target_registry)
        object.__setattr__(
            self, "_kind_handler_registrations", kind_handler_registrations
        )

    def supported_ops(self) -> Mapping[str, _OpSpec]:
        return self._supported_ops

    def target_registry(self) -> Mapping[object, _TargetInfo]:
        return self._target_registry

    def kind_handler_registrations(
        self,
    ) -> Mapping[OpKind, KindHandlerRegistration]:
        return self._kind_handler_registrations

    def analyzers(self) -> Sequence[GroupAnalyzer]:
        return self.build_analyzers(self._supported_ops, self._target_registry)

    @abstractmethod
    def build_supported_ops(self) -> Mapping[str, _OpSpec]:
        raise NotImplementedError

    @abstractmethod
    def build_target_registry(
        self, supported_ops: Mapping[str, _OpSpec]
    ) -> Mapping[object, _TargetInfo]:
        raise NotImplementedError

    def build_kind_handler_registrations(
        self,
    ) -> Mapping[OpKind, KindHandlerRegistration]:
        return {}

    @abstractmethod
    def build_analyzers(
        self,
        supported_ops: Mapping[str, _OpSpec],
        target_registry: Mapping[object, _TargetInfo],
    ) -> Sequence[GroupAnalyzer]:
        raise NotImplementedError


__all__ = ["OperatorGroupDefinition"]
