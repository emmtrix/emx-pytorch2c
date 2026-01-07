from __future__ import annotations

from typing import Dict, Mapping, Protocol

from codegen_backend.kinds import HandlerContext, OpKindHandler
from codegen_backend.registry import _TargetInfo
from codegen_backend.specs import OpKind, _OpSpec


class OperatorGroup(Protocol):
    name: str

    def kind_handlers(
        self, context: HandlerContext
    ) -> Dict[OpKind, OpKindHandler]: ...

    def supported_ops(self) -> Mapping[object, _OpSpec]: ...

    def target_registry(self) -> Mapping[object, _TargetInfo]: ...


__all__ = ["OperatorGroup"]
