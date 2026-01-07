from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

from codegen_backend.kinds import HandlerContext, OpKindHandler, build_kind_handlers
from codegen_backend.ops_registry import SUPPORTED_OPS
from codegen_backend.registry import TARGET_REGISTRY, _TargetInfo
from codegen_backend.specs import OpKind, _OpSpec


@dataclass(frozen=True)
class LegacyBackendGroup:
    name: str = "legacy"

    def kind_handlers(
        self, context: HandlerContext
    ) -> Dict[OpKind, OpKindHandler]:
        from codegen_backend.groups.builtin.conv import handlers as conv_handlers
        from codegen_backend.groups.builtin.elementwise import (
            handlers as elementwise_handlers,
        )
        from codegen_backend.groups.builtin.embedding import (
            handlers as embedding_handlers,
        )
        from codegen_backend.groups.builtin.pooling import handlers as pooling_handlers
        from codegen_backend.groups.builtin.reductions import (
            handlers as reductions_handlers,
        )
        from codegen_backend.groups.builtin.tensor import handlers as tensor_handlers

        kind_handlers = build_kind_handlers(context)
        kind_handlers.update(elementwise_handlers.build_handlers(context))
        kind_handlers.update(reductions_handlers.build_handlers(context))
        kind_handlers.update(pooling_handlers.build_handlers(context))
        kind_handlers.update(conv_handlers.build_handlers(context))
        kind_handlers.update(embedding_handlers.build_handlers(context))
        kind_handlers.update(tensor_handlers.build_handlers(context))
        return kind_handlers

    def supported_ops(self) -> Mapping[object, _OpSpec]:
        return SUPPORTED_OPS

    def target_registry(self) -> Mapping[object, _TargetInfo]:
        return TARGET_REGISTRY


OperatorGroup = LegacyBackendGroup


__all__ = ["LegacyBackendGroup", "OperatorGroup"]
