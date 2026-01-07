from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

from codegen_backend.kinds import HandlerContext, OpKindHandler, build_kind_handlers
from codegen_backend.ops_registry_conv import build_supported_ops as build_conv_ops
from codegen_backend.ops_registry_elementwise import (
    build_supported_ops as build_elementwise_ops,
)
from codegen_backend.ops_registry_embedding import (
    build_supported_ops as build_embedding_ops,
)
from codegen_backend.ops_registry_pooling import build_supported_ops as build_pooling_ops
from codegen_backend.ops_registry_reductions import (
    build_supported_ops as build_reductions_ops,
)
from codegen_backend.ops_registry_tensor import build_supported_ops as build_tensor_ops
from codegen_backend.registry import _TargetInfo, build_target_registry
from codegen_backend.specs import OpKind, _OpSpec


@dataclass(frozen=True)
class BaseBackendGroup:
    name: str = "base"

    def kind_handlers(
        self, context: HandlerContext
    ) -> Dict[OpKind, OpKindHandler]:
        return build_kind_handlers(context)

    def supported_ops(self) -> Mapping[str, _OpSpec]:
        return {}

    def target_registry(self) -> Mapping[object, _TargetInfo]:
        return {}


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

    def supported_ops(self) -> Mapping[str, _OpSpec]:
        supported_ops: Dict[str, _OpSpec] = {}
        supported_ops.update(build_elementwise_ops())
        supported_ops.update(build_reductions_ops())
        supported_ops.update(build_pooling_ops())
        supported_ops.update(build_conv_ops())
        supported_ops.update(build_embedding_ops())
        supported_ops.update(build_tensor_ops())
        return supported_ops

    def target_registry(self) -> Mapping[object, _TargetInfo]:
        return build_target_registry(self.supported_ops())


OperatorGroup = LegacyBackendGroup


__all__ = ["BaseBackendGroup", "LegacyBackendGroup", "OperatorGroup"]
