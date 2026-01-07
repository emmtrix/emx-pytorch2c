from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

from codegen_backend.groups.builtin.embedding import handlers
from codegen_backend.kinds import HandlerContext, OpKindHandler
from codegen_backend.ops_registry_embedding import build_supported_ops
from codegen_backend.registry import _TargetInfo, build_target_registry
from codegen_backend.specs import OpKind, _OpSpec


@dataclass(frozen=True)
class EmbeddingGroup:
    name: str = "embedding"

    def kind_handlers(
        self, context: HandlerContext
    ) -> Dict[OpKind, OpKindHandler]:
        return handlers.build_handlers(context)

    def supported_ops(self) -> Mapping[str, _OpSpec]:
        return build_supported_ops()

    def target_registry(self) -> Mapping[object, _TargetInfo]:
        return build_target_registry(self.supported_ops())


__all__ = ["EmbeddingGroup"]
