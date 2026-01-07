from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Sequence

from codegen_backend.groups.analysis import GroupAnalyzer
from codegen_backend.groups.builtin.embedding.analyzer import EmbeddingAnalyzer
from codegen_backend.groups.builtin.embedding import handlers
from codegen_backend.kinds import OpKindHandlerFactory
from codegen_backend.ops_registry_embedding import build_supported_ops
from codegen_backend.registry import _TargetInfo, build_target_registry
from codegen_backend.specs import _OpSpec


@dataclass(frozen=True)
class EmbeddingGroup:
    name: str = "embedding"

    def kind_handler_factories(self) -> List[OpKindHandlerFactory]:
        return [handlers.EmbeddingKindHandlerFactory()]

    def supported_ops(self) -> Mapping[str, _OpSpec]:
        return build_supported_ops()

    def target_registry(self) -> Mapping[object, _TargetInfo]:
        return build_target_registry(self.supported_ops())

    def analyzers(self) -> Sequence[GroupAnalyzer]:
        supported_ops = self.supported_ops()
        target_registry = build_target_registry(supported_ops)
        return [EmbeddingAnalyzer(supported_ops, target_registry)]


__all__ = ["EmbeddingGroup"]
