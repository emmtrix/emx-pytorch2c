from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Sequence

from codegen_backend.groups.analysis import GroupAnalyzer
from codegen_backend.kinds import OpKindHandlerFactory
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
from codegen_backend.specs import _OpSpec


@dataclass(frozen=True)
class BaseBackendGroup:
    name: str = "base"

    def kind_handler_factories(self) -> List[OpKindHandlerFactory]:
        return []

    def supported_ops(self) -> Mapping[str, _OpSpec]:
        return {}

    def target_registry(self) -> Mapping[object, _TargetInfo]:
        return {}

    def analyzers(self) -> Sequence[GroupAnalyzer]:
        return []


@dataclass(frozen=True)
class LegacyBackendGroup:
    name: str = "legacy"

    def kind_handler_factories(self) -> List[OpKindHandlerFactory]:
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

        return [
            elementwise_handlers.ElementwiseKindHandlerFactory(),
            reductions_handlers.ReductionsKindHandlerFactory(),
            pooling_handlers.PoolingKindHandlerFactory(),
            conv_handlers.ConvKindHandlerFactory(),
            embedding_handlers.EmbeddingKindHandlerFactory(),
            tensor_handlers.TensorKindHandlerFactory(),
        ]

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

    def analyzers(self) -> Sequence[GroupAnalyzer]:
        from codegen_backend.groups.builtin.conv.analyzer import ConvAnalyzer
        from codegen_backend.groups.builtin.elementwise.analyzer import (
            ElementwiseAnalyzer,
        )
        from codegen_backend.groups.builtin.embedding.analyzer import EmbeddingAnalyzer
        from codegen_backend.groups.builtin.pooling.analyzer import PoolingAnalyzer
        from codegen_backend.groups.builtin.reductions.analyzer import ReductionsAnalyzer
        from codegen_backend.groups.builtin.tensor.analyzer import TensorAnalyzer

        elementwise_ops = build_elementwise_ops()
        reductions_ops = build_reductions_ops()
        pooling_ops = build_pooling_ops()
        conv_ops = build_conv_ops()
        embedding_ops = build_embedding_ops()
        tensor_ops = build_tensor_ops()
        return [
            ElementwiseAnalyzer(
                elementwise_ops,
                build_target_registry(elementwise_ops),
            ),
            ReductionsAnalyzer(
                reductions_ops,
                build_target_registry(reductions_ops),
            ),
            PoolingAnalyzer(
                pooling_ops,
                build_target_registry(pooling_ops),
            ),
            ConvAnalyzer(conv_ops, build_target_registry(conv_ops)),
            EmbeddingAnalyzer(
                embedding_ops,
                build_target_registry(embedding_ops),
            ),
            TensorAnalyzer(
                tensor_ops,
                build_target_registry(tensor_ops),
            ),
        ]


OperatorGroup = LegacyBackendGroup


__all__ = ["BaseBackendGroup", "LegacyBackendGroup", "OperatorGroup"]
