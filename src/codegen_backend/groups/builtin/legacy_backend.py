from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

from codegen_backend.groups.analysis import GroupAnalyzer
from codegen_backend.groups.builtin.conv.registry import (
    build_supported_ops as build_conv_ops,
    build_target_registry as build_conv_targets,
)
from codegen_backend.groups.builtin.elementwise.registry import (
    build_supported_ops as build_elementwise_ops,
    build_target_registry as build_elementwise_targets,
)
from codegen_backend.groups.builtin.embedding.registry import (
    build_supported_ops as build_embedding_ops,
    build_target_registry as build_embedding_targets,
)
from codegen_backend.groups.builtin.pooling.registry import (
    build_supported_ops as build_pooling_ops,
    build_target_registry as build_pooling_targets,
)
from codegen_backend.groups.builtin.reductions.registry import (
    build_supported_ops as build_reductions_ops,
    build_target_registry as build_reductions_targets,
)
from codegen_backend.groups.builtin.tensor.registry import (
    build_supported_ops as build_tensor_ops,
    build_target_registry as build_tensor_targets,
)
from codegen_backend.kinds import OpKindHandlerFactory
from codegen_backend.registry import _TargetInfo
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
        merged: Dict[object, _TargetInfo] = {}
        merged.update(build_elementwise_targets(build_elementwise_ops()))
        merged.update(build_reductions_targets(build_reductions_ops()))
        merged.update(build_pooling_targets(build_pooling_ops()))
        merged.update(build_conv_targets(build_conv_ops()))
        merged.update(build_embedding_targets(build_embedding_ops()))
        merged.update(build_tensor_targets(build_tensor_ops()))
        return merged

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
                build_elementwise_targets(elementwise_ops),
            ),
            ReductionsAnalyzer(
                reductions_ops,
                build_reductions_targets(reductions_ops),
            ),
            PoolingAnalyzer(
                pooling_ops,
                build_pooling_targets(pooling_ops),
            ),
            ConvAnalyzer(conv_ops, build_conv_targets(conv_ops)),
            EmbeddingAnalyzer(
                embedding_ops,
                build_embedding_targets(embedding_ops),
            ),
            TensorAnalyzer(
                tensor_ops,
                build_tensor_targets(tensor_ops),
            ),
        ]


OperatorGroup = LegacyBackendGroup


__all__ = ["BaseBackendGroup", "LegacyBackendGroup", "OperatorGroup"]
