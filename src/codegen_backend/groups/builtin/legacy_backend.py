from __future__ import annotations

from dataclasses import dataclass, field
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

    def analyzers(
        self,
        supported_ops: Mapping[object, _OpSpec],
        target_registry: Mapping[object, _TargetInfo],
    ) -> Sequence[GroupAnalyzer]:
        return []


@dataclass(frozen=True)
class LegacyBackendGroup:
    name: str = "legacy"
    _group_ops: Dict[str, Mapping[str, _OpSpec]] = field(
        init=False,
        repr=False,
    )
    _group_targets: Dict[str, Mapping[object, _TargetInfo]] = field(
        init=False,
        repr=False,
    )
    _supported_ops: Mapping[str, _OpSpec] = field(init=False, repr=False)
    _target_registry: Mapping[object, _TargetInfo] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        group_ops = {
            "elementwise": build_elementwise_ops(),
            "reductions": build_reductions_ops(),
            "pooling": build_pooling_ops(),
            "conv": build_conv_ops(),
            "embedding": build_embedding_ops(),
            "tensor": build_tensor_ops(),
        }
        group_targets = {
            "elementwise": build_elementwise_targets(group_ops["elementwise"]),
            "reductions": build_reductions_targets(group_ops["reductions"]),
            "pooling": build_pooling_targets(group_ops["pooling"]),
            "conv": build_conv_targets(group_ops["conv"]),
            "embedding": build_embedding_targets(group_ops["embedding"]),
            "tensor": build_tensor_targets(group_ops["tensor"]),
        }
        supported_ops: Dict[str, _OpSpec] = {}
        for ops in group_ops.values():
            supported_ops.update(ops)
        target_registry: Dict[object, _TargetInfo] = {}
        for targets in group_targets.values():
            target_registry.update(targets)
        object.__setattr__(self, "_group_ops", group_ops)
        object.__setattr__(self, "_group_targets", group_targets)
        object.__setattr__(self, "_supported_ops", supported_ops)
        object.__setattr__(self, "_target_registry", target_registry)

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
        return self._supported_ops

    def target_registry(self) -> Mapping[object, _TargetInfo]:
        return self._target_registry

    def analyzers(
        self,
        supported_ops: Mapping[object, _OpSpec],
        target_registry: Mapping[object, _TargetInfo],
    ) -> Sequence[GroupAnalyzer]:
        from codegen_backend.groups.builtin.conv.analyzer import ConvAnalyzer
        from codegen_backend.groups.builtin.elementwise.analyzer import (
            ElementwiseAnalyzer,
        )
        from codegen_backend.groups.builtin.embedding.analyzer import EmbeddingAnalyzer
        from codegen_backend.groups.builtin.pooling.analyzer import PoolingAnalyzer
        from codegen_backend.groups.builtin.reductions.analyzer import ReductionsAnalyzer
        from codegen_backend.groups.builtin.tensor.analyzer import TensorAnalyzer

        elementwise_ops = self._group_ops["elementwise"]
        reductions_ops = self._group_ops["reductions"]
        pooling_ops = self._group_ops["pooling"]
        conv_ops = self._group_ops["conv"]
        embedding_ops = self._group_ops["embedding"]
        tensor_ops = self._group_ops["tensor"]
        return [
            ElementwiseAnalyzer(
                elementwise_ops,
                self._group_targets["elementwise"],
            ),
            ReductionsAnalyzer(
                reductions_ops,
                self._group_targets["reductions"],
            ),
            PoolingAnalyzer(
                pooling_ops,
                self._group_targets["pooling"],
            ),
            ConvAnalyzer(conv_ops, self._group_targets["conv"]),
            EmbeddingAnalyzer(
                embedding_ops,
                self._group_targets["embedding"],
            ),
            TensorAnalyzer(
                tensor_ops,
                self._group_targets["tensor"],
            ),
        ]


OperatorGroup = LegacyBackendGroup


__all__ = ["BaseBackendGroup", "LegacyBackendGroup", "OperatorGroup"]
