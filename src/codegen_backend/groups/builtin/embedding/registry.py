from __future__ import annotations

import torch
import torch.nn.functional as F

from codegen_backend.ops_registry import _OpRegistry
from codegen_backend.registry import _TargetInfo, build_target_registry as _base_target_registry
from codegen_backend.specs import OpKind, _OpSpec


def build_supported_ops() -> dict[str, _OpSpec]:
    registry = _OpRegistry()

    registry.register_op("embedding", kind=OpKind.EMBEDDING).targets(
        F.embedding,
        torch.ops.aten.embedding.default,
        torch.ops.aten.embedding,
    ).build()
    registry.register_op("_embedding_bag", kind=OpKind.EMBEDDING_BAG).targets(
        torch.ops.aten._embedding_bag.default,
        torch.ops.aten._embedding_bag,
    ).build()

    return registry.build()


def build_target_registry(
    supported_ops: dict[str, _OpSpec],
) -> dict[object, _TargetInfo]:
    return _base_target_registry(supported_ops)


__all__ = ["build_supported_ops", "build_target_registry"]
