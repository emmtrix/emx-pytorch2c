from __future__ import annotations

import torch
import torch.nn.functional as F

from codegen_backend.ops_registry import _OpRegistry
from codegen_backend.specs import OpKind, _OpSpec


def build_supported_ops() -> dict[str, _OpSpec]:
    registry = _OpRegistry()

    registry.register_op("avg_pool1d", kind=OpKind.POOL1D).targets(
        F.avg_pool1d,
        torch.ops.aten.avg_pool1d.default,
        torch.ops.aten.avg_pool1d,
    ).build()
    registry.register_op("adaptive_avg_pool1d", kind=OpKind.POOL1D).targets(
        F.adaptive_avg_pool1d,
        torch.ops.aten.adaptive_avg_pool1d.default,
        torch.ops.aten.adaptive_avg_pool1d,
    ).build()
    registry.register_op("adaptive_avg_pool2d", kind=OpKind.POOL2D).targets(
        F.adaptive_avg_pool2d,
        torch.ops.aten.adaptive_avg_pool2d.default,
        torch.ops.aten.adaptive_avg_pool2d,
        torch.ops.aten._adaptive_avg_pool2d.default,
        torch.ops.aten._adaptive_avg_pool2d,
    ).build()
    registry.register_op("adaptive_avg_pool3d", kind=OpKind.POOL3D).targets(
        F.adaptive_avg_pool3d,
        torch.ops.aten.adaptive_avg_pool3d.default,
        torch.ops.aten.adaptive_avg_pool3d,
        torch.ops.aten._adaptive_avg_pool3d.default,
        torch.ops.aten._adaptive_avg_pool3d,
    ).build()
    registry.register_op("_adaptive_avg_pool2d_backward", kind=OpKind.POOL2D_BACKWARD).targets(
        torch.ops.aten._adaptive_avg_pool2d_backward.default,
        torch.ops.aten._adaptive_avg_pool2d_backward,
    ).build()
    registry.register_op("max_pool1d", kind=OpKind.POOL1D).targets(
        F.max_pool1d,
        torch.ops.aten.max_pool1d.default,
        torch.ops.aten.max_pool1d,
    ).build()
    registry.register_op("avg_pool2d", kind=OpKind.POOL2D).targets(
        F.avg_pool2d,
        torch.ops.aten.avg_pool2d.default,
        torch.ops.aten.avg_pool2d,
    ).build()
    registry.register_op("max_pool2d", kind=OpKind.POOL2D).targets(
        F.max_pool2d,
        torch.ops.aten.max_pool2d.default,
        torch.ops.aten.max_pool2d,
    ).build()

    return registry.build()


__all__ = ["build_supported_ops"]
