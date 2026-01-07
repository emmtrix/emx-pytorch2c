from __future__ import annotations

import torch
import torch.nn.functional as F

from codegen_backend.ops_registry import _OpRegistry
from codegen_backend.specs import OpKind, _OpSpec


def build_supported_ops() -> dict[str, _OpSpec]:
    registry = _OpRegistry()

    registry.register_op("conv2d", kind=OpKind.CONV2D).targets(
        F.conv2d,
        torch.ops.aten.convolution.default,
        torch.ops.aten.convolution,
        torch.ops.aten.conv2d.default,
        torch.ops.aten.conv2d,
    ).build()
    registry.register_op("conv1d", kind=OpKind.CONV1D).targets(
        torch.ops.aten.conv1d.default,
        torch.ops.aten.conv1d,
    ).build()
    registry.register_op("col2im", kind=OpKind.COL2IM).targets(
        torch.ops.aten.col2im.default,
        torch.ops.aten.col2im,
    ).build()

    return registry.build()


__all__ = ["build_supported_ops"]
