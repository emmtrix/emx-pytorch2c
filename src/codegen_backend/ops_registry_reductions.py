from __future__ import annotations

import torch
import torch.nn.functional as F

from codegen_backend.ops_registry import _OpRegistry
from codegen_backend.specs import OpKind, _OpSpec


def build_supported_ops() -> dict[str, _OpSpec]:
    registry = _OpRegistry()

    registry.register_op("_softmax", kind=OpKind.SOFTMAX).targets(
        torch.ops.aten._softmax,
        torch.ops.aten._softmax.default,
    ).build()
    registry.register_op("cumsum", kind=OpKind.CUMSUM).targets(
        torch.cumsum,
        torch.ops.aten.cumsum.default,
        torch.ops.aten.cumsum,
    ).build()
    registry.register_op("argmax", kind=OpKind.ARG_REDUCTION).targets(
        torch.argmax,
        torch.ops.aten.argmax.default,
        torch.ops.aten.argmax,
    ).build()
    registry.register_op("argmin", kind=OpKind.ARG_REDUCTION).targets(
        torch.argmin,
        torch.ops.aten.argmin.default,
        torch.ops.aten.argmin,
    ).build()
    registry.register_op("sum", kind=OpKind.REDUCTION).targets(
        torch.ops.aten.sum.default,
        torch.ops.aten.sum.dim_IntList,
    ).build()
    registry.register_op("prod", kind=OpKind.REDUCTION).targets(
        torch.ops.aten.prod.default,
        torch.ops.aten.prod.dim_int,
    ).build()
    registry.register_op("mean", kind=OpKind.REDUCTION).targets(
        torch.mean,
        torch.ops.aten.mean.default,
        torch.ops.aten.mean,
        torch.ops.aten.mean.dim,
    ).build()
    registry.register_op("std", kind=OpKind.REDUCTION).targets(
        torch.std,
        torch.ops.aten.std.default,
        torch.ops.aten.std,
    ).build()
    registry.register_op("var", kind=OpKind.REDUCTION).targets(
        torch.var,
        torch.ops.aten.var.default,
        torch.ops.aten.var.dim,
    ).build()
    registry.register_op("norm", kind=OpKind.REDUCTION).targets(
        torch.norm,
        torch.ops.aten.norm.Scalar,
        torch.ops.aten.norm.ScalarOpt_dim,
    ).build()
    registry.register_op("any", kind=OpKind.REDUCTION).targets(
        torch.any,
        torch.ops.aten.any.default,
        torch.ops.aten.any.dim,
        torch.ops.aten.any.dims,
        torch.ops.aten.any,
    ).build()
    registry.register_op("all", kind=OpKind.REDUCTION).targets(
        torch.all,
        torch.ops.aten.all.default,
        torch.ops.aten.all,
    ).build()
    registry.register_op("amax", kind=OpKind.REDUCTION).targets(
        torch.amax,
        torch.ops.aten.amax.default,
        torch.ops.aten.amax,
    ).build()
    registry.register_op("amin", kind=OpKind.REDUCTION).targets(
        torch.amin,
        torch.ops.aten.amin.default,
        torch.ops.aten.amin,
    ).build()
    registry.register_op("softmax", kind=OpKind.SOFTMAX).targets(
        torch.softmax,
        F.softmax,
        torch.ops.aten.softmax.int,
        torch.ops.aten.softmax,
    ).build()
    registry.register_op("log_softmax", kind=OpKind.SOFTMAX).targets(
        torch.log_softmax,
        F.log_softmax,
        torch.ops.aten.log_softmax.int,
        torch.ops.aten.log_softmax,
    ).build()
    registry.register_op("_log_softmax", kind=OpKind.SOFTMAX).targets(
        torch.ops.aten._log_softmax.default,
        torch.ops.aten._log_softmax,
    ).build()

    return registry.build()


__all__ = ["build_supported_ops"]
