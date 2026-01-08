from __future__ import annotations

import operator

import torch

from codegen_backend.ops_registry import _OpRegistry
from codegen_backend.registry import _TargetInfo, build_target_registry as _base_target_registry
from codegen_backend.specs import OpKind, _OpSpec


def build_supported_ops() -> dict[str, _OpSpec]:
    registry = _OpRegistry()

    registry.register_op("constant_pad_nd", kind=OpKind.PAD).targets(
        torch.ops.aten.constant_pad_nd,
        torch.ops.aten.constant_pad_nd.default,
    ).build()
    registry.register_op("arange", kind=OpKind.ARANGE).targets(
        torch.ops.aten.arange.start_step,
    ).build()
    registry.register_op("flip", kind=OpKind.FLIP).targets(
        torch.flip,
        torch.ops.aten.flip.default,
        torch.ops.aten.flip,
    ).build()
    registry.register_op("resize_", kind=OpKind.RESIZE).targets(
        torch.ops.aten.resize_.default,
    ).build()
    registry.register_op("empty_strided", OpKind.EMPTY_STRIDED).targets(
        torch.empty_strided,
        torch.ops.aten.empty_strided.default,
        torch.ops.aten.empty_strided,
    ).build()
    registry.register_op("as_strided", kind=OpKind.VIEW).targets(
        torch.ops.aten.as_strided.default,
        torch.ops.aten.as_strided,
    ).build()
    registry.register_op("reshape", kind=OpKind.VIEW).targets(
        torch.reshape,
        torch.ops.aten.reshape.default,
        torch.ops.aten.reshape,
    ).build()
    registry.register_op("flatten", kind=OpKind.VIEW).targets(
        torch.flatten,
        torch.ops.aten.flatten.using_ints,
        torch.ops.aten.flatten,
    ).build()
    registry.register_op("squeeze", kind=OpKind.VIEW).targets(
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
    ).build()
    registry.register_op("cat", kind=OpKind.CONCAT).targets(
        torch.cat,
        torch.ops.aten.cat.default,
        torch.ops.aten.cat,
    ).build()
    registry.register_op("gather", kind=OpKind.GATHER).targets(
        torch.gather,
        torch.ops.aten.gather.default,
        torch.ops.aten.gather,
    ).build()
    registry.register_op("index_put", kind=OpKind.INDEX_PUT).targets(
        torch.ops.aten.index_put.default,
        torch.ops.aten.index_put,
        torch.ops.aten.index_put_.default,
        torch.ops.aten.index_put_,
    ).inplace(
        torch.ops.aten.index_put_.default,
        torch.ops.aten.index_put_,
        arg_index=0,
    ).build()
    registry.register_op("diagonal", kind=OpKind.DIAGONAL).targets(
        torch.diagonal,
        torch.ops.aten.diagonal.default,
        torch.ops.aten.diagonal,
    ).build()
    registry.register_op("addmm", kind=OpKind.ADDMM).targets(
        torch.addmm,
        torch.ops.aten.addmm.default,
        torch.ops.aten.addmm,
        torch.ops.aten.addmm_.default,
        torch.ops.aten.addmm_,
    ).inplace(
        torch.ops.aten.addmm_.default,
        torch.ops.aten.addmm_,
        arg_index=0,
    ).build()
    registry.register_op("addbmm", kind=OpKind.ADDBMM).targets(
        torch.addbmm,
        torch.ops.aten.addbmm.default,
        torch.ops.aten.addbmm,
        torch.ops.aten.addbmm_.default,
        torch.ops.aten.addbmm_,
    ).inplace(
        torch.ops.aten.addbmm_.default,
        torch.ops.aten.addbmm_,
        arg_index=0,
    ).build()
    registry.register_op("addmv", kind=OpKind.ADDMV).targets(
        torch.addmv,
        torch.ops.aten.addmv.default,
        torch.ops.aten.addmv,
        torch.ops.aten.addmv_.default,
        torch.ops.aten.addmv_,
    ).inplace(
        torch.ops.aten.addmv_.default,
        torch.ops.aten.addmv_,
        arg_index=0,
    ).build()
    registry.register_op("addr", kind=OpKind.ADDR).targets(
        torch.addr,
        torch.ops.aten.addr.default,
        torch.ops.aten.addr,
        torch.ops.aten.addr_.default,
        torch.ops.aten.addr_,
    ).inplace(
        torch.ops.aten.addr_.default,
        torch.ops.aten.addr_,
        arg_index=0,
    ).build()
    registry.register_op("matmul", kind=OpKind.MATMUL).targets(
        operator.matmul,
        torch.matmul,
        torch.ops.aten.mm,
        torch.ops.aten.mm.default,
        torch.ops.aten.matmul,
        torch.ops.aten.matmul.default,
    ).build()
    registry.register_op("linear", kind=OpKind.LINEAR).targets(
        torch._C._nn.linear,
        torch.ops.aten.linear.default,
        torch.ops.aten.linear,
    ).build()
    registry.register_op("bmm", kind=OpKind.MATMUL).targets(
        torch.bmm,
        torch.ops.aten.bmm,
        torch.ops.aten.bmm.default,
    ).build()
    registry.register_op("_native_batch_norm_legit", kind=OpKind.BATCH_NORM).targets(
        torch.ops.aten._native_batch_norm_legit,
        torch.ops.aten._native_batch_norm_legit.default,
    ).build()
    registry.register_op(
        "_native_batch_norm_legit_no_training", kind=OpKind.BATCH_NORM
    ).targets(
        torch.ops.aten._native_batch_norm_legit_no_training,
        torch.ops.aten._native_batch_norm_legit_no_training.default,
    ).build()
    registry.register_op("_pdist_forward", kind=OpKind.PDIST).targets(
        torch.ops.aten._pdist_forward,
        torch.ops.aten._pdist_forward.default,
    ).build()
    registry.register_op("_cdist_forward", kind=OpKind.CDIST).targets(
        torch.ops.aten._cdist_forward,
        torch.ops.aten._cdist_forward.default,
    ).build()

    return registry.build()


def build_target_registry(
    supported_ops: dict[str, _OpSpec],
) -> dict[object, _TargetInfo]:
    return _base_target_registry(supported_ops)


__all__ = ["build_supported_ops", "build_target_registry"]
