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
    registry.register_op("reflection_pad1d", kind=OpKind.PAD).targets(
        torch.ops.aten.reflection_pad1d,
        torch.ops.aten.reflection_pad1d.default,
    ).build()
    registry.register_op("reflection_pad2d", kind=OpKind.PAD).targets(
        torch.ops.aten.reflection_pad2d,
        torch.ops.aten.reflection_pad2d.default,
    ).build()
    registry.register_op("reflection_pad3d", kind=OpKind.PAD).targets(
        torch.ops.aten.reflection_pad3d,
        torch.ops.aten.reflection_pad3d.default,
    ).build()
    registry.register_op("replication_pad2d", kind=OpKind.PAD).targets(
        torch.ops.aten.replication_pad2d,
        torch.ops.aten.replication_pad2d.default,
    ).build()
    registry.register_op("replication_pad3d", kind=OpKind.PAD).targets(
        torch.ops.aten.replication_pad3d,
        torch.ops.aten.replication_pad3d.default,
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
    registry.register_op("select", kind=OpKind.VIEW).targets(
        torch.ops.aten.select.int,
        torch.ops.aten.select,
    ).build()
    registry.register_op("empty_strided", OpKind.EMPTY_STRIDED).targets(
        torch.empty_strided,
        torch.ops.aten.empty_strided.default,
        torch.ops.aten.empty_strided,
    ).build()
    registry.register_op("scalar_tensor", OpKind.SCALAR_TENSOR).targets(
        torch.scalar_tensor,
        torch.ops.aten.scalar_tensor.default,
        torch.ops.aten.scalar_tensor,
    ).build()
    registry.register_op("as_strided", kind=OpKind.VIEW).targets(
        torch.ops.aten.as_strided.default,
        torch.ops.aten.as_strided,
    ).build()
    registry.register_op("_local_scalar_dense", kind=OpKind.VIEW).targets(
        torch.ops.aten._local_scalar_dense.default,
        torch.ops.aten._local_scalar_dense,
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
    registry.register_op("repeat", kind=OpKind.REPEAT).targets(
        torch.ops.aten.repeat.default,
        torch.ops.aten.repeat,
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
    registry.register_op("masked_scatter", kind=OpKind.MASKED_SCATTER).targets(
        torch.masked_scatter,
        torch.ops.aten.masked_scatter.default,
        torch.ops.aten.masked_scatter,
        torch.ops.aten.masked_scatter_.default,
        torch.ops.aten.masked_scatter_,
    ).inplace(
        torch.ops.aten.masked_scatter_.default,
        torch.ops.aten.masked_scatter_,
        arg_index=0,
    ).build()
    registry.register_op("select_scatter", kind=OpKind.SELECT_SCATTER).targets(
        torch.ops.aten.select_scatter.default,
        torch.ops.aten.select_scatter,
    ).build()
    registry.register_op("scatter_src", kind=OpKind.SCATTER).targets(
        torch.ops.aten.scatter.src,
        torch.ops.aten.scatter_.src,
    ).inplace(
        torch.ops.aten.scatter_.src,
        arg_index=0,
    ).build()
    registry.register_op("scatter_value", kind=OpKind.SCATTER).targets(
        torch.ops.aten.scatter.value,
        torch.ops.aten.scatter_.value,
    ).inplace(
        torch.ops.aten.scatter_.value,
        arg_index=0,
    ).build()
    registry.register_op("index_select", kind=OpKind.INDEX_SELECT).targets(
        torch.index_select,
        torch.ops.aten.index_select.default,
        torch.ops.aten.index_select,
    ).build()
    registry.register_op("split_with_sizes", kind=OpKind.SPLIT_WITH_SIZES).targets(
        torch.ops.aten.split_with_sizes.default,
        torch.ops.aten.split_with_sizes,
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
    registry.register_op("native_layer_norm", kind=OpKind.LAYER_NORM).targets(
        torch.ops.aten.native_layer_norm,
        torch.ops.aten.native_layer_norm.default,
    ).build()
    registry.register_op(
        "native_layer_norm_backward", kind=OpKind.LAYER_NORM_BACKWARD
    ).targets(
        torch.ops.aten.native_layer_norm_backward,
        torch.ops.aten.native_layer_norm_backward.default,
    ).build()
    registry.register_op("native_group_norm", kind=OpKind.GROUP_NORM).targets(
        torch.ops.aten.native_group_norm,
        torch.ops.aten.native_group_norm.default,
    ).build()
    registry.register_op(
        "native_group_norm_backward", kind=OpKind.GROUP_NORM_BACKWARD
    ).targets(
        torch.ops.aten.native_group_norm_backward,
        torch.ops.aten.native_group_norm_backward.default,
    ).build()
    registry.register_op("_pdist_forward", kind=OpKind.PDIST).targets(
        torch.ops.aten._pdist_forward,
        torch.ops.aten._pdist_forward.default,
    ).build()
    registry.register_op("_cdist_forward", kind=OpKind.CDIST).targets(
        torch.ops.aten._cdist_forward,
        torch.ops.aten._cdist_forward.default,
    ).build()
    registry.register_op("nonzero", kind=OpKind.NONZERO).targets(
        torch.nonzero,
        torch.ops.aten.nonzero,
        torch.ops.aten.nonzero.default,
    ).build()
    registry.register_op("sort", kind=OpKind.SORT).targets(
        torch.sort,
        torch.ops.aten.sort.default,
        torch.ops.aten.sort,
    ).build()
    registry.register_op("native_dropout", kind=OpKind.DROPOUT).targets(
        torch.ops.aten.native_dropout,
        torch.ops.aten.native_dropout.default,
    ).build()
    registry.register_op("rand", kind=OpKind.RANDOM).targets(
        torch.ops.aten.rand,
        torch.ops.aten.rand.default,
    ).build()
    registry.register_op("randn", kind=OpKind.RANDOM).targets(
        torch.ops.aten.randn,
        torch.ops.aten.randn.default,
    ).build()
    registry.register_op("randperm", kind=OpKind.RANDPERM).targets(
        torch.ops.aten.randperm,
        torch.ops.aten.randperm.default,
    ).build()

    return registry.build()


def build_target_registry(
    supported_ops: dict[str, _OpSpec],
) -> dict[object, _TargetInfo]:
    return _base_target_registry(supported_ops)


__all__ = ["build_supported_ops", "build_target_registry"]
