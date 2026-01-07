from __future__ import annotations

from typing import Dict, List

from c_ref_backend.cffi_bindings import RefBackendError
from codegen_backend.emitters.arange import ArangeEmitter
from codegen_backend.emitters.base import KindEmitter, KindEmitterBase
from codegen_backend.emitters.elementwise import ElementwiseEmitter
from codegen_backend.emitters.flip import FlipEmitter
from codegen_backend.emitters.pad import PadEmitter
from codegen_backend.emitters.resize import ResizeEmitter
from codegen_backend.emitters.view import ViewEmitter
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import OpKind


class TemporaryEmitter(KindEmitterBase):
    def __init__(self, kind: OpKind) -> None:
        self._kind = kind

    def emit(self, req: KernelEmitRequest) -> List[str]:
        from codegen_backend import backend as backend_module

        kind = self._kind
        if kind == OpKind.EMPTY_STRIDED:
            return backend_module._write_empty_strided_kernel(
                req.node_index,
                req.op_spec,
                req.output_shape,
                req.dtype,
            )
        if kind == OpKind.DIAGONAL:
            input_shape = req.input_shapes[0]
            input_strides = req.input_strides[0]
            input_dtype = req.input_dtypes[0]
            return backend_module._write_diagonal_kernel(
                req.node_index,
                req.op_node,
                input_shape,
                input_strides,
                input_dtype,
                req.output_shape,
                req.output_strides,
                req.dtype,
            )
        if kind == OpKind.REDUCTION:
            if req.op_spec.name == "std":
                return backend_module._write_std_kernel(
                    req.node_index,
                    req.op_spec,
                    req.input_shapes[0],
                    req.input_strides[0],
                    req.output_shape,
                    req.output_strides,
                    req.reduction_dims or (),
                    bool(req.keepdim),
                    req.dtype,
                    unbiased=bool(req.params.get("unbiased", True)),
                )
            if req.op_spec.name == "var":
                return backend_module._write_var_kernel(
                    req.node_index,
                    req.op_spec,
                    req.input_shapes[0],
                    req.input_strides[0],
                    req.output_shape,
                    req.output_strides,
                    req.reduction_dims or (),
                    bool(req.keepdim),
                    req.dtype,
                    unbiased=bool(req.params.get("unbiased", True)),
                )
            if req.op_spec.name == "norm":
                return backend_module._write_norm_kernel(
                    req.node_index,
                    req.op_spec,
                    req.input_shapes[0],
                    req.input_strides[0],
                    req.output_shape,
                    req.output_strides,
                    req.reduction_dims or (),
                    bool(req.keepdim),
                    req.dtype,
                    p_value=float(req.params.get("p_value", 2.0)),
                )
            return backend_module._write_reduction_kernel(
                req.node_index,
                req.op_spec,
                req.input_shapes[0],
                req.input_strides[0],
                req.output_shape,
                req.output_strides,
                req.reduction_dims or (),
                bool(req.keepdim),
                req.dtype,
            )
        if kind == OpKind.ARG_REDUCTION:
            return backend_module._write_argminmax_kernel(
                req.node_index,
                req.op_spec,
                req.input_shapes[0],
                req.input_strides[0],
                req.output_shape,
                req.output_strides,
                req.reduction_dims or (),
                bool(req.keepdim),
                bool(req.params.get("reduce_all", False)),
                req.dtype,
            )
        if kind == OpKind.SOFTMAX:
            return backend_module._write_softmax_kernel(
                req.node_index,
                req.op_spec,
                req.input_shapes[0],
                req.input_strides[0],
                req.output_strides,
                int(req.params["dim"]),
                req.dtype,
            )
        if kind == OpKind.CUMSUM:
            return backend_module._write_cumsum_kernel(
                req.node_index,
                req.op_spec,
                req.input_shapes[0],
                req.input_strides[0],
                req.output_strides,
                int(req.params["dim"]),
                req.dtype,
                req.params["output_dtype"],
            )
        if kind == OpKind.EMBEDDING:
            return backend_module._write_embedding_kernel(
                req.node_index,
                req.op_spec,
                req.input_shapes[0],
                req.input_shapes[1],
                req.input_strides[0],
                req.input_strides[1],
                req.output_shape,
                req.output_strides,
                req.input_dtypes[1],
                req.dtype,
                padding_idx=int(req.params.get("padding_idx", -1)),
            )
        if kind == OpKind.EMBEDDING_BAG:
            return backend_module._write_embedding_bag_kernel(
                req.node_index,
                req.op_spec,
                req.input_shapes[0],
                req.input_shapes[1],
                req.input_shapes[2],
                req.input_strides[0],
                req.input_strides[1],
                req.input_strides[2],
                req.output_shape,
                req.output_strides,
                req.input_dtypes[1],
                req.input_dtypes[2],
                req.dtype,
                mode=int(req.params.get("mode", 0)),
                padding_idx=int(req.params.get("padding_idx", -1)),
                include_last_offset=bool(
                    req.params.get("include_last_offset", False)
                ),
            )
        if kind == OpKind.GATHER:
            return backend_module._write_gather_kernel(
                req.node_index,
                req.op_spec,
                req.input_shapes[0],
                req.input_shapes[1],
                req.input_strides[0],
                req.input_strides[1],
                req.output_shape,
                req.output_strides,
                req.input_dtypes[1],
                int(req.params["dim"]),
                req.dtype,
            )
        if kind == OpKind.CONCAT:
            return backend_module._write_concat_kernel(
                req.node_index,
                req.op_spec,
                req.input_shapes,
                req.input_strides,
                req.output_shape,
                req.output_strides,
                int(req.params.get("dim", 0)),
                req.dtype,
            )
        if kind == OpKind.POOL2D:
            return backend_module._write_pool2d_kernel(
                req.node_index,
                req.op_spec,
                req.input_shapes[0],
                req.output_shape,
                req.params["kernel_size"],
                req.params["stride"],
                req.params["padding"],
                req.params["dilation"],
                req.dtype,
                bool(req.params.get("ceil_mode", False)),
                bool(req.params.get("count_include_pad", False)),
                req.params.get("divisor_override"),
            )
        if kind == OpKind.POOL3D:
            return backend_module._write_pool3d_kernel(
                req.node_index,
                req.op_spec,
                req.input_shapes[0],
                req.output_shape,
                req.params["kernel_size"],
                req.params["stride"],
                req.params["padding"],
                req.params["dilation"],
                req.dtype,
                bool(req.params.get("ceil_mode", False)),
                bool(req.params.get("count_include_pad", False)),
                req.params.get("divisor_override"),
            )
        if kind == OpKind.POOL2D_BACKWARD:
            return backend_module._write_adaptive_avg_pool2d_backward_kernel(
                req.node_index,
                req.op_spec,
                req.input_shapes[0],
                req.input_shapes[1],
                req.output_shape,
                req.params["kernel_size"],
                req.params["stride"],
                req.dtype,
            )
        if kind == OpKind.POOL1D:
            return backend_module._write_pool1d_kernel(
                req.node_index,
                req.op_spec,
                req.input_shapes[0],
                req.output_shape,
                req.params["kernel_size"],
                req.params["stride"],
                req.params["padding"],
                req.params["dilation"],
                req.dtype,
                bool(req.params.get("ceil_mode", False)),
                bool(req.params.get("count_include_pad", False)),
                req.params.get("divisor_override"),
            )
        if kind == OpKind.COL2IM:
            return backend_module._write_col2im_kernel(
                req.node_index,
                req.op_spec,
                req.input_shapes[0],
                req.output_shape,
                req.params["output_size"],
                req.params["kernel_size"],
                req.params["dilation"],
                req.params["padding"],
                req.params["stride"],
                req.dtype,
                int(req.params.get("out_blocks_h", 1)),
                int(req.params.get("out_blocks_w", 1)),
            )
        if kind == OpKind.BATCH_NORM:
            return backend_module._write_batch_norm_kernel(
                req.node_index,
                req.op_spec,
                req.input_shapes[0],
                req.output_shape,
                req.dtype,
                float(req.params.get("eps", 1e-5)),
                bool(req.params.get("has_weight", False)),
                bool(req.params.get("has_bias", False)),
            )
        if kind == OpKind.PDIST:
            return backend_module._write_pdist_kernel(
                req.node_index,
                req.op_spec,
                req.input_shapes[0],
                req.output_shape,
                req.dtype,
            )
        if kind == OpKind.CDIST:
            return backend_module._write_cdist_kernel(
                req.node_index,
                req.op_spec,
                req.input_shapes[0],
                req.input_shapes[1],
                req.output_shape,
                req.dtype,
            )
        if kind == OpKind.CONV1D:
            return backend_module._write_conv1d_kernel(
                req.node_index,
                req.op_spec,
                req.input_shapes[0],
                req.input_shapes[1],
                req.output_shape,
                int(req.params.get("stride", 1)),
                int(req.params.get("padding", 0)),
                int(req.params.get("dilation", 1)),
                int(req.params.get("groups", 1)),
                req.dtype,
                bool(req.params.get("has_bias", False)),
            )
        if kind == OpKind.CONV2D:
            return backend_module._write_conv2d_kernel(
                req.node_index,
                req.op_spec,
                req.input_shapes[0],
                req.input_shapes[1],
                req.output_shape,
                bool(req.params.get("transposed", False)),
                req.params.get("stride", (1, 1)),
                req.params.get("padding", (0, 0)),
                req.params.get("dilation", (1, 1)),
                int(req.params.get("groups", 1)),
                req.dtype,
                bool(req.params.get("has_bias", False)),
            )
        if kind == OpKind.ADDMM:
            return backend_module._write_addmm_kernel(
                req.node_index,
                req.op_spec,
                req.input_shapes[0],
                req.output_shape,
                req.input_shapes[1],
                req.input_shapes[2],
                req.input_strides[0],
                req.input_strides[1],
                req.input_strides[2],
                req.output_strides,
                req.dtype,
                alpha=float(req.params.get("alpha", 1.0)),
                beta=float(req.params.get("beta", 1.0)),
            )
        if kind == OpKind.ADDBMM:
            return backend_module._write_addbmm_kernel(
                req.node_index,
                req.op_spec,
                req.input_shapes[0],
                req.output_shape,
                req.input_shapes[1],
                req.input_shapes[2],
                req.input_strides[0],
                req.input_strides[1],
                req.input_strides[2],
                req.output_strides,
                req.dtype,
                alpha=float(req.params.get("alpha", 1.0)),
                beta=float(req.params.get("beta", 1.0)),
            )
        if kind == OpKind.ADDMV:
            return backend_module._write_addmv_kernel(
                req.node_index,
                req.op_spec,
                req.input_shapes[0],
                req.output_shape,
                req.input_shapes[1],
                req.input_shapes[2],
                req.input_strides[0],
                req.input_strides[1],
                req.input_strides[2],
                req.output_strides,
                req.dtype,
                alpha=float(req.params.get("alpha", 1.0)),
                beta=float(req.params.get("beta", 1.0)),
            )
        if kind == OpKind.ADDR:
            return backend_module._write_addr_kernel(
                req.node_index,
                req.op_spec,
                req.input_shapes[0],
                req.output_shape,
                req.input_shapes[1],
                req.input_shapes[2],
                req.input_strides[0],
                req.input_strides[1],
                req.input_strides[2],
                req.output_strides,
                req.dtype,
                alpha=float(req.params.get("alpha", 1.0)),
                beta=float(req.params.get("beta", 1.0)),
            )
        if kind == OpKind.MATMUL:
            return backend_module._write_matmul_kernel(
                req.node_index,
                req.op_spec,
                req.input_shapes[0],
                req.input_shapes[1],
                req.input_strides[0],
                req.input_strides[1],
                req.dtype,
            )
        raise RefBackendError(f"Unsupported kernel kind: {kind.value}")


def build_kind_emitters() -> Dict[OpKind, KindEmitter]:
    elementwise_emitter = ElementwiseEmitter()
    elementwise_kinds = {
        OpKind.BINARY,
        OpKind.FILL,
        OpKind.UNARY,
        OpKind.WHERE,
    }
    kind_emitters: Dict[OpKind, KindEmitter] = {
        OpKind.ARANGE: ArangeEmitter(),
        OpKind.FLIP: FlipEmitter(),
        OpKind.PAD: PadEmitter(),
        OpKind.VIEW: ViewEmitter(),
        OpKind.RESIZE: ResizeEmitter(),
    }
    return {
        kind: (
            elementwise_emitter
            if kind in elementwise_kinds
            else kind_emitters.get(kind, TemporaryEmitter(kind))
        )
        for kind in OpKind
    }
