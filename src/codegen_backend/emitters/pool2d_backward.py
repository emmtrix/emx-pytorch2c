from __future__ import annotations

from typing import List, Sequence, Tuple

from c_ref_backend.cffi_bindings import RefBackendError
from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.base import KindEmitterBase, _format_array_suffix
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec
from codegen_backend.templates import get_template_env


def _write_adaptive_avg_pool2d_backward_kernel(
    node_index: int,
    op_spec: _OpSpec,
    grad_output_shape: Sequence[int],
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    dtype: _CodegenDType,
) -> List[str]:
    pool2d_template = get_template_env().get_template(
        "adaptive_avg_pool2d_backward_kernel.c.j2"
    )
    batch, channels, in_h, in_w = input_shape
    out_h, out_w = grad_output_shape[2], grad_output_shape[3]
    k_h, k_w = kernel_size
    stride_h, stride_w = stride
    grad_output_suffix = _format_array_suffix(grad_output_shape)
    input_suffix = _format_array_suffix(input_shape)
    output_suffix = _format_array_suffix(output_shape)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} grad_output{grad_output_suffix}, "
        f"const {dtype.c_type} input{input_suffix}, "
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    rendered = pool2d_template.render(
        signature=signature,
        batch=batch,
        channels=channels,
        in_h=in_h,
        in_w=in_w,
        out_h=out_h,
        out_w=out_w,
        k_h=k_h,
        k_w=k_w,
        stride_h=stride_h,
        stride_w=stride_w,
        c_type=dtype.c_type,
    )
    return rendered.strip().splitlines()


class Pool2dBackwardEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise RefBackendError("pool2d_backward requires op spec and dtype")
        return _write_adaptive_avg_pool2d_backward_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.input_shapes[1],
            req.output_shape,
            req.params["kernel_size"],
            req.params["stride"],
            dtype,
        )
