from __future__ import annotations

from typing import List, Sequence, Tuple

from c_ref_backend.cffi_bindings import RefBackendError
from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.base import KindEmitterBase, _format_array_suffix
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec
from codegen_backend.templates import get_template_env


def _write_col2im_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    output_size: Tuple[int, int],
    kernel_size: Tuple[int, int],
    dilation: Tuple[int, int],
    padding: Tuple[int, int],
    stride: Tuple[int, int],
    dtype: _CodegenDType,
    out_blocks_h: int,
    out_blocks_w: int,
) -> List[str]:
    col2im_template = get_template_env().get_template("col2im_kernel.c.j2")
    if len(output_shape) == 4:
        batch, channels, out_h, out_w = output_shape
        has_batch = True
    else:
        channels, out_h, out_w = output_shape
        batch = 1
        has_batch = False
    k_h, k_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    input_suffix = _format_array_suffix(input_shape)
    output_suffix = _format_array_suffix(output_shape)
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} input{input_suffix}, "
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    rendered = col2im_template.render(
        signature=signature,
        batch=batch,
        channels=channels,
        out_h=out_h,
        out_w=out_w,
        k_h=k_h,
        k_w=k_w,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dil_h=dil_h,
        dil_w=dil_w,
        c_type=dtype.c_type,
        out_blocks_h=out_blocks_h,
        out_blocks_w=out_blocks_w,
        has_batch=has_batch,
    )
    return rendered.strip().splitlines()


class Col2imEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise RefBackendError("col2im requires op spec and dtype")
        return _write_col2im_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.output_shape,
            req.params["output_size"],
            req.params["kernel_size"],
            req.params["dilation"],
            req.params["padding"],
            req.params["stride"],
            dtype,
            int(req.params.get("out_blocks_h", 1)),
            int(req.params.get("out_blocks_w", 1)),
        )
