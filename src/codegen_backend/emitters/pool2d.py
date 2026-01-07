from __future__ import annotations

from typing import List, Sequence, Tuple

from c_ref_backend.cffi_bindings import RefBackendError
from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.base import KindEmitterBase, _format_array_suffix
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec
from codegen_backend.templates import get_template_env


def _write_pool2d_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    dtype: _CodegenDType,
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: int | None,
) -> List[str]:
    pool2d_template = get_template_env().get_template("pool2d_kernel.c.j2")
    batch, channels, in_h, in_w = input_shape
    out_h, out_w = output_shape[2], output_shape[3]
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
    rendered = pool2d_template.render(
        signature=signature,
        pool_kind=op_spec.name,
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
        pad_h=pad_h,
        pad_w=pad_w,
        dil_h=dil_h,
        dil_w=dil_w,
        c_type=dtype.c_type,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
        has_divisor_override=divisor_override is not None,
    )
    return rendered.strip().splitlines()


class Pool2dEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise RefBackendError("pool2d requires op spec and dtype")
        return _write_pool2d_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.output_shape,
            req.params["kernel_size"],
            req.params["stride"],
            req.params["padding"],
            req.params["dilation"],
            dtype,
            bool(req.params.get("ceil_mode", False)),
            bool(req.params.get("count_include_pad", False)),
            req.params.get("divisor_override"),
        )
